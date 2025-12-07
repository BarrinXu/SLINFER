"""CacheEngine class for managing the KV cache."""
import gc
import time
from typing import List

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        st = time.time()
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        ed = time.time()
        logger.info(f'gpu kv-cache allocation duration: {ed - st:.3f} s')
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        logger.info(f'allocating {device} kv-cache, num_attention_layers: {self.num_attention_layers}')
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        logger.info(f'kv_cache_shape per layer: {kv_cache_shape}')
        kv_total_size = self.num_attention_layers * 2
        for length in kv_cache_shape:
            kv_total_size *= length
        logger.info(f'kv_cache_size total: {kv_total_size / 1024 / 1024 / 1024:.3f} GB')
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def scale_kv_cache(self, new_num_blocks, src_dst_list):
        logger.warning('scale physical kv-cache start')
        assert self.device_config.device_type == 'cuda'
        logger.warning(f'original num_blocks {self.num_gpu_blocks}, new num_blocks {new_num_blocks}')
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            new_num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        src_list = [entry[0] for entry in src_dst_list]
        dst_list = [entry[1] for entry in src_dst_list]
        for i in range(self.num_attention_layers):
            new_tensor = torch.empty(kv_cache_shape, dtype=self.dtype, device='cuda')
            old_tensor = self.gpu_cache[i]
            if new_num_blocks < self.num_gpu_blocks:
                new_tensor[:, dst_list, :, :, :] = old_tensor[:, src_list, :, :, :]
                # for src_idx, dst_idx in src_dst_list:
                #     new_tensor[:, dst_idx, :, :, :] = old_tensor[:, src_idx, :, :, :]
            else:
                assert len(src_dst_list) == 0
                new_tensor[:, :self.num_gpu_blocks, :, :, :] = old_tensor
            self.gpu_cache[i] = new_tensor
            del old_tensor
            torch.cuda.empty_cache()
        self.num_gpu_blocks = self.cache_config.num_gpu_blocks = new_num_blocks
        logger.warning('scale physical kv-cache end')

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
