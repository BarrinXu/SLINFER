import socket
from typing import Optional
from vllm.logger import init_logger
import torch
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import threading
import asyncio
import aiohttp

from vllm.core.kv_transfer import kv_transfer_manager

logger = init_logger(__name__)


class KVInfo:
    def __init__(self, data_config, tensor):
        self.data_config: dict = data_config
        self.tensor: torch.Tensor = tensor


class KVManager:
    def __init__(self):
        self.requests: dict[str, KVInfo] = {}
        self.send_sock: Optional[socket.socket] = None
        self.receive_sock: Optional[socket.socket] = None

        self.transfer_process: Optional[mp.Process] = None
        self.receive_thread: Optional[threading.Thread] = None
        self.logger_thread: Optional[threading.Thread] = None

        self.send_queue: Optional[mp.Queue] = None
        self.receive_queue: Optional[mp.Queue] = None

        self.logger_queue: Optional[mp.Queue] = None

        self.worker_info: Optional[dict] = None

    def register_worker(self, worker_info):
        self.worker_info = worker_info

    def init_gloo(self, gloo_config):
        if self.transfer_process is not None:
            self.transfer_process.kill()

        ctx = mp.get_context('spawn')
        self.send_queue = ctx.Queue()
        self.receive_queue = ctx.Queue()
        self.logger_queue = ctx.Queue()

        self.receive_thread = threading.Thread(target=self.receive_kv)
        self.receive_thread.start()

        self.logger_thread = threading.Thread(target=self.print_logger)
        self.logger_thread.start()

        self.transfer_process = ctx.Process(
            target=kv_transfer_manager,
            args=(gloo_config, self.send_queue, self.receive_queue, self.logger_queue),
            daemon=True)
        self.transfer_process.start()

    async def post_migration_complete(self, request_id_list):
        url = f'http://{self.worker_info["gateway_ip"]}:7000/migration_complete'
        headers = {"Content-Type": "application/json"}
        data = {'request_id_list': request_id_list}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                res = await response.json()
        assert res['result'] is True

    def receive_kv(self):
        while True:
            data_config = self.receive_queue.get()
            logger.warning(f'main process receive data_config')
            tensor = self.receive_queue.get()
            logger.warning(f'main process receive tensor')
            request_id = data_config['request_id']
            self.requests[request_id] = KVInfo(data_config, tensor)
            logger.warning(f'kv_tensor receive success, info: {data_config}')
            asyncio.run(self.post_migration_complete(request_id_list=[request_id]))

    def print_logger(self):
        while True:
            output = self.logger_queue.get()
            logger.warning('From kv_manager_subprocess: ' + output)

    def have_request_kv_cache(self, request_id):
        return request_id in self.requests

    def add_request(self, data_config, tensor):
        request_id = data_config['request_id']
        assert request_id not in self.requests
        self.requests[request_id] = KVInfo(data_config, tensor)

    def pop_request(self, request_id):
        assert request_id in self.requests
        return self.requests.pop(request_id)

    def send_kv(self, request_id_list: list, transfer_config: dict):
        for request_id in request_id_list:
            assert request_id in self.requests
            kv_info = self.requests.pop(request_id)
            data_config = kv_info.data_config
            tensor = kv_info.tensor
            assert tensor.is_shared()
            data_config.update(transfer_config)
            self.send_queue.put(data_config)
            self.send_queue.put(tensor)


kv_manager = KVManager()
