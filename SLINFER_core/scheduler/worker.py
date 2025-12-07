import asyncio
import subprocess
import time
import math
import aiohttp
from abc import ABC
from typing import Optional
from enum import Enum, auto
import shlex
import os
import json

import config
from request_info import ReqTracker
from power_estimator import estimate_duration
import logging

logger = logging.getLogger(__name__)


class WorkerHangingReleaseType(Enum):
    being_scheduled = auto()
    finished_action = auto()


class WorkerActionBase(ABC):
    pass


class WorkerKVScaleAction(WorkerActionBase):
    def __init__(self, worker_id, action_id, new_num_blocks):
        self.worker_id = worker_id
        self.action_id = action_id
        self.new_num_blocks = new_num_blocks


class WorkerEvictRequestsAction(WorkerActionBase):
    def __init__(self, worker_id, action_id):
        self.worker_id = worker_id
        self.action_id = action_id


class WorkerLoadAction(WorkerActionBase):
    def __init__(self, worker_id, action_id):
        self.worker_id = worker_id
        self.action_id = action_id


class WorkerOffloadAction(WorkerActionBase):
    def __init__(self, worker_id, action_id):
        self.worker_id = worker_id
        self.action_id = action_id


class WorkerGiveOutMemory(WorkerActionBase):
    def __init__(self, worker_id):
        self.worker_id = worker_id


class WorkerSleepAction(WorkerActionBase):
    def __init__(self, worker_id, action_id, wakeup_time):
        self.worker_id = worker_id
        self.action_id = action_id
        self.wakeup_time = wakeup_time


class Worker:
    def __init__(self, node_type, node_label, node_id, worker_id, worker_info: dict,
                 node_ip, base_port, rank, using_dist_scheduler,
                 worker_actions_queue: asyncio.Queue[WorkerActionBase], session: aiohttp.ClientSession):
        self.node_type = node_type
        self.node_label = node_label
        self.node_id = node_id
        self.worker_id = worker_id
        self.rank = rank
        self.using_dist_scheduler = using_dist_scheduler
        self.model_type: str = worker_info['model_type']
        self.model_memory_KB = worker_info['model_memory_GB'] * 1024 * 1024
        self.session = session

        self.kv_block_size = worker_info['block_size'][self.node_type]
        # Each block need how many memory space.
        self.per_kv_block_memory_KB = worker_info['per_token_kv_memory_KB'] * self.kv_block_size

        if self.node_type == 'cpu':
            # CPU has plenty memory of num_blocks
            self.cpu_kv_gb = worker_info['cpu_kv_gb']
            self.num_blocks_remote = math.floor(self.cpu_kv_gb * 1024 * 1024 / self.per_kv_block_memory_KB)
            logger.warning(f'{self.node_type}-{self.node_id}-{self.worker_id} num_block: {self.num_blocks_remote}')
        elif self.node_type == 'gpu':
            # We limit the GPU num_blocks
            self.num_blocks_remote = 0
        else:
            raise Exception
        self.num_blocks_remote_version = 0
        # num_blocks_local is the final num_block after scaling, num_blocks_remote is the real-time num_block
        self.num_blocks_local = self.num_blocks_remote
        self.num_blocks_local_version = 0

        self.action_version = 0
        self.dispatched_action_list: list[WorkerActionBase] = []
        self.is_performing_action = False
        self.actions_register_queue: asyncio.Queue[WorkerActionBase] = worker_actions_queue

        self.allocated = False
        # If a worker will be removed, it will not receive any new requests. This action can not be override or cancel.
        self.will_be_removed = False
        self.being_scheduled = False
        self.being_hanged = False
        self.idle_start_time = 0
        self.keep_alive_timeout = 1

        self.node_ip = node_ip
        self.port = base_port + worker_id
        self.running_requests: dict[int, ReqTracker] = {}

        self.time_slice = 0
        self.TPOT = config.TPOT
        self.model_id = -1
        # self.hanging_event: Optional[asyncio.Event] = None
        self.hanging_events = asyncio.Queue()
        # self.releasing_event: Optional[asyncio.Event] = asyncio.Event()

        # CPU worker hold model at start, while gpu worker not.
        self.hold_model_remote = (self.node_type == 'cpu')
        self.hold_model_remote_version = 0
        self.hold_model_local = (self.node_type == 'cpu')
        self.hold_model_local_version = 0
        self.is_loading_model = False
        self.loading_event = asyncio.Event()
        self.is_offloading_model = False
        self.offloading_event = asyncio.Event()
        self.model_path = config.models_path[self.model_type][self.node_label]

        self.start_complete = asyncio.Event()
        self.served_request_cnt = 0

        self.past_lifecycle_kv_scale_log_list = []
        self.cur_kv_scale_log_list = []
        self.allocate_time = 0

        # special variables for sllm+share
        self.bill_start_time = 0
        self.exec_total_duration = 0
        self.exec_start_time = 0

    def sllm_start_billing(self):
        self.bill_start_time = time.time()
        self.exec_total_duration = 0

    def sllm_handle_one_iteration_complete(self):
        cur_time = time.time()
        self.exec_total_duration += cur_time - self.exec_start_time
        nxt_wakeup_time = self.bill_start_time + self.exec_total_duration * config.sllm_max_shares
        if nxt_wakeup_time > cur_time:
            # We add a force sleep event...
            self.register_sleep_action(nxt_wakeup_time)

    def sllm_handle_one_iteration_start(self):
        self.exec_start_time = time.time()

    def under_prefill(self):
        for req in self.running_requests.values():
            if req.under_prefill:
                return True
        return False

    def get_recommend_quota_size(self):
        if self.under_prefill() or self.node_type == 'cpu':
            return 1
        if self.model_type.startswith('llama-3.2-3b'):
            return 4
        elif self.model_type.startswith('llama-2-7b'):
            return 2
        elif self.model_type.startswith('llama-3.1-8b'):
            return 2
        elif self.model_type.startswith('llama-2-13b'):
            return 1
        elif self.model_type.startswith('codestral-22b'):
            return 1
        else:
            raise Exception

    def can_be_scheduled(self):
        return (self.being_hanged and (not self.is_performing_action) and
                self.hold_model_remote and self.hold_model_local and
                self.num_blocks_remote > 0 and self.num_blocks_local > 0)

    async def deallocate(self):
        assert self.allocated
        assert self.model_id != -1
        while len(self.running_requests) > 0:
            await asyncio.sleep(0.1)
        lifetime = round(time.time() - self.allocate_time, 3)
        if self.node_type == 'gpu':
            while self.hold_model_remote or self.num_blocks_remote > 0:
                await asyncio.sleep(0.1)
            self.past_lifecycle_kv_scale_log_list.append((lifetime, self.cur_kv_scale_log_list))
            self.cur_kv_scale_log_list = []

        self.allocated = False
        self.model_id = -1

    def exist_loading_event(self):
        if (not self.hold_model_remote) or (not self.hold_model_local):
            return True
        for action in self.dispatched_action_list:
            if isinstance(action, WorkerLoadAction):
                return True
        return False

    def get_memory_footprint(self):
        memory_footprint_KB = self.per_kv_block_memory_KB * self.num_blocks_local
        if self.hold_model_local:
            memory_footprint_KB += self.model_memory_KB
        return memory_footprint_KB

    def get_monitor_memory_detail(self):
        model_memory = 0
        use_kv_memory = 0
        schedule_kv_memory = self.per_kv_block_memory_KB * self.num_blocks_local
        if self.hold_model_local:
            model_memory = self.model_memory_KB
        for req in self.running_requests.values():
            use_kv_memory += math.ceil(req.total_length() / self.kv_block_size) * self.per_kv_block_memory_KB
        return model_memory, use_kv_memory, schedule_kv_memory

    def get_kv_usage(self):
        num = 0
        for req in self.running_requests.values():
            num += math.ceil((req.total_length() + 512) / self.kv_block_size)
        return num

    def get_kv_minimal_required_blocks(self):
        minimal_required_blocks = self.get_kv_usage()
        if minimal_required_blocks > 0:
            minimal_required_blocks = max(minimal_required_blocks, math.ceil(config.minimal_tokens_per_instance / self.kv_block_size))
        return minimal_required_blocks

    def get_kv_recommended_blocks(self):
        recommended_blocks = math.ceil(self.get_kv_usage() * (1 + config.kv_scale_watermark))
        if recommended_blocks > 0:
            recommended_blocks = max(recommended_blocks, math.ceil(config.minimal_tokens_per_instance / self.kv_block_size))
        return recommended_blocks

    async def digest_dispatched_actions(self):
        assert self.is_performing_action
        while len(self.dispatched_action_list) > 0:
            action = self.dispatched_action_list[0]
            logger.debug(
                f'{self.node_type}-{self.node_id:02d}-{self.worker_id:02d} perform {action.__class__.__name__}')
            if isinstance(action, WorkerLoadAction):
                await self.perform_load_action()
            elif isinstance(action, WorkerOffloadAction):
                await self.perform_offload_action(action)
            elif isinstance(action, WorkerKVScaleAction):
                await self.perform_kv_scale_action(action)
            elif isinstance(action, WorkerEvictRequestsAction):
                await self.perform_evict_requests_action(action)
            elif isinstance(action, WorkerSleepAction):
                await self.perform_sleep_action(action)
            else:
                raise Exception
            # Late pop, make sure an ongoing action is still in the list.
            self.dispatched_action_list.pop(0)
        # Have already digested all pending actions...
        self.is_performing_action = False
        self.hanging_events.put_nowait(WorkerHangingReleaseType.finished_action)

    def check_whether_performing_dispatched_actions(self):
        if self.being_scheduled:
            return
        if self.is_performing_action or len(self.dispatched_action_list) == 0:
            return

        self.is_performing_action = True
        asyncio.create_task(self.digest_dispatched_actions())

    def check_whether_need_kv_scale_in(self):
        if config.system == 'serverlessllm':
            # serverlessllm does not consider kv scale-in.
            return
        if self.node_type == 'cpu':
            return
        recommend_blocks = self.get_kv_recommended_blocks()
        if recommend_blocks < self.num_blocks_local / (1 + config.kv_scale_watermark):
            self.register_kv_scale_action(new_num_blocks=recommend_blocks)

    def register_sleep_action(self, wakeup_time):
        self.action_version += 1
        # We directly add thie action to the dispatched queue, as it do not need center coordination.
        self.dispatched_action_list.append(WorkerSleepAction(self.worker_id, self.action_version, wakeup_time))

    async def perform_sleep_action(self, action: WorkerSleepAction):
        await asyncio.sleep(max(0, action.wakeup_time - time.time()))

    def register_evict_requests_action(self):
        # In this event, we only evict the requests.
        self.action_version += 1
        new_action = WorkerEvictRequestsAction(self.worker_id, self.action_version)
        for req in self.running_requests.values():
            assert not req.detach_from_worker
            req.detach_from_worker = True
        if self.using_dist_scheduler:
            assert self.node_type == 'cpu'
            asyncio.create_task(self.perform_evict_requests_action(new_action))
            return
        self.actions_register_queue.put_nowait(new_action)

    def dispatch_evict_requests_action(self, action: WorkerEvictRequestsAction):
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    async def perform_evict_requests_action(self, action: WorkerEvictRequestsAction):
        # Todo. Potential bug: a running request may be just finished or just fired...
        evict_requests = list(self.running_requests.values())
        await self.evict_requests_remote(evict_requests, save_kv=False)

        for request in evict_requests:
            if not request.check_finish():
                request.perform_evict(save_kv=False)
        self.running_requests.clear()

    def register_load_action(self):
        # If hold_model_local is True, we do not need to register load again.
        assert self.node_type == 'gpu'
        assert not self.hold_model_local

        self.action_version += 1

        self.hold_model_local = True
        self.hold_model_local_version = self.action_version
        self.actions_register_queue.put_nowait(WorkerLoadAction(self.worker_id, self.action_version))

    def dispatch_load_action(self, action: WorkerLoadAction):
        assert self.hold_model_local is True

        self.is_loading_model = True
        self.loading_event.clear()

        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    async def perform_load_action(self):
        if self.node_type == 'gpu':
            load_st = time.perf_counter()
            await self.trigger_remote_model_loading()
            load_ed = time.perf_counter()
            self.delay_requests_due_to_model_loading(load_ed - load_st)
        assert self.hold_model_remote is True
        self.is_loading_model = False
        self.loading_event.set()

    def register_offload_action(self):
        # If load_model_local is False, we do not need to register offload again.
        assert self.node_type == 'gpu'
        assert self.hold_model_local is True

        self.action_version += 1

        self.hold_model_local = False
        self.hold_model_local_version = self.action_version
        self.actions_register_queue.put_nowait(WorkerOffloadAction(self.worker_id, self.action_version))

    def dispatch_offload_action(self, action: WorkerOffloadAction):
        self.is_offloading_model = True
        self.offloading_event.clear()
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    async def perform_offload_action(self, action: WorkerOffloadAction):
        await self.trigger_remote_model_offloading(new_num_blocks=-1)
        self.is_offloading_model = False
        self.offloading_event.set()
        # A new load action has been committed, but not yet performed.
        if self.hold_model_remote_version > action.action_id:
            pass
        else:
            # Current action is offload, remote status will not be set early.
            assert self.hold_model_remote_version < action.action_id
            self.hold_model_remote = False
            self.hold_model_remote_version = action.action_id
            self.actions_register_queue.put_nowait(WorkerGiveOutMemory(self.worker_id))

    def register_kv_scale_action(self, new_num_blocks):
        assert self.node_type == 'gpu'
        self.action_version += 1
        self.num_blocks_local = new_num_blocks
        self.num_blocks_local_version = self.action_version
        self.actions_register_queue.put_nowait(WorkerKVScaleAction(self.worker_id, self.action_version, new_num_blocks))

    def dispatch_kv_scale_action(self, action: WorkerKVScaleAction):
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    async def perform_kv_scale_action(self, action: WorkerKVScaleAction):
        await self.scale_remote_kv(action.new_num_blocks)

        # A new scale-up action has been commited, but not yet performed
        if self.num_blocks_remote_version > action.action_id:
            pass
        # Current action is scale-up, remote value has been early commited.
        elif self.num_blocks_remote_version == action.action_id:
            assert self.num_blocks_remote == action.new_num_blocks
        # Current action is scale-down, optionally update num_blocks_remote
        else:
            assert self.num_blocks_remote >= action.new_num_blocks
            # Check whether exist newer action
            if self.num_blocks_local_version == action.action_id:
                self.num_blocks_remote = action.new_num_blocks
                self.num_blocks_remote_version = action.action_id
                self.actions_register_queue.put_nowait(WorkerGiveOutMemory(self.worker_id))

    def get_prefill_ddl_and_duration(self):
        prefill_duration = 0
        for req in self.running_requests.values():
            if req.under_prefill and (not req.detach_from_worker):
                prefill_duration += estimate_duration(self.model_type, self.node_type, self.node_label,
                                                      req.total_length(), -1, 'prefill')
        return self.get_next_token_ddl(), prefill_duration

    def get_next_token_ddl(self):
        ddl = 1e18
        for req in self.running_requests.values():
            if not req.detach_from_worker:
                ddl = min(ddl, req.expect_next_token_time())
        return ddl

    def get_info(self):
        return {'pool': self.node_type,
                'node': self.node_id,
                'worker': self.worker_id,
                'model': self.model_id}

    def get_total_token_num_and_batch_size(self):
        total_token_num = 0
        bs = 0
        for req in self.running_requests.values():
            if not req.detach_from_worker:
                bs += 1
                total_token_num += req.total_length()
        return total_token_num, bs

    def update_time_slice(self):
        if self.model_id >= 256:
            # This is a prefill instance
            self.time_slice = 0
            return self.time_slice
        total_token_num, batch_size = self.get_total_token_num_and_batch_size()
        if batch_size == 0:
            self.time_slice = 0
        else:
            self.time_slice = estimate_duration(self.model_type, self.node_type, self.node_label,
                                                total_token_num / batch_size, batch_size, 'decode') / self.TPOT
        return self.time_slice

    def activate_request(self, request_tracker: ReqTracker):
        request_id = request_tracker.request_id
        assert request_id not in self.running_requests
        self.served_request_cnt += 1
        self.running_requests[request_id] = request_tracker
        assert request_tracker.detach_from_worker
        request_tracker.detach_from_worker = False
        request_tracker.set_location(self.node_type, self.node_id, self.worker_id)

    def shadow_add_request(self, request_tracker: ReqTracker):
        request_id = request_tracker.request_id
        assert request_id not in self.running_requests
        self.running_requests[request_id] = request_tracker
        assert request_tracker.detach_from_worker
        request_tracker.detach_from_worker = False

    def shadow_del_request(self, request_tracker: ReqTracker):
        request_id = request_tracker.request_id
        assert request_id in self.running_requests
        self.running_requests.pop(request_id)
        assert not request_tracker.detach_from_worker
        request_tracker.detach_from_worker = True

    def delete_request_tracker(self, request_tracker: ReqTracker):
        assert not request_tracker.detach_from_worker
        request_tracker.detach_from_worker = True
        self.running_requests.pop(request_tracker.request_id)
        self.check_whether_need_kv_scale_in()
        if len(self.running_requests) == 0:
            self.idle_start_time = time.time()

    def empty(self):
        return (self.model_id == -1 and (not self.allocated) and len(self.running_requests) == 0 and
                (self.node_type == 'cpu' or
                 ((not self.hold_model_remote) and (not self.hold_model_local) and
                  (not self.is_loading_model) and (not self.is_offloading_model) and
                  self.num_blocks_local == 0 and self.num_blocks_remote == 0)))

    def can_allocate_with_model(self, target_model_type):
        return self.model_type == target_model_type and (not self.allocated)

    def delay_requests_due_to_model_loading(self, delay_time):
        # This is too optimistic. Request may join this worker in the middle of loading, with less delay_time.
        for req in self.running_requests.values():
            req.tolerate_time += delay_time
            if req.output_length == 0:
                req.TTFT_cold_start = True

    async def trigger_remote_model_loading(self):
        # Only GPU need loading...
        assert self.node_type == 'gpu'
        url = f'http://{self.node_ip}:{self.port}/load_model'
        async with self.session.post(url) as response:
            res = await response.json()
        assert res['result'] is True

    async def trigger_remote_model_offloading(self, new_num_blocks):
        assert self.node_type == 'gpu'
        url = f'http://{self.node_ip}:{self.port}/offload_model'
        async with self.session.post(url, json={'new_num_blocks': new_num_blocks}) as response:
            res = await response.json()
        assert res['result'] is True

    def allocate_with_model(self, model_id):
        assert not self.allocated
        assert self.model_id == -1
        self.allocated = True
        self.cur_kv_scale_log_list = []
        self.allocate_time = time.time()
        self.model_id = model_id

    async def scale_remote_kv(self, new_num_blocks):
        logger.debug(f'{self.node_type}-{self.node_id:02d}-{self.worker_id:02d} scale num_blocks to {new_num_blocks}')
        st = time.perf_counter()
        try:
            async with self.session.post(
                    f'http://{self.node_ip}:{self.port}/kv_scale',
                    json={'new_num_blocks': new_num_blocks}) as response:
                res = await response.json()
        except Exception as e:
            return
        ed = time.perf_counter()
        assert res['result'] is True
        self.cur_kv_scale_log_list.append(
            (res['old_num_blocks'] * self.per_kv_block_memory_KB,
             res['new_num_blocks'] * self.per_kv_block_memory_KB,
             round(ed - st, 3)))

    async def clear_worker(self):
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/clear_worker') as response:
            res = await response.json()
        assert res['result'] is True

    async def register_worker_info(self, worker_info: dict, dist_info: dict):
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/register_worker',
                json={'worker_info': worker_info, 'dist_info': dist_info}) as response:
            res = await response.json()
        assert res['result'] is True

    async def start_worker(self, worker_info: dict, dist_info: dict, start_lock: asyncio.Lock):
        # The model may be lazy loading, we fire a test request to activate it.
        async with start_lock:
            await self.clear_worker()
            if self.node_type == 'gpu':
                self.register_load_action()
                self.register_kv_scale_action(256)
                while (not self.hold_model_remote) or self.is_performing_action:
                    await asyncio.sleep(0.1)
                while self.num_blocks_remote != 256 or self.is_performing_action:
                    await asyncio.sleep(0.1)

            assert self.hold_model_local and self.hold_model_remote
            await self.fire_a_test_request()
            await asyncio.sleep(0.1)
            await self.register_worker_info(worker_info, dist_info)

            if self.node_type == 'gpu':
                self.register_offload_action()
                self.register_kv_scale_action(0)
                while self.hold_model_remote or self.num_blocks_remote > 0:
                    await asyncio.sleep(0.1)
            self.start_complete.set()

    async def fire_a_test_request(self):
        url = f'http://{self.node_ip}:{self.port}/v1/completions'
        data = {
            "model": self.model_path,
            "prompt": [1, 2, 3, 4],
            "min_tokens": 16,
            "max_tokens": 16,
            "stream": True
        }
        async with self.session.post(url, json=data) as response:
            await response.read()
            assert response.status == 200

    def fire_request_async(self, request_tracker: ReqTracker):
        request_tracker.fire_version += 1
        assert request_tracker.fire_task is None
        request_tracker.fire_task = asyncio.create_task(
            self.fire_request_sync(request_tracker, request_tracker.fire_version))

    async def fire_request_sync(self, request_tracker: ReqTracker, fire_ver):
        try:
            assert not request_tracker.check_finish()
            assert self.model_id == request_tracker.model_id

            if request_tracker.text is None:
                request_tracker.text = 'There are many countries in the world'
            prompt = request_tracker.text
            expect_length = request_tracker.expect_output_length - request_tracker.output_length
            url = f'http://{self.node_ip}:{self.port}/v1/completions'
            remote_request_id = f'{request_tracker.request_id}-{fire_ver}'
            if config.enable_PD and request_tracker.output_length == 0:
                # This is a prefill-only request
                expect_length = 1
            data = {
                "model": self.model_path,
                "prompt": prompt,
                "min_tokens": expect_length,
                "max_tokens": expect_length,
                'request_id': remote_request_id,
                "stream": True
            }
            logger.info(f'fire request {remote_request_id} to '
                        f'{request_tracker.node_type}-{request_tracker.node_id:02d}-{request_tracker.worker_id:02d}, '
                        f'cur_out: {request_tracker.output_length}, '
                        f'exp_out: {request_tracker.expect_output_length}')
            async with self.session.post(url, json=data) as response:
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line == '\n':
                            continue
                        if decoded_line[:12] == 'data: [DONE]':
                            continue
                        now_json = json.loads(decoded_line[6:])
                        request_tracker.receive_new_token(now_json['choices'][0]['text'])
                        request_tracker.check_finish()
        except asyncio.CancelledError:
            logger.info(f'{request_tracker.request_id} fire_ver {fire_ver} canceled')

    async def evict_requests_remote(self, requests_list: list[ReqTracker], save_kv: bool):
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/evict_requests',
                json={'request_id_list': [f'{req.request_id}-{req.fire_version}-0' for req in requests_list],
                      'save_kv': save_kv}) as response:
            res = await response.json()
        assert res['result'] is True

    async def perform_kv_send(self, dst_worker: "Worker", request_list: list[ReqTracker]):
        request_id_list = [request.request_id for request in request_list]
        transfer_config = {
            'src_rank': self.rank,
            'dst_rank': dst_worker.rank,
            'dst_socket_addr': dst_worker.node_ip,
            'dst_socket_port': dst_worker.port + 10000
        }
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/kv_send',
                json={'transfer_config': transfer_config,
                      'request_id_list': request_id_list}) as response:
            res = await response.json()
        assert res['result'] is True
