import math
import time
from typing import Optional
import json
import aiohttp
import requests
import websockets
import config
from worker import Worker, WorkerHangingReleaseType, WorkerActionBase, WorkerEvictRequestsAction, WorkerLoadAction, \
    WorkerOffloadAction, WorkerKVScaleAction, WorkerGiveOutMemory
from request_info import ReqTracker
from power_estimator import get_serverlessllm_concurrency
import logging
import asyncio

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, node_type, node_id, node_info: dict, base_rank, world_size, session: aiohttp.ClientSession, gateway_logs):
        self.node_type = node_type
        self.node_id = node_id
        self.node_label = node_info['node_label']
        self.workers_info = node_info['workers']
        self.worker_num = len(self.workers_info)
        self.node_memory_capacity_GB = node_info['node_memory_capacity_GB'] - self.worker_num * 1.5
        self.node_memory_capacity_KB = self.node_memory_capacity_GB * 1024 * 1024
        self.node_ip = node_info['node_ip']
        self.gateway_ip = node_info['gateway_ip']
        self.current_power = 0
        self.base_port = node_info['base_port']
        self.session = session

        self.rank_cnt = 0
        self.gateway_logs = gateway_logs

        self.workers: dict[int, Worker] = {}
        self.worker_actions_register_queue: asyncio.Queue[WorkerActionBase] = asyncio.Queue()
        cur_rank = base_rank

        self.using_dist_scheduler = node_info['dist_scheduler']
        self.scheduler_port = 7000
        if self.using_dist_scheduler:
            assert self.node_type == 'cpu'
            self.scheduler_port = self.base_port - 1
        if self.using_dist_scheduler:
            self.init_dist_scheduler()
            asyncio.create_task(self.periodic_update_ddl_to_dist_scheduler())
        asyncio.create_task(self.worker_action_monitor())

        start_lock = asyncio.Lock()

        for worker_id, worker_info in self.workers_info.items():
            self.workers[worker_id] = Worker(self.node_type, self.node_label, self.node_id, worker_id, worker_info,
                                             self.node_ip, self.base_port, cur_rank, self.using_dist_scheduler,
                                             self.worker_actions_register_queue, session)
            asyncio.create_task(self.workers[worker_id].start_worker(
                {
                    'pool_type': self.node_type,
                    'node_id': self.node_id,
                    'worker_id': worker_id,
                    'gateway_ip': self.gateway_ip,
                    'using_dist_scheduler': self.using_dist_scheduler,
                    'scheduler_port': self.scheduler_port,
                    'worker_ip': self.workers[worker_id].node_ip,
                    'worker_port': self.workers[worker_id].port
                }, {
                    'master_addr': self.gateway_ip,
                    'master_port': 30000,
                    'rank': cur_rank,
                    'world_size': world_size,
                    'socket_addr': self.workers[worker_id].node_ip,
                    'socket_port': self.workers[worker_id].port + 10000
                },
                start_lock))
            cur_rank += 1
            self.rank_cnt += 1

        self.have_schedule_permission = True
        self.last_schedule_time = -1

    def init_dist_scheduler(self):
        is_sllm_share = False
        if config.system == 'serverlessllm' and config.sllm_enable_sharing:
            is_sllm_share = True
        r = requests.post(f'http://{self.node_ip}:{self.scheduler_port}/init',
                          json={'worker_num': self.worker_num,
                                'ddl_based_schedule_config': config.ddl_based_schedule,
                                'is_sllm_share': is_sllm_share,})
        assert r.status_code == 200

    def update_dist_scheduler(self):
        is_sllm_share = False
        if config.system == 'serverlessllm' and config.sllm_enable_sharing:
            is_sllm_share = True
        r = requests.post(f'http://{self.node_ip}:{self.scheduler_port}/update_system_config',
                          json={'is_sllm_share': is_sllm_share, })
        assert r.status_code == 200

    async def periodic_update_ddl_to_dist_scheduler(self):
        info_version = 0
        uri = f'ws://{self.node_ip}:{self.scheduler_port}/ws/update_ddl'
        async with websockets.connect(uri) as websocket:
            while True:
                info_version += 1
                workers_ddl = {}
                workers_batch_num = {}
                now_time = time.time()
                for worker_id, worker in self.workers.items():
                    workers_batch_num[worker_id] = len(worker.running_requests)
                    if len(worker.running_requests) > 0:
                        workers_ddl[worker_id] = round(worker.get_next_token_ddl() - now_time, 3)
                    else:
                        # No running requests, set to 0. Normally this worker will not be scheduled. 0 is for quick act.
                        workers_ddl[worker_id] = 0
                message = json.dumps(
                    {'info_version': info_version, 'workers_ddl': workers_ddl, 'workers_batch_num': workers_batch_num})
                await websocket.send(message)
                await asyncio.sleep(0.1)

    # async def periodic_update_ddl_to_dist_scheduler(self):
    #     info_version = 0
    #     while True:
    #         info_version += 1
    #         workers_ddl = {}
    #         for worker_id, worker in self.workers.items():
    #             if len(worker.running_requests) > 0:
    #                 workers_ddl[worker_id] = worker.get_next_token_ddl()
    #             else:
    #                 # No running requests, we set 0. Normally this worker will not be scheduled. 0 is for info delay.
    #                 workers_ddl[worker_id] = 0
    #         async with self.session.post(f'http://{self.node_ip}:{self.scheduler_port}/update_ddl',
    #                                      json={'info_version': info_version,
    #                                            'workers_ddl': workers_ddl}) as response:
    #             await response.read()
    #         await asyncio.sleep(0.1)

    def is_allocated(self):
        for worker in self.workers.values():
            if worker.allocated:
                return True
        return False

    def get_density(self):
        res = 0
        for worker in self.workers.values():
            if worker.allocated:
                res += 1
        return res

    async def check_start_complete(self):
        for worker in self.workers.values():
            await worker.start_complete.wait()

    def acquire_schedule_permission(self):
        if self.have_schedule_permission:
            self.have_schedule_permission = False
            return True
        else:
            return False

    def try_to_commit_worker_load_action(self, target_worker: Worker, action_id):
        assert target_worker.hold_model_remote is False
        cur_memory_KB = self.get_cur_node_memory_footprint()
        if cur_memory_KB + target_worker.model_memory_KB <= self.node_memory_capacity_KB:
            # Early commit
            target_worker.hold_model_remote = True
            target_worker.hold_model_remote_version = action_id
            return True
        else:
            return False

    def try_to_commit_worker_kv_scale_action(self, target_worker: Worker, action_id, new_num_blocks):
        # scale-down can always commit
        if new_num_blocks <= target_worker.num_blocks_remote:
            return True
        # For scale-up, we check the budget
        cur_memory_KB = self.get_cur_node_memory_footprint()
        if cur_memory_KB + (
                new_num_blocks - target_worker.num_blocks_remote) * target_worker.per_kv_block_memory_KB <= self.node_memory_capacity_KB:
            # Early commit
            target_worker.num_blocks_remote = new_num_blocks
            target_worker.num_blocks_remote_version = action_id
            return True
        else:
            return False

    def try_commit_and_dispatch_action(self, action: WorkerActionBase) -> bool:
        target_worker = self.workers[action.worker_id]
        if isinstance(action, WorkerLoadAction):
            # We just drop this out-dated action
            if action.action_id != target_worker.hold_model_local_version:
                return True
            success = self.try_to_commit_worker_load_action(target_worker, action.action_id)
            if success:
                target_worker.dispatch_load_action(action)
                return True
            else:
                return False
        elif isinstance(action, WorkerOffloadAction):
            # We just drop this out-dated action
            if action.action_id != target_worker.hold_model_local_version:
                return True
            target_worker.dispatch_offload_action(action)
            return True
        elif isinstance(action, WorkerKVScaleAction):
            # We just drop this out-dated action
            if action.action_id != target_worker.num_blocks_local_version:
                return True
            success = self.try_to_commit_worker_kv_scale_action(target_worker, action.action_id, action.new_num_blocks)
            if success:
                target_worker.dispatch_kv_scale_action(action)
                return True
            else:
                return False
        elif isinstance(action, WorkerEvictRequestsAction):
            target_worker.dispatch_evict_requests_action(action)
            return True
        else:
            raise Exception

    async def worker_action_monitor(self):
        pending_actions: list[WorkerActionBase] = []
        while True:
            worker_action = await self.worker_actions_register_queue.get()
            logger.debug(f'{self.node_type}-{self.node_id:02d}-{worker_action.worker_id:02d} '
                         f'register {worker_action.__class__.__name__}')
            if isinstance(worker_action, WorkerGiveOutMemory):
                logger.debug(f'{self.node_type}-{self.node_id:02d} pending_actions: {len(pending_actions)}')
                while len(pending_actions) > 0:
                    success = self.try_commit_and_dispatch_action(pending_actions[0])
                    if success:
                        logger.debug(f'{self.node_type}-{self.node_id:02d}-{pending_actions[0].worker_id:02d} '
                                     f'commit {pending_actions[0].__class__.__name__}.')
                        pending_actions.pop(0)
                    else:
                        break
            else:
                success = self.try_commit_and_dispatch_action(worker_action)
                if not success:
                    logger.debug(f'{self.node_type}-{self.node_id:02d}-{worker_action.worker_id:02d} '
                                 f'move {worker_action.__class__.__name__} to pending list.')
                    pending_actions.append(worker_action)
                else:
                    logger.debug(f'{self.node_type}-{self.node_id:02d}-{worker_action.worker_id:02d} '
                                 f'commit {worker_action.__class__.__name__}.')

    async def schedule(self):
        # Todo. Priority schedule for those low-batch req.
        logger.debug(f'node schedule start: {self.node_type}-{self.node_id:02d}')
        # When exec this function, the schedule_permission has already been taken by the caller function.
        assert not self.have_schedule_permission

        ddl_min = 1e18
        target_worker = None
        now_time = time.time()

        can_be_scheduled_workers = []
        for worker in self.workers.values():
            # A worker might be exec_status, but not being hanged. E.g., We just fire a request to a worker.
            if worker.can_be_scheduled():
                assert not worker.being_scheduled
                can_be_scheduled_workers.append(worker)
                ddl_cur = worker.get_next_token_ddl() - now_time
                logger.debug(f'{self.node_type}-{self.node_id:02d}-worker{worker.worker_id:02d} '
                             f'ddl: {ddl_cur:.2f} s')
                if ddl_cur < ddl_min:
                    ddl_min = ddl_cur
                    target_worker = worker
        if config.enable_detailed_logging:
            self.gateway_logs['node_schedule_times'].append(time.time() - now_time)

        # We do not find a worker to schedule
        if target_worker is None:
            self.have_schedule_permission = True
            logger.debug(f'node schedule end: {self.node_type}-{self.node_id:02d}: None')
            return
        # We find a worker to schedule
        if (config.ddl_based_schedule['enable_batch_aware'] and
                ddl_min > config.ddl_based_schedule['safe_ddl_threshold']):
            # All worker ddl are long enough, consider select the highest batch one.
            batch_num_max = 0
            for worker in can_be_scheduled_workers:
                if len(worker.running_requests) > batch_num_max:
                    batch_num_max = len(worker.running_requests)
                    target_worker = worker

        self.have_schedule_permission = False
        self.last_schedule_time = now_time
        logger.debug(f'node schedule end: {self.node_type}-{self.node_id:02d}: '
                     f'schedule worker-{target_worker.worker_id:02d}')
        target_worker.being_scheduled = True
        # target_worker.releasing_event.set()
        await target_worker.hanging_events.put(WorkerHangingReleaseType.being_scheduled)

    def check_future_workload(self, workers_ddl_time_pair: list):
        workers_ddl_time_pair.sort()
        cur_time = time.time()
        for ddl_time_pair in workers_ddl_time_pair:
            cur_time += ddl_time_pair[1]
            if cur_time > ddl_time_pair[0]:
                return False
        return True

    def get_cur_node_memory_footprint(self):
        memory_KB = 0
        for worker in self.workers.values():
            if worker.hold_model_remote:
                memory_KB += worker.model_memory_KB
            memory_KB += worker.num_blocks_remote * worker.per_kv_block_memory_KB
        return memory_KB

    def cal_node_memory_budget_with_target_worker(self, target_worker: Optional[Worker], new_num_blocks: Optional[int]):
        memory_budget_KB = 0
        for worker in self.workers.values():
            if worker.hold_model_local or worker == target_worker:
                # target worker currently may not hold model, but will hold in the near future
                memory_budget_KB += worker.model_memory_KB
            if worker != target_worker:
                memory_budget_KB += worker.num_blocks_local * worker.per_kv_block_memory_KB
            else:
                memory_budget_KB += max(worker.num_blocks_local, new_num_blocks) * worker.per_kv_block_memory_KB
        return memory_budget_KB

    def find_decode_preemption_with_target_worker(self, target_worker: Worker):
        assert self.node_type == 'cpu'
        assert config.decode_preempt_metric in ['batch', 'compute']
        decode_time_slice_gap = self.get_decode_time_slice() - 1
        target_time_slice = target_worker.update_time_slice()
        # >1 indicates this worker has exceeded the node limit.
        if target_time_slice > 1:
            return None
        preemption_candidates = []

        if config.decode_preempt_metric == 'compute':
            for worker in self.workers.values():
                if worker == target_worker:
                    continue
                this_time_slice = worker.update_time_slice()
                if 0 < this_time_slice <= target_time_slice:
                    preemption_candidates.append((worker, this_time_slice))

            preemption_candidates.sort(key=lambda x: x[1])
            for entry in preemption_candidates:
                preempted_worker = entry[0]
                this_time_slice = entry[1]
                if this_time_slice >= decode_time_slice_gap:
                    return preempted_worker
        elif config.decode_preempt_metric == 'batch':
            for worker in self.workers.values():
                if worker == target_worker:
                    continue
                if 0 < len(worker.running_requests) <= len(target_worker.running_requests):
                    preemption_candidates.append((worker, len(worker.running_requests)))

            preemption_candidates.sort(key=lambda x: x[1])
            for entry in preemption_candidates:
                preempted_worker = entry[0]
                if preempted_worker.update_time_slice() >= decode_time_slice_gap:
                    return preempted_worker
        return None

    def find_memory_preemption_with_target_worker(self, target_worker: Worker):
        assert self.node_type == 'gpu'
        assert config.memory_preempt_metric in ['batch', 'memory']
        minimal_required_blocks = target_worker.get_kv_minimal_required_blocks()
        memory_gap_KB = self.cal_node_memory_budget_with_target_worker(
            target_worker, minimal_required_blocks) - self.node_memory_capacity_KB
        target_batch_degree = len(target_worker.running_requests)

        preemption_candidates = []
        for worker in self.workers.values():
            if worker == target_worker:
                continue
            if worker.hold_model_local and len(worker.running_requests) <= target_batch_degree:
                # Attention: hold_model is True, this worker may be loading or offloading worker...
                preemption_candidates.append((worker, len(worker.running_requests)))

        preemption_candidates.sort(key=lambda x: x[1])
        for entry in preemption_candidates:
            preempted_worker = entry[0]
            if preempted_worker.get_memory_footprint() >= memory_gap_KB:
                return preempted_worker
        return None

    def check_worker_need_kv_scale_up(self, target_worker: Worker):
        minimal_required_blocks = target_worker.get_kv_minimal_required_blocks()
        # Check pass, no need to scale-up or load_model
        if (minimal_required_blocks <= target_worker.num_blocks_local and
                (target_worker.node_type == 'cpu' or target_worker.hold_model_local)):
            return True, -1
        # need to scale-up
        else:
            # Check fail, CPU cannot scale.
            if target_worker.node_type == 'cpu':
                return False, -1
            else:
                # Special case for sllm+share
                if config.system == 'serverlessllm' and config.sllm_enable_sharing:
                    # For sllm, we do not care memory budget, as it relies on concurrency to control the memory usage.
                    # Always set a fixed block_num
                    return True, math.floor((self.node_memory_capacity_KB / config.sllm_max_shares -
                                  target_worker.model_memory_KB) / target_worker.per_kv_block_memory_KB)
                
                minimal_node_budget_KB = self.cal_node_memory_budget_with_target_worker(target_worker,
                                                                                        minimal_required_blocks)
                if minimal_node_budget_KB > self.node_memory_capacity_KB:
                    return False, -1
                else:
                    if config.system == 'serverlessllm':
                        # serverlessllm try to allocate the whole GPU as the KV.
                        recommend_blocks = 1000000
                    else:
                        recommend_blocks = target_worker.get_kv_recommended_blocks()
                    return True, min(recommend_blocks, minimal_required_blocks + math.floor(
                        (self.node_memory_capacity_KB - minimal_node_budget_KB) / target_worker.per_kv_block_memory_KB))

    def get_decode_time_slice(self):
        occupied_time_slice = 0
        for worker in self.workers.values():
            # Todo. This can be optimized...
            occupied_time_slice += worker.update_time_slice()
        return occupied_time_slice

    def check_decode_time_slice(self):
        return self.get_decode_time_slice() <= 1

    def check_prefill_DDLs_with_target_worker(self, target_worker: Worker):
        workers_ddl_time_pair = []
        for worker in self.workers.values():
            # We ignore TTFT for GPU-workers suffering model-loading
            if self.node_type == 'cpu' or (not worker.exist_loading_event()):
                prefill_ddl, prefill_duration = worker.get_prefill_ddl_and_duration()
                if prefill_duration > 0:
                    if worker == target_worker:
                        workers_ddl_time_pair.append((prefill_ddl, prefill_duration, 1))
                    else:
                        workers_ddl_time_pair.append((prefill_ddl, prefill_duration, 0))
                else:
                    if config.enable_PD:
                        if worker == target_worker:
                            workers_ddl_time_pair.append((worker.get_next_token_ddl(), worker.update_time_slice() * config.TPOT, 1))
                        else:
                            workers_ddl_time_pair.append((worker.get_next_token_ddl(), worker.update_time_slice() * config.TPOT, 0))
        workers_ddl_time_pair.sort()

        # When using dist_scheduler, we do not know last schedule time...
        if self.have_schedule_permission or self.using_dist_scheduler:
            cur_time = time.time()
        else:
            cur_time = self.last_schedule_time
        enter_influence_zone = False
        for ddl_time_pair in workers_ddl_time_pair:
            # Only consider if target_worker will influence other DDLs. Previous violations are ignored.
            if ddl_time_pair[2] == 1:
                enter_influence_zone = True
            cur_time += ddl_time_pair[1]
            if enter_influence_zone and cur_time > ddl_time_pair[0]:
                return False
        return True

    def can_add_request_to_worker(self, target_worker: Worker, request_tracker: ReqTracker, ignores=None) -> bool:
        if ignores is None:
            ignores = []
        if config.system == 'serverlessllm':
            # serverlessllm should only consider concurrency metric.
            assert len(ignores) == 0
            sllm_max_shares = 1
            if config.sllm_enable_sharing:
                sllm_max_shares = config.sllm_max_shares
            return len(target_worker.running_requests) <= get_serverlessllm_concurrency(
                target_worker.model_type, target_worker.node_type, target_worker.node_label, sllm_max_shares)

        logger.debug(f'Checking can add request {request_tracker.request_id} to worker {target_worker.get_info()}...')

        if 'memory' in ignores:
            logger.debug('Ignore worker memory check..')
        else:
            logger.debug('Checking worker memory usage...')
            success, new_num_blocks = self.check_worker_need_kv_scale_up(target_worker)
            if success:
                logger.debug('Memory check pass.')
            else:
                logger.debug("Memory check failed!")
                return False

        if 'decode' in ignores:
            logger.debug('Ignore decode check...')
        else:
            logger.debug('Checking decode time-slice...')
            if self.check_decode_time_slice():
                logger.debug('Decode check pass.')
            else:
                logger.debug('Decode check failed!')
                return False

        logger.debug('Checking prefill DDLs...')
        start_time = time.time()
        shadow_validation_pass = self.check_prefill_DDLs_with_target_worker(target_worker)
        if config.enable_detailed_logging:
            self.gateway_logs['shadow_validation_times'].append(time.time() - start_time)
        if shadow_validation_pass:
            logger.debug('Prefill check pass. Request can add to worker!')
            return True
        else:
            logger.debug('Prefill check failed!')
            return False

    def try_allocate_worker(self, request_tracker: ReqTracker):
        if config.system == 'serverlessllm' or (not config.enable_sharing):
            # serverlessllm cannot allocate two worker in one node.
            allocate_num = 0
            hold_13b_model = False
            for worker in self.workers.values():
                if worker.allocated:
                    allocate_num += 1
                    if worker.model_type == 'llama-2-13b':
                        hold_13b_model = True
            if config.system == 'serverlessllm' and config.sllm_enable_sharing:
                if allocate_num >= config.sllm_max_shares:
                    return None
                if self.node_type == 'cpu':
                    # For sllm's CPU node, it can only hold one 13B model.
                    if hold_13b_model or (allocate_num >= 1 and request_tracker.model_type == 'llama-2-13b'):
                        return None
            else:
                if allocate_num > 0:
                    return None

        for target_worker in self.workers.values():
            if target_worker.can_allocate_with_model(target_model_type=request_tracker.model_type):
                # This worker should be empty, with no running requests.
                assert target_worker.empty()
                target_worker.shadow_add_request(request_tracker)
                success = self.can_add_request_to_worker(target_worker, request_tracker)
                target_worker.shadow_del_request(request_tracker)
                if success:
                    return target_worker
                else:
                    return None
        return None
