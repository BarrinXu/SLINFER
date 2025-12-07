import asyncio
import time

from node import Node
from worker import Worker
from request_info import ReqTracker
from typing import Optional
import logging
import config

logger = logging.getLogger(__name__)


class Pool:
    def __init__(self, pool_type: str, nodes_config: dict[int, dict], base_rank, world_size, session, gateway_logs):

        assert pool_type in ['cpu', 'gpu']
        self.pool_type = pool_type
        self.nodes: dict[int, Node] = {}
        cur_rank = base_rank
        self.rank_cnt = 0
        for node_id, node_info in nodes_config.items():
            new_node = Node(
                node_type=pool_type,
                node_id=node_id,
                node_info=node_info,
                base_rank=cur_rank,
                world_size=world_size,
                session=session,
                gateway_logs=gateway_logs)
            self.nodes[node_id] = new_node
            cur_rank += new_node.rank_cnt
            self.rank_cnt += new_node.rank_cnt

    async def check_start_complete(self):
        for node in self.nodes.values():
            await node.check_start_complete()

    def count_allocated_nodes(self):
        res = 0
        for node in self.nodes.values():
            if node.is_allocated():
                res += 1
        return res

    def collect_kv_scale_logs(self):
        res = []
        for node in self.nodes.values():
            for worker in node.workers.values():
                res.extend(worker.past_lifecycle_kv_scale_log_list)
                worker.past_lifecycle_kv_scale_log_list = []
        return res

    def count_nodes_density(self):
        res = []
        for node in self.nodes.values():
            res.append(node.get_density())
        return res

    def count_node_batch(self):
        res = []
        for node in self.nodes.values():
            cur_node = []
            for worker in node.workers.values():
                cur_node.append(len(worker.running_requests))
            res.append(cur_node)
        return res

    def count_node_memory(self):
        res = []
        for node in self.nodes.values():
            cur_node = []
            for worker in node.workers.values():
                cur_node.append(worker.get_monitor_memory_detail())
            res.append(cur_node)
        return res

    def get_node_list_in_x_descent(self, metric):
        assert metric in ['compute', 'memory']
        node_list = []
        for node in self.nodes.values():
            if metric == 'compute':
                node_list.append((node, node.get_decode_time_slice()))
            elif metric == 'memory':
                node_list.append((node, node.cal_node_memory_budget_with_target_worker(None, None)))
        node_list.sort(key=lambda x: x[1], reverse=True)
        return [entry[0] for entry in node_list]

    def try_allocate_worker(self, request_tracker: ReqTracker):
        if self.pool_type == 'cpu' and (config.enable_cpu is False):
            return None
        if self.pool_type == 'cpu':
            sort_metric = 'compute'
        elif self.pool_type == 'gpu':
            sort_metric = 'memory'
        else:
            raise Exception
        node_list = self.get_node_list_in_x_descent(sort_metric)
        if not config.enable_defragmentation:
            # No defragmentation, consider load-balance, select lowest first.
            node_list.reverse()

        unallocated_node_list = []
        for node in node_list:
            if not node.is_allocated():
                node_list.remove(node)
                unallocated_node_list.append(node)
        node_list.extend(unallocated_node_list)
        for node in node_list:
            worker = node.try_allocate_worker(request_tracker)
            if worker is not None:
                return worker
        return None

    def get_worker(self, node_id, worker_id):
        return self.nodes[node_id].workers[worker_id]


class ModelManager:
    def __init__(self, pool_type, node_id, worker_id):
        self.pool_type = pool_type
        self.node_id = node_id
        self.worker_id = worker_id


class PoolManager:
    def __init__(self, pools_config, session):

        world_size = 0
        for pool_type in ['cpu', 'gpu']:
            for node_info in pools_config[pool_type].values():
                world_size += len(node_info['workers'])
        self.world_size = world_size
        self.logs = {}
        self.logs['node_schedule_times'] = []
        self.logs['shadow_validation_times'] = []
        base_rank = 0
        self.gpu_pool = Pool('gpu', pools_config['gpu'],
                             base_rank=base_rank, world_size=world_size, session=session, gateway_logs=self.logs)
        base_rank += self.gpu_pool.rank_cnt
        self.cpu_pool = Pool('cpu', pools_config['cpu'],
                             base_rank=base_rank, world_size=world_size, session=session, gateway_logs=self.logs)

        self.models_worker_list: dict[int, list[Worker]] = {i: [] for i in range(512)}
        # For serverlessllm
        self.models_worker_round_robin_id: dict[int, int] = {i: 0 for i in range(512)}

        self.requests_tracker: dict[str, ReqTracker] = {}


        self.under_logging = False

    def start_monitor_async(self):
        self.under_logging = True
        asyncio.create_task(self.log_node_usage())

    def end_monitor(self):
        self.under_logging = False
        self.logs['workers_kv_scale'] = self.gpu_pool.collect_kv_scale_logs()

    async def log_node_usage(self):
        monitor_interval = 1
        self.logs['node_usage'] = {'cpu': [], 'gpu': []}
        self.logs['node_density'] = {'cpu': [], 'gpu': []}
        self.logs['batch'] = {'cpu': [], 'gpu': []}
        self.logs['memory'] = {'cpu': [], 'gpu': []}
        self.logs['node_schedule_times'] = []
        self.logs['shadow_validation_times'] = []
        wake_up_time = time.perf_counter()
        while self.under_logging:
            self.logs['node_usage']['cpu'].append(self.cpu_pool.count_allocated_nodes())
            self.logs['node_usage']['gpu'].append(self.gpu_pool.count_allocated_nodes())
            self.logs['node_density']['cpu'].append(self.cpu_pool.count_nodes_density())
            self.logs['node_density']['gpu'].append(self.gpu_pool.count_nodes_density())
            if config.enable_detailed_logging:
                self.logs['batch']['cpu'].append(self.cpu_pool.count_node_batch())
                self.logs['batch']['gpu'].append(self.gpu_pool.count_node_batch())
                self.logs['memory']['cpu'].append(self.cpu_pool.count_node_memory())
                self.logs['memory']['gpu'].append(self.gpu_pool.count_node_memory())
            wake_up_time += monitor_interval
            await asyncio.sleep(wake_up_time - time.perf_counter())

    async def check_start_complete(self):
        for pool in [self.cpu_pool, self.gpu_pool]:
            await pool.check_start_complete()

    def get_pool_from_pooltype(self, pool_type):
        if pool_type == 'cpu':
            return self.cpu_pool
        elif pool_type == 'gpu':
            return self.gpu_pool
        else:
            raise Exception

    def allocate_worker_for_request(self, request_tracker: ReqTracker):
        if config.pool_priority == 'cpu':
            pools = [self.cpu_pool, self.gpu_pool]
        elif config.pool_priority == 'gpu':
            pools = [self.gpu_pool, self.cpu_pool]
        else:
            raise Exception
        for pool in pools:
            logger.debug(f'try allocate a worker for model-{request_tracker.model_id} in {pool.pool_type} pool')
            worker = pool.try_allocate_worker(request_tracker)
            if worker is not None:
                logger.debug(f'allocation in {pool.pool_type} success! info: {worker.get_info()}')
                return worker
            else:
                logger.debug('allocation failed!')
        logger.debug(f'Allocation in all pools failed! '
                     f'This request will be dropped! {request_tracker.request_id}')
        return None

    def activate_request(self, target_worker: Worker, request_tracker: ReqTracker):
        if config.system == 'serverlessllm':
            # update serverlessllm's round_robin
            now_id = -1
            for idx, worker in enumerate(self.models_worker_list[request_tracker.model_id]):
                if worker == target_worker:
                    now_id = idx
                    break
            assert now_id != -1
            self.models_worker_round_robin_id[request_tracker.model_id] = now_id + 1

        request_id = request_tracker.request_id
        self.requests_tracker[request_id] = request_tracker
        target_worker.activate_request(request_tracker)

    def get_node_from_worker(self, worker: Worker) -> Node:
        pool = self.get_pool_from_pooltype(worker.node_type)
        return pool.nodes[worker.node_id]

    def can_add_request_to_worker(self, request_tracker: ReqTracker):
        if config.system == 'serverlessllm':
            model_id = request_tracker.model_id
            now_choice = self.models_worker_round_robin_id[model_id]
            target_worker_list = self.models_worker_list[model_id]
            for offset in range(0, len(target_worker_list)):
                target_worker = target_worker_list[(now_choice + offset) % len(target_worker_list)]
                target_node = self.get_node_from_worker(target_worker)
                target_worker.shadow_add_request(request_tracker)
                success = target_node.can_add_request_to_worker(target_worker, request_tracker)
                target_worker.shadow_del_request(request_tracker)
                if success:
                    return target_worker
            return None

        target_worker_list = []
        for worker in self.models_worker_list[request_tracker.model_id]:
            target_worker_list.append((worker, len(worker.running_requests)))
        target_worker_list.sort(key=lambda x: x[1], reverse=True)
        if not config.enable_defragmentation:
            # No defragmentation, consider load-balance
            target_worker_list.sort(key=lambda x: x[1], reverse=False)
        for entry in target_worker_list:
            target_worker = entry[0]
            target_node = self.get_node_from_worker(target_worker)
            target_worker.shadow_add_request(request_tracker)
            success = target_node.can_add_request_to_worker(target_worker, request_tracker)
            target_worker.shadow_del_request(request_tracker)
            if success:
                return target_worker
        return None

    def can_add_request_to_worker_via_preemption(self, request_tracker: ReqTracker):
        if config.enable_PD:
            return None, None
        if not config.enable_defragmentation or (not config.enable_preempt):
            return None, None
        if config.system == 'serverlessllm':
            return None, None
        target_worker_list = []
        for worker in self.models_worker_list[request_tracker.model_id]:
            target_worker_list.append((worker, len(worker.running_requests)))
        target_worker_list.sort(key=lambda x: x[1], reverse=True)

        for entry in target_worker_list:
            target_worker = entry[0]
            preempted_worker = None
            target_node = self.get_node_from_worker(target_worker)

            target_worker.shadow_add_request(request_tracker)
            if target_node.node_type == 'gpu':
                is_memory_bound = target_node.can_add_request_to_worker(
                    target_worker, request_tracker, ignores=['memory'])
                # is_memory_bound indicates this worker can serve this request, if we can scale up its memory.
                if is_memory_bound:
                    # Next, We calculate the memory_gap, and select a preemption target.
                    preempted_worker = target_node.find_memory_preemption_with_target_worker(target_worker)
            elif target_node.node_type == 'cpu':
                is_decode_bound = target_node.can_add_request_to_worker(
                    target_worker, request_tracker, ignores=['decode'])
                if is_decode_bound:
                    preempted_worker = target_node.find_decode_preemption_with_target_worker(target_worker)
            target_worker.shadow_del_request(request_tracker)
            if preempted_worker is not None:
                return target_worker, preempted_worker

        return None, None

    def schedule_incoming_request(self, request_tracker: ReqTracker):
        # Careful!!! This function should be executed atomically!!!

        model_id = request_tracker.model_id
        assert model_id in self.models_worker_list

        target_worker: Optional[Worker] = None

        # 1. We check whether an existing worker can handle this request.
        # If Node-A has an ongoing-preempted worker-2, consider add request to worker-1.
        target_worker = self.can_add_request_to_worker(request_tracker)

        # 2. We check whether an existing GPU-worker can scale up kv-cache by preempting its neighbour
        if target_worker is None:
            target_worker, preempted_worker = self.can_add_request_to_worker_via_preemption(request_tracker)
            # We find a preemption strategy
            if (target_worker is not None) and (preempted_worker is not None):
                self.preempt_worker_async(preempted_worker)

        # 3. We allocate a new worker
        if target_worker is None:
            target_worker = self.allocate_worker_for_request(request_tracker)
            if target_worker is None:
                return False
            assert target_worker.model_id == -1
            assert target_worker.empty()
            target_worker.allocate_with_model(model_id)
            self.models_worker_list[model_id].append(target_worker)
            if target_worker.node_type == 'gpu':
                target_worker.register_load_action()

        assert target_worker is not None
        self.activate_request(target_worker, request_tracker)

        target_node = self.get_node_from_worker(target_worker)
        success, new_num_blocks = target_node.check_worker_need_kv_scale_up(target_worker)
        assert success
        if new_num_blocks != -1:
            target_worker.register_kv_scale_action(new_num_blocks)

        target_worker.fire_request_async(request_tracker)
        return True

    def preempt_worker_async(self, preempted_worker: Worker):
        logger.info(f'{preempted_worker.node_type}-{preempted_worker.node_id:02d}-{preempted_worker.worker_id:02d} '
                    f'is preempted!')
        # Early remove from the models_worker_list
        self.models_worker_list[preempted_worker.model_id].remove(preempted_worker)
        # Attention: The following three events should execute one by one.
        # Early offload, modify the memory budget
        if preempted_worker.node_type == 'gpu':
            preempted_worker.register_offload_action()
        # Evict requests and save the kv-cache
        preempted_worker.register_evict_requests_action()
        # Then, this worker has no requests. Set num_blocks to 0.
        if preempted_worker.node_type == 'gpu':
            preempted_worker.register_kv_scale_action(new_num_blocks=0)
        # After everything finished, this worker became available.
        # Todo. Should deallocate until action finish... This might can be optimized.
        asyncio.create_task(preempted_worker.deallocate())

    async def clean_a_worker_after_keep_alive(self, worker: Worker, keep_alive_time,
                                              cur_served_request_cnt, cur_action_version):
        await asyncio.sleep(keep_alive_time)
        # Worker receive new request
        if worker.served_request_cnt != cur_served_request_cnt or worker.action_version != cur_action_version:
            return
        logger.debug(f'clean {worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} due to keep-alive timeout')
        # Perform worker clean
        assert worker.allocated
        assert len(worker.running_requests) == 0
        # Early remove from the models_worker_list
        self.models_worker_list[worker.model_id].remove(worker)
        if worker.node_type == 'gpu':
            worker.register_offload_action()
            worker.register_kv_scale_action(new_num_blocks=0)
            # Todo. Should wait until action finish... This might can be optimized.
        await worker.deallocate()

    def delete_request(self, request_tracker: ReqTracker):
        self.requests_tracker.pop(request_tracker.request_id, None)
        if request_tracker.detach_from_worker:
            # This request has never been attached to a worker or has already been detached from a worker.
            # In such cases, we do not need to handle worker recycle.
            return

        pool = self.get_pool_from_pooltype(request_tracker.node_type)
        worker = pool.nodes[request_tracker.node_id].workers[request_tracker.worker_id]
        worker.delete_request_tracker(request_tracker)
        if len(worker.running_requests) == 0:
            keep_alive_time = config.keep_alive_time
            if worker.node_type == 'cpu':
                keep_alive_time = 0
            # This worker enters idle
            # Todo. Potential Bug. What happen if a worker is preempted, just after the last request has finish?
            worker.action_version += 1
            asyncio.create_task(
                self.clean_a_worker_after_keep_alive(worker=worker, keep_alive_time=keep_alive_time,
                                                     cur_served_request_cnt=worker.served_request_cnt,
                                                     cur_action_version=worker.action_version))

    def get_worker(self, worker_info):
        target_pool = self.get_pool_from_pooltype(worker_info['pool_type'])
        return target_pool.get_worker(worker_info['node_id'], worker_info['worker_id'])
