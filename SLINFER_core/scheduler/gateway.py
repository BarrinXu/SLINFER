import asyncio
import time
import json
import os
import logging

from pool import PoolManager
from worker import WorkerHangingReleaseType
from request_info import ReqTracker
import config
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import aiohttp
import subprocess
import shlex

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/ask_for_quota")
async def ask_for_quota(request: Request):
    body = await request.json()
    worker_info = body['worker_info']
    was_being_scheduled = body['was_being_scheduled']
    worker = pool_manager.get_worker(worker_info)
    node = pool_manager.get_node_from_worker(worker)
    logger.debug(f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} is asking for quota')

    assert not worker.being_hanged
    worker.being_hanged = True
    assert worker.being_scheduled == was_being_scheduled
    worker.being_scheduled = False
    while not worker.hanging_events.empty():
        worker.hanging_events.get_nowait()

    # if next(iter(worker.running_requests.values())).output_length == 2:
    #     pool_manager.preempt_worker_async(worker)

    # special handle for sllm+share
    if config.system == 'serverlessllm' and config.sllm_enable_sharing:
        if not was_being_scheduled:
            worker.sllm_start_billing()
        else:
            worker.sllm_handle_one_iteration_complete()

    worker.check_whether_performing_dispatched_actions()
    # re-schedule
    # Attention: When node.worker.can_be_scheduled() return True, we must node.run_schedule(). Otherwise would stuck.
    if was_being_scheduled or (worker.can_be_scheduled() and node.acquire_schedule_permission()):
        asyncio.create_task(node.schedule())

    # Wait for schedule for this worker
    while True:
        event_type = await worker.hanging_events.get()
        if event_type == WorkerHangingReleaseType.being_scheduled:
            logger.debug(f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} '
                         f'hanging is released due to being_scheduled')
            break
        elif event_type == WorkerHangingReleaseType.finished_action:
            logger.debug(f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} '
                         f'hanging is released due to action finish')
            if worker.can_be_scheduled() and node.acquire_schedule_permission():
                asyncio.create_task(node.schedule())
        else:
            raise Exception
    worker.being_hanged = False
    if config.system == 'serverlessllm' and config.sllm_enable_sharing:
        worker.sllm_handle_one_iteration_start()
    return JSONResponse({'quota': worker.get_recommend_quota_size()})


@app.post("/worker_go_idle")
async def worker_go_idle(request: Request):
    body = await request.json()
    worker_info = body['worker_info']
    worker = pool_manager.get_worker(worker_info)
    node = pool_manager.get_node_from_worker(worker)
    logger.debug(f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} is going idle')

    assert not worker.being_hanged
    assert worker.being_scheduled
    worker.being_scheduled = False
    worker.check_whether_performing_dispatched_actions()

    asyncio.create_task(node.schedule())
    return JSONResponse({'result': True})


@app.post("/v1/completions")
async def create_completion(request: Request):
    body = await request.json()
    # payload = body['payload']
    request_info = body['request_info']
    request_info['start_time'] = time.time()

    request_tracker = ReqTracker(request_info)
    logger.info(f'receive request: {request_tracker.get_info()}')
    while not request_tracker.check_finish():
        # Try to schedule
        while True:
            success = pool_manager.schedule_incoming_request(request_tracker)
            if success:
                break
            # Not success, wait for a while.
            await asyncio.sleep(0.25)
            if request_tracker.ddl_violate():
                # violate DDL, drop request
                logger.info(f'failed request: {request_tracker.get_info()}')
                pool_manager.delete_request(request_tracker)
                return JSONResponse({'result': False})
        # Start exec, wait for termination
        logger.info(f'schedule request: {request_tracker.get_info()}')
        await request_tracker.terminate_event.wait()
        request_tracker.terminate_event.clear()

        if config.enable_PD and request_tracker.model_id < 256 and request_tracker.output_length == 1:
            # This request has just finished prefill, we transform it to decode, and modify the worker-request mapping
            assert request_tracker.fire_task is not None
            request_tracker.fire_task.cancel()
            request_tracker.fire_task = None
            pool_manager.delete_request(request_tracker)
            request_tracker.transform_to_decode_only_request()

            # estimated for kv transfer
            per_token_kv_memory_in_kb = 128
            wait_time = request_tracker.total_length() * per_token_kv_memory_in_kb / 1024 / 1024 / 8
            request_tracker.tolerate_time += wait_time
            await asyncio.sleep(wait_time)

    e2e_metrics = request_tracker.get_e2e_metrics()
    pool_manager.delete_request(request_tracker)
    request_tracker.transform_to_normal_request()
    logger.info(f'complete request: {request_tracker.get_info()}')
    return JSONResponse({'result': True, 'e2e_metrics': e2e_metrics})


@app.post('/set_config')
async def set_config(request: Request):

    # We first set some config to default
    config.sllm_enable_sharing = False
    config.enable_cpu = True

    body = await request.json()
    for config_key in body.keys():
        if 'system' == config_key:
            new_system = body['system']
            assert new_system in ['serverlessllm', 'sota']
            config.system = new_system
        elif 'keep_alive_time' == config_key:
            new_keep_alive_time = body['keep_alive_time']
            assert new_keep_alive_time >= 0
            config.keep_alive_time = new_keep_alive_time
        elif 'pool_priority' == config_key:
            new_pool_priority = body['pool_priority']
            assert new_pool_priority in ['cpu', 'gpu']
            config.pool_priority = new_pool_priority
        elif 'enable_defragmentation' == config_key:
            new_enable_defragmentation = body['enable_defragmentation']
            assert new_enable_defragmentation in [True, False]
            config.enable_defragmentation = new_enable_defragmentation
        elif 'enable_preempt' == config_key:
            new_enable_preempt = body['enable_preempt']
            assert new_enable_preempt in [True, False]
            config.enable_preempt = new_enable_preempt
        elif 'enable_sharing' == config_key:
            new_enable_sharing = body['enable_sharing']
            assert new_enable_sharing in [True, False]
            config.enable_sharing = new_enable_sharing
        elif 'sllm_enable_sharing' == config_key:
            new_sllm_enable_sharing = body['sllm_enable_sharing']
            assert new_sllm_enable_sharing in [True, False]
            config.sllm_enable_sharing = new_sllm_enable_sharing
        elif 'enable_detailed_logging' == config_key:
            new_enable_detailed_logging = body['enable_detailed_logging']
            assert new_enable_detailed_logging in [True, False]
            config.enable_detailed_logging = new_enable_detailed_logging
        elif 'minimal_tokens_per_instance' == config_key:
            new_minimal_tokens_per_instance = body['minimal_tokens_per_instance']
            assert new_minimal_tokens_per_instance >= 0
            config.minimal_tokens_per_instance = new_minimal_tokens_per_instance
        elif 'kv_scale_watermark' == config_key:
            new_kv_scale_watermark = body['kv_scale_watermark']
            assert 0 <= new_kv_scale_watermark <= 1
            config.kv_scale_watermark = new_kv_scale_watermark
        elif 'enable_PD' == config_key:
            new_enable_PD = body['enable_PD']
            assert new_enable_PD in [True, False]
            config.enable_PD = new_enable_PD
        elif 'enable_cpu' == config_key:
            new_enable_cpu = body['enable_cpu']
            assert new_enable_cpu in [True, False]
            config.enable_cpu = new_enable_cpu
        else:
            raise Exception(f'unknown config key: {config_key}')

    for cpu_node in pool_manager.cpu_pool.nodes.values():
        cpu_node.update_dist_scheduler()
    return JSONResponse({'result': True})


@app.post('/get_config')
async def get_config(request: Request):
    return JSONResponse({'pools_config': config.pools_config,
                         'keep_alive_time': config.keep_alive_time,
                         'system': config.system,
                         'decode_preempt_metric': config.decode_preempt_metric,
                         'memory_preempt_metric': config.memory_preempt_metric,
                         'pool_priority': config.pool_priority,
                         'enable_defragmentation': config.enable_defragmentation,
                         'enable_preempt': config.enable_preempt,
                         'enable_sharing': config.enable_sharing,
                         'enable_detailed_logging': config.enable_detailed_logging,
                         'minimal_tokens_per_instance': config.minimal_tokens_per_instance,
                         'kv_scale_watermark': config.kv_scale_watermark,
                         'ddl_based_schedule': config.ddl_based_schedule})


@app.post('/start_monitor')
async def start_monitor(request: Request):
    pool_manager.start_monitor_async()
    return Response(status_code=200)


@app.post('/end_monitor')
async def end_monitor(request: Request):
    pool_manager.end_monitor()
    return JSONResponse(pool_manager.logs)


@asynccontextmanager
async def lifespan(app):
    logger.warning("Starting up...")
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024, force_close=True),
                                     timeout=aiohttp.ClientTimeout()) as session:
        global pool_manager
        pool_manager = PoolManager(config.pools_config, session)
        await pool_manager.check_start_complete()
        logger.warning('Start-up complete')
        yield
    logger.warning("Shutting down...")


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn

    # result = subprocess.run(['lsof', '-i', ':30000'], stdout=subprocess.PIPE, text=True)
    # for line in result.stdout.splitlines()[1:]:
    #     pid = int(line.split()[1])
    #     subprocess.run(['kill', '-9', str(pid)])
    #     print(f"Killed process {pid} on port 30000")
    pool_manager: PoolManager
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level='warning')
