import asyncio
import time
import json
import argparse
import os
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from enum import Enum, auto
import aiohttp

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()


class WorkerHangingReleaseType(Enum):
    being_scheduled = auto()
    finished_action = auto()


class MiniWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.being_hanged = False
        self.being_scheduled = False
        self.ddl = 0
        self.batch_num = 0
        self.hanging_events = asyncio.Queue()


class MiniNode:
    def __init__(self):
        self.workers: dict[int, MiniWorker] = {}
        self.have_schedule_permission = True
        self.info_version = 0
        self.enable_stupid_schedule = False
        self.ddl_based_schedule_config = {'enable_batch_aware': False,
                                          'safe_ddl_threshold': 5}

    def initialize_workers(self, worker_num):
        self.workers.clear()
        for worker_id in range(worker_num):
            self.workers[worker_id] = MiniWorker(worker_id)
        self.have_schedule_permission = True
        self.info_version = 0

    async def schedule(self):
        assert not self.have_schedule_permission
        ddl_min = 1e18
        target_worker = None

        can_be_scheduled_workers = []
        for worker in self.workers.values():
            if worker.being_hanged:
                assert not worker.being_scheduled
                can_be_scheduled_workers.append(worker)
                if worker.ddl < ddl_min:
                    ddl_min = worker.ddl
                    target_worker = worker
        if target_worker is None:
            self.have_schedule_permission = True
            return
        # We find a worker to schedule.
        if (self.ddl_based_schedule_config['enable_batch_aware'] and
                ddl_min > self.ddl_based_schedule_config['safe_ddl_threshold']):
            # All worker ddl are long enough, consider select the highest batch one.
            batch_num_min = 1000
            for worker in can_be_scheduled_workers:
                if worker.batch_num < batch_num_min:
                    batch_num_min = worker.batch_num
                    target_worker = worker

        self.have_schedule_permission = False
        target_worker.being_scheduled = True
        target_worker.hanging_events.put_nowait(WorkerHangingReleaseType.being_scheduled)

    def acquire_schedule_permission(self):
        if self.have_schedule_permission:
            self.have_schedule_permission = False
            return True
        else:
            return False


node = MiniNode()


@app.post("/ask_for_quota")
async def ask_for_quota(request: Request):
    body = await request.json()
    worker_info = body['worker_info']
    was_being_scheduled = body['was_being_scheduled']
    worker_id = worker_info['worker_id']
    worker = node.workers[worker_id]
    logger.debug(f'{worker.worker_id:02d} is asking for quota')
    if node.enable_stupid_schedule:
        logger.debug(f'{worker.worker_id:02d} '
                     f'hanging is released due to being_scheduled')
        return JSONResponse({'quota': 1})
    assert not worker.being_hanged
    worker.being_hanged = True
    assert worker.being_scheduled == was_being_scheduled
    worker.being_scheduled = False

    if was_being_scheduled or (node.acquire_schedule_permission()):
        asyncio.create_task(node.schedule())

    # Wait for schedule for this worker
    while True:
        event_type = await worker.hanging_events.get()
        if event_type == WorkerHangingReleaseType.being_scheduled:
            logger.debug(f'{worker.worker_id:02d} '
                         f'hanging is released due to being_scheduled')
            break
        else:
            raise Exception
    worker.being_hanged = False
    return JSONResponse({'quota': 1})


@app.post("/worker_go_idle")
async def worker_go_idle(request: Request):
    body = await request.json()
    worker_info = body['worker_info']
    worker_id = worker_info['worker_id']
    worker = node.workers[worker_id]
    logger.debug(f'{worker.worker_id:02d} is going idle')

    if node.enable_stupid_schedule:
        return JSONResponse({'result': True})

    assert not worker.being_hanged
    assert worker.being_scheduled
    worker.being_scheduled = False

    asyncio.create_task(node.schedule())
    return JSONResponse({'result': True})


# @app.post('/update_ddl')
# async def update_ddl(request: Request):
#     body = await request.json()
#     info_version = body['info_version']
#     if info_version > node.info_version:
#         node.info_version = info_version
#         workers_ddl: dict = body['workers_ddl']
#         workers_batch_num: dict = body['workers_batch_num']
#         logger.info(f'receive new ddl info: {workers_ddl}')
#         logger.info(f'receive new batch info: {workers_batch_num}')
#         for worker_id, ddl in workers_ddl.items():
#             node.workers[int(worker_id)].ddl = ddl
#         for worker_id, batch_num in workers_batch_num.items():
#             node.workers[int(worker_id)].batch_num = batch_num
#     return Response(status_code=200)


@app.websocket('/ws/update_ddl')
async def update_ddl(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            info_version = data['info_version']
            if info_version > node.info_version:
                node.info_version = info_version
                workers_ddl: dict = data['workers_ddl']
                workers_batch_num: dict = data['workers_batch_num']
                logger.info(f'receive new ddl info: {workers_ddl}, batch_num info: {workers_batch_num}')
                for worker_id, ddl in workers_ddl.items():
                    node.workers[int(worker_id)].ddl = ddl
                for worker_id, batch_num in workers_batch_num.items():
                    node.workers[int(worker_id)].batch_num = batch_num
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket communication: {e}")


@app.post('/init')
async def init(request: Request):
    body = await request.json()
    worker_num = body['worker_num']
    ddl_based_schedule_config: dict = body['ddl_based_schedule_config']
    is_sllm_share: bool = body['is_sllm_share']
    logger.info(f'initialize with {worker_num} workers')
    logger.info(f'update ddl_based_schedule_config with {ddl_based_schedule_config}')
    logger.info(f'is_sllm_share? {is_sllm_share}')
    if is_sllm_share:
        node.enable_stupid_schedule = True
    else:
        node.enable_stupid_schedule = False
    node.initialize_workers(worker_num)
    for k, v in ddl_based_schedule_config.items():
        assert k in node.ddl_based_schedule_config
        node.ddl_based_schedule_config[k] = v

@app.post('/update_system_config')
async def update_system_config(request: Request):
    body = await request.json()
    is_sllm_share: bool = body['is_sllm_share']
    logger.warning(f'update: is_sllm_share? {is_sllm_share}')
    if is_sllm_share:
        node.enable_stupid_schedule = True
    else:
        node.enable_stupid_schedule = False

@asynccontextmanager
async def lifespan(app):
    logger.info("Starting up...")
    logger.info('Start-up complete')
    yield
    logger.info("Shutting down...")


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level='warning')
