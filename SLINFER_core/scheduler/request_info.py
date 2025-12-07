import asyncio
import time
import logging
from typing import Optional
import config

input_table = list(range(1, 4097)) * 8
logger = logging.getLogger(__name__)


class ReqTracker:
    def __init__(self, request_info: dict):
        self.start_time = request_info['start_time']

        self.request_id = request_info['request_id']
        self.model_id = request_info['model_id']
        self.model_type = request_info['model_type']

        self.input_length = request_info['input_length']
        self.expect_output_length = request_info['expect_output_length']
        self.output_length = 0
        self.text = input_table[:self.input_length]
        self.under_prefill = True

        self.first_token_time = -1
        self.finish_time = -1
        self.TPOT_SLO = config.TPOT
        self.TTFT_SLO = min(max(config.TTFT_baseline, self.input_length / 512 * 0.95), config.TTFT_max_threshold)

        self.TTFT_cold_start = False

        self.tolerate_time = 0

        self.node_type = ''
        self.node_id = -1
        self.worker_id = -1

        self.fire_version = 0
        self.fire_task: Optional[asyncio.Task] = None
        self.finish = False
        # A detached request will not occupy compute resource, but still occupy memory for a while...
        self.detach_from_worker = True
        # Terminate can due to finish or evict.
        self.terminate_event = asyncio.Event()

        self.status = 'exec'
        self.migrate_event: Optional[asyncio.Event] = None

        self.handled_workers = []


    def get_info(self):
        return {'request_id': self.request_id, 'fire_version': self.fire_version,
                'model_type': self.model_type, 'model_id': self.model_id,
                'pool_type': self.node_type, 'node_id': self.node_id, 'worker_id': self.worker_id,
                'in': self.input_length, 'cur_out': self.output_length, 'exp_out': self.expect_output_length}

    def start_migration(self):
        # assert self.migrate_event is None
        # self.migrate_event = asyncio.Event()
        self.status = 'migrate'

    def end_migration(self):
        self.status = 'exec'

    async def wait_for_migration_complete(self):
        await self.migrate_event.wait()
        self.migrate_event = None

    def total_length(self):
        return self.input_length + self.output_length

    def expect_next_token_time(self):
        if self.finish:
            return 1e18
        return self.start_time + self.tolerate_time + self.TTFT_SLO + self.TPOT_SLO * self.output_length

    def ddl_violate(self):
        return self.expect_next_token_time() < time.time()

    def set_location(self, node_type: str, node_id: int, worker_id: int):
        self.node_type = node_type
        self.node_id = node_id
        self.worker_id = worker_id
        if node_type != '':
            self.handled_workers.append((self.output_length, f'{node_type}-{node_id}-{worker_id}'))

    def transform_to_decode_only_request(self):
        # We hack the model_id to force it decode...
        assert self.model_id < 256
        request_id_suffix = self.request_id.split('-', maxsplit=1)[1]
        self.model_id += 256
        self.request_id = f'{self.model_id}-{request_id_suffix}'

    def transform_to_normal_request(self):
        # We hack the model_id to force it decode...
        if config.enable_PD:
            if self.model_id >= 256:
                request_id_suffix = self.request_id.split('-', maxsplit=1)[1]
                self.model_id -= 256
                self.request_id = f'{self.model_id}-{request_id_suffix}'

    def receive_new_token(self, token_list):
        assert len(token_list) == 1
        self.text.append(token_list[0])
        self.output_length += 1
        if self.output_length == 1:
            self.first_token_time = time.time()
        if self.under_prefill:
            self.under_prefill = False
            if config.enable_PD:
                self.terminate_event.set()

    def perform_evict(self, save_kv: bool):
        assert self.fire_task is not None
        if not save_kv and not config.enable_PD:
            self.under_prefill = True
        self.fire_task.cancel()
        self.fire_task = None
        self.set_location('', -1, -1)
        self.terminate_event.set()

    def check_finish(self):
        if self.finish:
            return True
        assert self.output_length <= self.expect_output_length
        if self.output_length == self.expect_output_length:
            self.finish = True
            self.finish_time = time.time()
            self.terminate_event.set()
            return True
        else:
            return False

    def get_e2e_metrics(self):
        TTFT = self.first_token_time - self.start_time
        if self.output_length > 1:
            TPOT = (self.finish_time - self.first_token_time) / (self.output_length - 1)
        else:
            TPOT = 0
        return {'TTFT': round(TTFT, 3),
                'TPOT': round(TPOT, 3),
                'tolerate_time': round(self.tolerate_time, 3),
                'cold_start': self.TTFT_cold_start,
                'handled_workers': self.handled_workers}
