import time
import aiohttp
import asyncio
import os
import logging
import datetime
import json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

total_fired_cnt = 0
result = {'requests_response': []}

models_size_type_mapping = {
    '3b': 'llama-3.2-3b',
    '7b': 'llama-2-7b',
    '8b': 'llama-3.1-8b',
    '13b': 'llama-2-13b',
}


async def start_gateway_monitor(session: aiohttp.ClientSession):
    async with session.post('http://localhost:7000/start_monitor') as response:
        assert response.status == 200


async def end_gateway_monitor(session: aiohttp.ClientSession):
    async with session.post('http://localhost:7000/end_monitor') as response:
        res = await response.json()
    return res


async def post_request(session: aiohttp.ClientSession, payload):
    global total_fired_cnt
    total_fired_cnt += 1
    logger.debug(f'start {payload["request_info"]}')
    async with session.post('http://localhost:7000/v1/completions', json=payload) as response:
        res = await response.json()
    if res['result'] is False:
        logger.debug(f'failed {payload["request_info"]}')
    else:
        assert res['result'] is True
        logger.debug(f'end {payload["request_info"]}\n\tE2E metrics:{res["e2e_metrics"]}')
    if 'requests_response' not in result:
        result['requests_response'] = []
    result['requests_response'].append({'request_info': payload['request_info'], 'response': res})


async def fire_one_function(session: aiohttp.ClientSession, function_info,
                            test_start_time, test_duration_limit, model_type, drop_interval):
    assert drop_interval >= 0
    model_id = function_info['model_id']
    base_payload = {
        'model_id': model_id,
        'model_type': model_type,
    }
    request_id = 0
    start_time = time.perf_counter()
    for idx, timestamp in enumerate(function_info['timestamps']):
        if timestamp > test_duration_limit:
            break
        cur_payload = base_payload.copy()
        cur_payload['request_id'] = f'{model_id}-{request_id:06d}'
        cur_payload['input_length'] = function_info['input'][idx]
        cur_payload['expect_output_length'] = function_info['output'][idx]
        await asyncio.sleep(timestamp - (time.perf_counter() - start_time))
        if drop_interval > 0 and (idx % drop_interval) == (drop_interval - 1):
            # We drop this request to simulate NEO's CPU capacity
            pass
        else:
            asyncio.create_task(post_request(session, {'request_info': cur_payload}))
        request_id += 1


async def initialize_gateway(session: aiohttp.ClientSession, new_configs):
    async with session.post('http://localhost:7000/set_config', json=new_configs) as response:
        res = await response.json()
        assert res['result'] is True
    async with session.post('http://localhost:7000/get_config') as response:
        res = await response.json()
    return res


model_size_list = ['3b', '7b', '13b']


async def main(trace_start_time, trace_end_time, sample_function_cnt, sample_group_size, random_seed,
               selected_group_size, testing_minute, system_config: dict, testing_config, extra_info='', trace_info=''):
    assert 0 < selected_group_size <= sample_group_size
    model_size = testing_config['model_size']

    global total_fired_cnt
    global result
    total_fired_cnt = 0
    result.clear()
    system_name = system_config['system']
    logger.info(f'fire with trace (start_time: {trace_start_time}, end_time: {trace_end_time}, '
                f'sample_function_cnt: {sample_function_cnt}, sample_group_size: {sample_group_size}, '
                f'seed: {random_seed}), \n'
                f'selected_group_size: {selected_group_size}, testing_minute: {testing_minute}, '
                f'syetem_config: {system_config}, \n'
                f'testing_config: {testing_config}, extra_info: {extra_info}, trace_info: {trace_info}')
    skip_even_group = testing_config.get('skip_even_group', False)
    skip_odd_group = testing_config.get('skip_odd_group', False)
    selected_func_ids = testing_config.get('selected_func_ids', None)
    drop_interval = testing_config.get('drop_interval', 0)
    if selected_func_ids:
        assert type(selected_func_ids) == list
        logger.warning(f'selected_func_id: {selected_func_ids}')
    logger.warning(f'skip_even_group? {skip_even_group}')
    logger.warning(f'skip_odd_group? {skip_odd_group}')
    logger.warning(f'drop_interval? Only for NEO! {drop_interval}')

    if trace_info != '':
        trace_info = f'({trace_info})'
    input_file_name = (f'{trace_start_time}_{trace_end_time}_'
                       f'{sample_function_cnt}_{sample_group_size}_{random_seed}{trace_info}_timestamps.json')
    logger.warning(f'using trace: {input_file_name}')
    with open(f'../trace/{input_file_name}') as f:
        function_list = json.load(f)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024, force_close=True),
                                     timeout=aiohttp.ClientTimeout()) as session:
        gateway_config = await initialize_gateway(session, system_config)
        logger.info(f'gateway_config: {gateway_config}')

        test_start_time = time.perf_counter()
        await start_gateway_monitor(session)
        test_duration_limit = testing_minute * 60
        test_end_time = test_start_time + test_duration_limit

        tasks = []
        for idx, function_info in enumerate(function_list):
            if selected_func_ids is not None:
                if idx not in selected_func_ids:
                    continue
            cur_group_id = idx // sample_group_size
            cur_group_offset = idx % sample_group_size
            if skip_even_group:
                if cur_group_id % 2 == 0:
                    continue
            if skip_odd_group:
                if cur_group_id % 2 == 1:
                    continue
            if cur_group_offset >= selected_group_size:
                continue
            model_type = None
            if model_size != 'mixed':
                model_type = models_size_type_mapping[model_size]
            else:
                if 'model_cdf' not in testing_config:
                    # This is a fixed ratio...
                    model_type = models_size_type_mapping[model_size_list[idx % len(model_size_list)]]
                else:
                    assert selected_group_size == testing_config['model_cdf'][-1]
                    for model_idx, model_max_offset in enumerate(testing_config['model_cdf']):
                        if cur_group_offset < model_max_offset:
                            model_type = models_size_type_mapping[model_size_list[model_idx]]
                            break
            assert model_type is not None
            function_info['model_id'] = idx
            tasks.append(asyncio.create_task(
                fire_one_function(session, function_info, test_start_time, test_duration_limit, model_type,
                                  drop_interval)))

        await asyncio.gather(*tasks)
        await asyncio.sleep(test_end_time - time.perf_counter())
        gateway_logs = await end_gateway_monitor(session)
        result['gateway_logs'] = gateway_logs
        # Wait for unfinished requests to complete
        await asyncio.sleep(150)

    for pool_type in ['cpu', 'gpu']:
        print(f'analyzing {pool_type} usage...')
        total_usage = sum(gateway_logs['node_usage'][pool_type])
        print(f'\ttotal_usage: {total_usage}, avg_usage: {total_usage / test_duration_limit:.3f}')

    total_cnt = 0
    fail_cnt = 0

    for request_response in result['requests_response']:
        total_cnt += 1
        entry = request_response['response']
        if entry['result'] is False:
            fail_cnt += 1
    result['gateway_config'] = gateway_config
    pools_config = gateway_config['pools_config']
    cpu_cnt = len(pools_config['cpu'])
    gpu_cnt = len(pools_config['gpu'])
    if system_config.get('enable_cpu', True) is False:
        cpu_cnt = 0

    print(f'total_fired_cnt: {total_fired_cnt}')
    print(f'Response: total_cnt {total_cnt}, success_cnt: {total_cnt - fail_cnt}, fail_cnt: {fail_cnt}')

    nowtime = str(datetime.datetime.now())
    nowtime = nowtime.replace(':', '-')
    nowtime = nowtime.replace(' ', '-')
    nowtime = nowtime.split('.')[0]
    if extra_info != '':
        extra_info = f'-{extra_info}'
    if system_name == 'serverlessllm' and system_config.get('sllm_enable_sharing', None) is True:
        system_name += 'share'
    suffix = (f'{trace_start_time}_{trace_end_time}_'
              f'{sample_function_cnt}_{sample_group_size}_{random_seed}_{selected_group_size}_{testing_minute}_'
              f'{system_name}_({gpu_cnt}G{cpu_cnt}C-{model_size}{extra_info})')
    if not os.path.exists('result'):
        os.mkdir('result')
    output_filepath = os.path.join('result', f'{nowtime}_{suffix}.json')
    with open(output_filepath, 'w') as f:
        json.dump(result, f)


for cur_selected_group_size in [2, 4, 8]:
    asyncio.run(main(1, 30, 128, 8, 0,
                     cur_selected_group_size, 10,
                     {'system': 'serverlessllm', 'enable_cpu': False},
                     {'model_size': '13b'}, ))
    asyncio.run(main(1, 30, 128, 8, 0,
                     cur_selected_group_size, 10,
                     {'system': 'serverlessllm'},
                     {'model_size': '13b'}, ))
    asyncio.run(main(1, 30, 128, 8, 0,
                     cur_selected_group_size, 10,
                     {'system': 'serverlessllm', 'sllm_enable_sharing': True},
                     {'model_size': '13b'},))
    asyncio.run(main(1, 30, 128, 8, 0,
                     cur_selected_group_size, 10,
                     {'system': 'sota'},
                     {'model_size': '13b'}, ))
