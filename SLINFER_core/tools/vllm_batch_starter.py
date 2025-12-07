import argparse
import datetime
import sys
import os
import subprocess
import shlex

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
target_dir = os.path.join(parent_dir, "scheduler", "config_template")
sys.path.append(target_dir)
from models_path_template import models_path_template

models_template = {
    'llama-2-13b': {
        'gpu': f'python vllm_backend_starter.py --device gpu --block_num 256 --use_sllm True --no_load True --model {models_path_template["llama-2-13b"]["gpu"]}',
        'cpu': f'python vllm_backend_starter.py --device cpu --numa -1 --model {models_path_template["llama-2-13b"]["cpu"]}'
    },
    # 'llama-3.1-8b': {
    #     'gpu': f'python vllm_backend_starter.py --max_model_len 65536 --device gpu --block_num 4096 --use_sllm True --no_load True --model {models_path_template["llama-3.1-8b"]["gpu"]}',
    #     'cpu': f'python vllm_backend_starter.py --max_model_len 65536 --device cpu --numa -1 --model {models_path_template["llama-3.1-8b"]["cpu"]}'
    # },
    'llama-2-7b': {
        'gpu': f'python vllm_backend_starter.py --device gpu --block_num 256 --use_sllm True --no_load True --model {models_path_template["llama-2-7b"]["gpu"]}',
        'cpu': f'python vllm_backend_starter.py --device cpu --numa -1 --model {models_path_template["llama-2-7b"]["cpu"]}'
    },
    'llama-3.2-3b': {
        'gpu': f'python vllm_backend_starter.py --device gpu --block_num 256 --use_sllm False --no_load True --model {models_path_template["llama-3.2-3b"]["gpu"]}',
        'cpu': f'python vllm_backend_starter.py --device cpu --numa -1 --model {models_path_template["llama-3.2-3b"]["cpu"]}',
    },
}


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--gpu', type=str, required=False)
parser.add_argument('--cpu_kv_gb', type=int, required=False)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--worker_num', type=int, required=True)
args = parser.parse_args()

if args.device == 'gpu' and args.gpu is None:
    raise Exception('Argument gpu is needed')



nowtime = str(datetime.datetime.now())
nowtime = nowtime.replace(':', '-')
nowtime = nowtime.replace(' ', '-')

if not os.path.exists('output'):
    os.mkdir('output')

final_commands = []
for worker_id in range(args.worker_num):
    cur_port = args.port + worker_id
    if args.device == 'gpu':
        assert args.gpu is not None
        cur_command = f'{models_template[args.model][args.device]} --gpu {args.gpu} --port {cur_port}'
        cur_output_file = f'{nowtime}_{args.model}_{args.gpu}_{cur_port}.out'
    elif args.device == 'cpu':
        assert args.cpu_kv_gb is not None
        cur_command = f'{models_template[args.model][args.device]} --kv_gb {args.cpu_kv_gb} --port {cur_port}'
        cur_output_file = f'{nowtime}_{args.model}_{cur_port}.out'
    else:
        raise Exception('Invalid argument device')
    final_commands.append((cur_command, cur_output_file))

processes = []

for cmd, output_file in final_commands:
    print(f'cur_cmd: {cmd}, cur_output_file: {output_file}')
    with open(f'output/{output_file}', 'w') as f:
        process = subprocess.Popen(shlex.split(cmd), stdout=f, stderr=subprocess.STDOUT)
        processes.append(process)

try:
    for process in processes:
        process.wait()
except KeyboardInterrupt:
    print('Main process received Ctrl-C.')
    for process in processes:
        process.kill()
    for process in processes:
        process.wait()
