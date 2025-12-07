import subprocess
import argparse
import shlex
import os

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--max_model_len', type=int, required=False)

# CPU-only parameter
parser.add_argument('--kv_gb', type=int, required=False)
parser.add_argument('--numa', type=int, required=False)

# GPU-only parameter
parser.add_argument('--gpu', type=int, required=False)
parser.add_argument('--block_num', type=int, required=False)
parser.add_argument('--use_sllm', type=str, required=False)
parser.add_argument('--no_load', type=str, required=False)

args = parser.parse_args()
env_vars = {
    'VLLM_ENGINE_ITERATION_TIMEOUT_S': '3600'
}
max_model_len = 4096
if args.max_model_len is not None:
    max_model_len = args.max_model_len
if args.device == 'cpu':
    if args.numa is None:
        raise Exception('Argument numa is needed.')
    if args.kv_gb is None:
        raise Exception('Argument kv_gb is needed')

    numa_command = ''
    if args.numa >= 0:
        numa_command = f'numactl --cpunodebind={args.numa} --membind={args.numa}'

    command_str = (
        f'{numa_command} '
        f'python -m vllm.entrypoints.openai.api_server '
        f'--port {args.port} --max-model-len {max_model_len} '
        f'--model {args.model} '
    )
    env_vars.update(
        {'VLLM_OPENVINO_KVCACHE_SPACE': f'{args.kv_gb}',
         })
elif args.device == 'gpu':
    if args.gpu is None:
        raise Exception('Argument gpu is needed.')
    if args.block_num is None:
        raise Exception('Argument block_num is needed')
    if args.use_sllm is None:
        raise Exception('Argument use_sllm is needed')
    if args.no_load is None:
        raise Exception('Argument no_load is needed')

    block_num_override_command = ''
    if args.block_num >= 0:
        block_num_override_command = f'--num-gpu-blocks-override {args.block_num}'
    assert args.use_sllm in ['True', 'False']
    loader_override_command = ''
    if args.use_sllm == 'True':
        loader_override_command = '--load-format serverless_llm'
    assert args.no_load in ['True', 'False']

    command_str = (
        f'python -m vllm.entrypoints.openai.api_server '
        f'--port {args.port} --max-model-len {max_model_len} '
        f'--model {args.model} '
        f'--enforce-eager --gpu-memory-utilization 0.975 {block_num_override_command} --block-size 16 '
        f'{loader_override_command} '
    )
    env_vars.update(
        {'CUDA_VISIBLE_DEVICES': f'{args.gpu}',
         'NO_MODEL_LOADING_AT_START': f'{args.no_load}'})
else:
    raise Exception
print(command_str)
process = subprocess.Popen(shlex.split(command_str), env={**os.environ, **env_vars})
try:
    process.wait()
except KeyboardInterrupt:
    print('Main process received Ctrl-C.')
    process.kill()
    process.wait()
