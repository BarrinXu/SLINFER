models_info_template = {
    'llama-2-13b': {
        'model_type': 'llama-2-13b',
        'model_memory_GB': 24.3,
        'per_token_kv_memory_KB': 800,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 32
    },
    'llama-3.1-8b': {
        'model_type': 'llama-3.1-8b',
        # This is a hack to avoid OOM.
        'model_memory_GB': 22.0,
        'per_token_kv_memory_KB': 128,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 32
    },
    'llama-2-7b': {
        'model_type': 'llama-2-7b',
        'model_memory_GB': 12.6,
        'per_token_kv_memory_KB': 512,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 32
    },
    'llama-3.2-3b': {
        'model_type': 'llama-3.2-3b',
        'model_memory_GB': 6.1,
        'per_token_kv_memory_KB': 112,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 16
    },
}
