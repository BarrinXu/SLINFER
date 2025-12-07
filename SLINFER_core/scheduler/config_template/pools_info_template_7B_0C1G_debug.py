from .models_info_template import models_info_template

models_info = models_info_template

pools_info_template = {
    'gpu': {
        0: {
            'node_memory_capacity_GB': 78,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8000,
            'node_label': 'gpu',
            'workers': {
                0: models_info['llama-2-7b'],
                1: models_info['llama-2-7b'],
                2: models_info['llama-2-7b'],
                3: models_info['llama-2-7b'],
            },
        },
    },
    'cpu': {
    },
}
