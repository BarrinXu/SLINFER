from models_info_template import models_info_template

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
                0: models_info['llama-3.2-3b'],
                1: models_info['llama-3.2-3b'],
                2: models_info['llama-3.2-3b'],
                3: models_info['llama-3.2-3b'],
                4: models_info['llama-3.2-3b'],
                5: models_info['llama-3.2-3b'],
                6: models_info['llama-3.2-3b'],
                7: models_info['llama-3.2-3b'],
            },
        },
        1: {
            'node_memory_capacity_GB': 78,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8100,
            'node_label': 'gpu',
            'workers': {
                0: models_info['llama-3.2-3b'],
                1: models_info['llama-3.2-3b'],
                2: models_info['llama-3.2-3b'],
                3: models_info['llama-3.2-3b'],
                4: models_info['llama-3.2-3b'],
                5: models_info['llama-3.2-3b'],
                6: models_info['llama-3.2-3b'],
                7: models_info['llama-3.2-3b'],
            },
        },
        2: {
            'node_memory_capacity_GB': 78,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8200,
            'node_label': 'gpu',
            'workers': {
                0: models_info['llama-3.2-3b'],
                1: models_info['llama-3.2-3b'],
                2: models_info['llama-3.2-3b'],
                3: models_info['llama-3.2-3b'],
                4: models_info['llama-3.2-3b'],
                5: models_info['llama-3.2-3b'],
                6: models_info['llama-3.2-3b'],
                7: models_info['llama-3.2-3b'],
            },
        },
        3: {
            'node_memory_capacity_GB': 78,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8300,
            'node_label': 'gpu',
            'workers': {
                0: models_info['llama-3.2-3b'],
                1: models_info['llama-3.2-3b'],
                2: models_info['llama-3.2-3b'],
                3: models_info['llama-3.2-3b'],
                4: models_info['llama-3.2-3b'],
                5: models_info['llama-3.2-3b'],
                6: models_info['llama-3.2-3b'],
                7: models_info['llama-3.2-3b'],
            },
        },
    },
    'cpu': {
    },
}
