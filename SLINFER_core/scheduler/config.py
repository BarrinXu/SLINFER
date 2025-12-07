from config_template.models_path_template import models_path_template
from config_template.models_info_template import models_info_template
from config_template.pools_info_template import pools_info_template

keep_alive_time = 1
system = 'sota'
gateway_ip = '127.0.0.1'
decode_preempt_metric = 'batch'
memory_preempt_metric = 'batch'
pool_priority = 'cpu'
enable_defragmentation = True
enable_sharing = True
enable_preempt = True
kv_scale_watermark = 0.25
enable_detailed_logging = False
enable_PD = False
sllm_enable_sharing = False
sllm_max_shares = 2
TTFT_baseline = 0.5 * 0.95
TTFT_max_threshold = 8 * 0.95
TPOT = 0.25 * 0.95
enable_cpu = True
minimal_tokens_per_instance = 4096
ddl_based_schedule = {
    'enable_batch_aware': False,
    'safe_ddl_threshold': 5
}
models_path = models_path_template
models_info = models_info_template
pools_config = pools_info_template
