import os
import json
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib.ticker as tk
from matplotlib.ticker import LogLocator, LogFormatter
import pandas as pd
import json

mpl.rcParams.update(mpl.rcParamsDefault)
# %matplotlib inline
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 30})
plt.rcParams['hatch.linewidth'] = 2
systems_name_maping = {
    'Ours': 'sota',
    'Ours-int4': 'sota',
    'sllm+cpu': 'serverlessllm',
    'sllm+share': 'serverlessllmshare',
    'sllm': 'serverlessllm',
    'NEO': 'serverlessllm',
    'NEO+share': 'serverlessllmshare',
}

systems_color = {
    'Ours': '#388697',
    'Ours-int4': '#bbded6',
    'NEO+share': '#bbded6',
    'sllm+cpu': '#bbded6',
    'sllm+share': '#61c0bf',
    'sllm': '#FFB6B9',
    'NEO': '#FFB6B9'
}
dark_color_list = ['#383838', '#5CA7C7', '#D4352D', '#FBCE6A', 'green']
cpu_gpu_hatch = {
    'cpu': '//',
    # 'cpu': 'xx',
    # 'gpu': r'\\'
    'gpu': ''
}

systems_marker = {
    'Ours': '*',
    'sllm+cpu': 's',
    'sllm+share': 'p',
    'sllm': '^'
}

nodes_marker = {
    'cpu': '^',
    'gpu': 'v'
}

# systems_hatch = {
#     'Ours': '*',
#     'Ours-int4': 'O',
#     'sllm+cpu': 'O.',
#     'sllm': 'O'
# }
systems_hatch = {
    'Ours': '',
    'Ours-int4': '',
    'sllm+cpu': '',
    'sllm+share': '',
    'sllm': ''
}

systems_linestyle = {
    'Ours': '-',
    'Ours-int4': '-',
    'sllm+cpu': '--',
    'sllm+share': ':',
    'sllm': '-'
}

# global_hatch_color = '#436850'
global_hatch_color = 'white'

systems_label_translation = {
    'Ours': r'SLINFER',
    # 'Ours': r'LLM-Mesh',
    'sllm+cpu': r'sllm+c',
    'sllm+share': r'sllm+c+s',
    'sllm': r'sllm',
    'NEO': r'sllm+NEO',
    'NEO+share': r'NEO+',
}

trace_start_time = 1
trace_end_time = 30
sample_function_cnt = 128
sample_group_size = 8
random_seed = 0

global_systems = ['sllm', 'sllm+cpu', 'sllm+share', 'Ours']


def read_trace(trace_start_time, trace_end_time, sample_function_cnt, sample_group_size, random_seed):
    filename = (f'{trace_start_time}_{trace_end_time}_'
                f'{sample_function_cnt}_{sample_group_size}_{random_seed}_timestamps.json')
    with open(f'../trace/{filename}') as f:
        trace_data = json.load(f)
    return trace_data


def get_rpm_list_list_from_trace_data(trace_data):
    rpm_list_list = []
    for entry in trace_data:
        rpm_list_list.append(entry['rpm_list'])
    return rpm_list_list


def get_samples_from_raw_list(raw_list, group_size, select_group_size):
    assert len(raw_list) % group_size == 0
    sample_list = []
    for idx, entry in enumerate(raw_list):
        if (idx % group_size) < select_group_size:
            sample_list.append(entry)
    return sample_list


def get_test_result(trace_start_time, trace_end_time, sample_function_cnt, sample_group_size, random_seed,
                    select_group_size, testing_minute, system_name, extra_info):
    system_name = systems_name_maping[system_name]
    expect_suffix = f'{trace_start_time}_{trace_end_time}_{sample_function_cnt}_{sample_group_size}_{random_seed}_{select_group_size}_{testing_minute}_{system_name}_({extra_info}).json'
    # print(expect_suffix)
    directory = '../test/result'
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    filenames.sort(reverse=True)
    for filename in filenames:
        if filename.endswith(expect_suffix):
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                return data
    return None


def get_result_analysis_info(result_data, testing_minute):
    analysis_info = {}
    fire_duration_second = testing_minute * 60
    TTFTs = []
    TPOTs = []
    total_fired_request = len(result_data['requests_response'])
    analysis_info['total_fired_request'] = total_fired_request
    slo_satisfied_cnt = 0
    cold_start_cnt = 0
    tokens_device_cnt = {'cpu': 0, 'gpu': 0}
    for entry in result_data['requests_response']:
        cur_request_info = entry['request_info']
        cur_response = entry['response']
        if cur_response['result'] is True:
            cur_e2e_metrics = cur_response['e2e_metrics']
            TTFTs.append(cur_e2e_metrics['TTFT'])
            TPOTs.append(cur_e2e_metrics['TPOT'])
            cold_start = cur_e2e_metrics['cold_start']
            if cold_start:
                cold_start_cnt += 1
            TTFT_slo = max(cur_request_info['input_length'] / 512, 0.5) + cur_e2e_metrics['tolerate_time']
            # TTFT_slo = max(cur_request_info['input_length'] / 512, 2)
            if cur_e2e_metrics['TTFT'] < TTFT_slo and cur_e2e_metrics['TPOT'] < 0.25:
                slo_satisfied_cnt += 1
            expect_output_length = entry['request_info']['expect_output_length']
            cur_handled_workers = cur_e2e_metrics['handled_workers']
            for idx, handled_worker_pair in enumerate(cur_handled_workers):
                if 'cpu' in handled_worker_pair[1]:
                    cur_device = 'cpu'
                elif 'gpu' in handled_worker_pair[1]:
                    cur_device = 'gpu'
                else:
                    raise Exception
                cur_tokens_cnt = 0
                # if idx == 0:
                #     cur_tokens_cnt += entry['request_info']['input_length']
                if idx == len(cur_handled_workers) - 1:
                    cur_tokens_cnt += expect_output_length - handled_worker_pair[0]
                else:
                    cur_tokens_cnt += cur_handled_workers[idx + 1][0] - handled_worker_pair[0]
                tokens_device_cnt[cur_device] += cur_tokens_cnt
    analysis_info['total_fired_cnt'] = len(result_data['requests_response'])
    analysis_info['served_cnt'] = len(TTFTs)
    analysis_info['slo_satisfied_cnt'] = slo_satisfied_cnt
    analysis_info['cold_start_cnt'] = cold_start_cnt
    analysis_info['TTFTs'] = TTFTs
    analysis_info['TPOTs'] = TPOTs
    analysis_info['device_info'] = {}
    analysis_info['device_info']['tokens_device_cnt'] = tokens_device_cnt
    analysis_info['device_info']['total_usage'] = {}
    analysis_info['device_info']['avg_usage'] = {}
    analysis_info['device_info']['token_throughput'] = {}
    analysis_info['device_info']['usage_detail'] = {}
    analysis_info['device_info']['node_density'] = {}
    analysis_info['device_info']['batch'] = {}
    analysis_info['device_info']['memory'] = {}
    if 'workers_kv_scale' in result_data['gateway_logs']:
        analysis_info['workers_kv_scale'] = result_data['gateway_logs']['workers_kv_scale']
    for node_type in ['cpu', 'gpu']:
        for optional_key in ['node_density', 'batch', 'memory']:
            if optional_key in result_data['gateway_logs']:
                analysis_info['device_info'][optional_key][node_type] = result_data['gateway_logs'][optional_key][node_type]
        analysis_info['device_info']['usage_detail'][node_type] = result_data['gateway_logs']['node_usage'][node_type]
        total_usage = sum(result_data['gateway_logs']['node_usage'][node_type])
        analysis_info['device_info']['total_usage'][node_type] = total_usage
        analysis_info['device_info']['avg_usage'][node_type] = total_usage / fire_duration_second
        analysis_info['device_info']['token_throughput'][node_type] = tokens_device_cnt[
                                                                          node_type] / total_usage if total_usage > 0 else 0
    return analysis_info


reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])

def make_cdf_data_from_pdf(raw_data: list):
    # raw_data: [(value_1, proportion_1), (), ...]
    raw_data.sort()
    x_data = []
    y_data = []
    total_proportion = sum(entry[1] for entry in raw_data)
    cur_cumulated_proportion = 0
    for entry in raw_data:
        cur_cumulated_proportion += entry[1]
        x_data.append(entry[0])
        y_data.append(cur_cumulated_proportion / total_proportion)
    return x_data, y_data

def make_cdf_data(raw_data: list, max_value, extra_sample_num):
    now_data = raw_data.copy()
    for i in range(extra_sample_num):
        now_data.append(2e18)
    sorted_data = sorted(now_data)
    cdf = list(np.arange(1, len(sorted_data) + 1) / len(sorted_data))
    for idx, value in enumerate(sorted_data):
        if value >= max_value:
            sorted_data = sorted_data[:idx]
            cdf = cdf[:idx]
            break
    # Add one last.
    if extra_sample_num > 0:
        sorted_data.append(max_value)
        cdf.append(cdf[-1])
    return sorted_data, cdf

def get_real_density_list_from_density_2d_list(raw_data):
    assert type(raw_data[0][0]) == int
    real_density_list = []
    # We skip the unallocated node
    for cur_density_list in raw_data:
        for worker_density in cur_density_list:
            if worker_density > 0:
                real_density_list.append(worker_density)
    return real_density_list

def get_real_batch_list_from_batch_3d_list(raw_data):
    real_batch_list = []
    for cur_pool_batch_list in raw_data:
        for cur_node_batch_list in cur_pool_batch_list:
            if sum(cur_node_batch_list) > 0:
                real_batch_list.append(max(cur_node_batch_list))
            # for cur_worker_batch in cur_node_batch_list:
            #     if cur_worker_batch > 0:
            #         real_batch_list.append(cur_worker_batch)
    return real_batch_list

def get_memory_utilization_ratio_list_from_memory_3d_list(raw_data):
    memory_uti_list = []
    for cur_pool_memory_list in raw_data:
        for cur_node_memory_list in cur_pool_memory_list:
            for cur_worker_memory in cur_node_memory_list:
                assert len(cur_worker_memory) == 3
                if cur_worker_memory[0] > 0 and cur_worker_memory[2] > 0:
                    memory_uti_list.append((cur_worker_memory[0] + cur_worker_memory[1]) / (cur_worker_memory[0] + cur_worker_memory[2]))
    return memory_uti_list
def get_kv_cache_utilization_ratio_list_from_memory_3d_list(raw_data):
    kv_uti_list = []
    for cur_pool_memory_list in raw_data:
        for cur_node_memory_list in cur_pool_memory_list:
            for cur_worker_memory in cur_node_memory_list:
                assert len(cur_worker_memory) == 3
                if 0 < cur_worker_memory[1] <= cur_worker_memory[2]:
                    kv_uti_list.append(cur_worker_memory[1] / cur_worker_memory[2])
    return kv_uti_list

def cal_kv_scale_overhead_time_ratio(raw_data):
    total_lifetime = 0
    total_scale_time = 0
    total_scale_cnt = 0
    for worker_lifetime, kv_scale_list in raw_data:
        if len(kv_scale_list) < 2:
            continue
        assert len(kv_scale_list) >= 2
        total_scale_cnt += len(kv_scale_list)
        cur_scale_time = sum(entry[2] for entry in kv_scale_list)
        total_lifetime += worker_lifetime
        total_scale_time += cur_scale_time
    return total_scale_time / total_lifetime

# draw e2e system metric
# This is the new code!!!, 4 systems, include sllm+share
# 3b

testing_minute = int(input("Enter testing_minute: (e.g., 10 or 30)"))
CPU_count = int(input("Enter CPU_count: (e.g., 0 or 4)"))
GPU_count = int(input("Enter GPU_count: (e.g., 4)"))

system_width = 0.23
model_size = input("Enter model_size: (e.g. 3b or 7b or 13b)")
if model_size not in ['3b', '7b', '13b']:
    raise ValueError("model_size must be either 3b or 7b or 13b")
extra_info = f'{GPU_count}G{CPU_count}C-{model_size}'
select_group_size_list = [2, 4, 8]
x_cnt = len(select_group_size_list)
fig = plt.figure(figsize=(9.6, 15))
gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[4, 5.5, 6.5, 5.5])
axes = []
for i in range(4):
    axes.append(fig.add_subplot(gs[i]))
for ax in axes:
    for x_pos in range(x_cnt - 1):
        ax.axvline(x=x_pos + 0.5, linestyle='--', color='pink')
for ax in axes:
    ax.set_xlim(-0.5, x_cnt - 0.5)
    ax.set_xticks(list(range(x_cnt)))
    ax.set_xticklabels(['' for i in range(x_cnt)])
    # ax.set_xticklabels([])
    # ax.set_xlabel('')
for ax in [axes[-1]]:
    ax.set_xticklabels(
        [sample_function_cnt * select_group_size // sample_group_size for select_group_size in select_group_size_list])
    # ax.set_xlabel('Load (Number of Models)')
    ax.set_xlabel('Number of Models')
    ax.set_ylabel('Average\nNodes Used')
    ax.set_ylim(0, 4.8)
    ax.set_yticks([0, 2, 4])
    if model_size:
        ax.text(
            0.02, 0.98,
            '↓ is better',
            transform=ax.transAxes,
            fontsize=24,
            ha='left',
            va='top',
            zorder=10,
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='grey',
                linestyle='dotted',
                pad=2
            ))


for ax in [axes[-2]]:
    if model_size:
        ax.text(
            0.02, 0.58,
            '↑ is better',
            transform=ax.transAxes,
            fontsize=24,
            ha='left',
            va='top',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='grey',
                linestyle='dotted',
            ))
    # ax.set_ylim(0, 220)
    # ax.set_yticks([0, 500, 1000])
    # ax.set_yticklabels(['0', '', '1K'])
    ax.set_ylabel('Decode Speed\nTokens/(Node·s)')
    legend_handles = []
    for now_system in global_systems:
        legend_handles.append(
            Patch(facecolor=systems_color[now_system], edgecolor=global_hatch_color, hatch=systems_hatch[now_system],
                  label=systems_label_translation[now_system]))
    first_legend = ax.legend(handles=legend_handles, ncol=4, loc='upper left', fontsize=24, handlelength=1, columnspacing=0.5,
              handletextpad=0.3, borderaxespad=0.2)
    ax.add_artist(first_legend)
    legend_handles = []
    legend_handles.extend([
        Patch(facecolor='white', edgecolor='#436850', hatch=cpu_gpu_hatch['cpu'], label='CPU-Node'),
        Patch(facecolor='white', edgecolor='#436850', hatch=cpu_gpu_hatch['gpu'], label='GPU-Node'), ])
    # legend_handles = reorder(legend_handles, 3)
    ax.legend(handles=legend_handles, ncol=2, loc='lower left', bbox_to_anchor=(0, 0.6), fontsize=24, handlelength=1, columnspacing=0.5,
              handletextpad=0.3, borderaxespad=0.2)

systems_throughput = {}
for system in global_systems:
    systems_throughput[system] = {'total_fired': [], 'goodput': [], 'TTFTs': [], 'served_cnt': []}
    systems_throughput[system]['total_fired'] = [0 for i in range(len(select_group_size_list))]
    systems_throughput[system]['served_cnt'] = [0 for i in range(len(select_group_size_list))]
    systems_throughput[system]['goodput'] = [0 for i in range(len(select_group_size_list))]
    systems_throughput[system]['TTFTs'] = [[] for i in range(len(select_group_size_list))]
for x_group, select_group_size in enumerate(select_group_size_list):
    for system_idx, now_system in enumerate(global_systems):
        x_system_middle = x_group + (system_idx - (len(global_systems) - 1) / 2) * system_width
        cur_extra_info = extra_info
        if now_system == 'sllm':
            # assert cur_extra_info[2:4] == '4C'
            # cur_extra_info = cur_extra_info.replace('4C', '0C')
            cur_extra_info = cur_extra_info.replace(f'{GPU_count}G{CPU_count}C', f'{GPU_count}G0C')
        cur_test_result = get_test_result(trace_start_time, trace_end_time, sample_function_cnt, sample_group_size,
                                          random_seed, select_group_size, testing_minute, now_system, cur_extra_info)
        if cur_test_result is not None:
            result_info = get_result_analysis_info(cur_test_result, testing_minute)
        else:
            result_info = {}
            result_info['TTFTs'] = [0]
            result_info['total_fired_cnt'] = 0
            result_info['served_cnt'] = 0
            result_info['slo_satisfied_cnt'] = 0
            result_info['device_info'] = {'avg_usage': {'cpu': 0, 'gpu': 0},
                                          'token_throughput': {'cpu': 0, 'gpu': 0},}
        systems_throughput[now_system]['TTFTs'][x_group] = result_info['TTFTs']
        systems_throughput[now_system]['total_fired'][x_group] = result_info['total_fired_cnt']
        systems_throughput[now_system]['served_cnt'][x_group] = result_info['served_cnt']
        systems_throughput[now_system]['goodput'][x_group] = result_info['slo_satisfied_cnt']
        # Draw node_avg_usage
        cur_ax = axes[-1]
        bar_width = 0.1
        avg_usage = result_info['device_info']['avg_usage']
        # print(select_group_size, now_system, result_info['device_info']['tokens_device_cnt'], result_info['served_cnt'])
        for type_idx, node_type in enumerate(['cpu', 'gpu']):
            cur_bar_pos = x_system_middle + (type_idx - 0.5) * bar_width
            cur_x = cur_bar_pos
            cur_y = avg_usage[node_type]

            cur_ax.bar(cur_x, cur_y, width=bar_width, color=systems_color[now_system],
                       hatch=f'{cpu_gpu_hatch[node_type]}{systems_hatch[now_system]}', edgecolor=global_hatch_color)
            text_x_offset = -0.00 if type_idx == 0 else 0.03
            text_y_offset = -0.04
            text_x_offset = 0.01
            text_y_offset = 0.05
            cur_ax.text(
                cur_x + text_x_offset, cur_y + text_y_offset,
                f'{cur_y:.1f}',
                fontsize=18,
                ha='center',
                va='bottom',
                rotation=90,
            )

        # Draw node_throughput
        cur_ax = axes[-2]
        node_throughput = result_info['device_info']['token_throughput']
        # if now_system in ['sllm+share', 'Ours']:
        #     print(now_system, node_throughput['cpu'], node_throughput['gpu'])
        for type_idx, node_type in enumerate(['cpu', 'gpu']):
            cur_bar_pos = x_system_middle + (type_idx - 0.5) * bar_width
            cur_x = cur_bar_pos
            cur_y = node_throughput[node_type]
            cur_ax.bar(cur_x, cur_y, width=bar_width, color=systems_color[now_system],
                       hatch=f'{cpu_gpu_hatch[node_type]}{systems_hatch[now_system]}', edgecolor=global_hatch_color)
            text_x_offset = -0.02 if type_idx == 0 else 0.02
            # cur_ax.text(
            #     cur_x + text_x_offset, cur_y,
            #     f'{cur_y:.1f}',
            #     fontsize=12,
            #     ha='center',
            #     va='bottom',
            # )
        # Draw TTFT CDF

for cur_ax in [axes[-3]]:
    # cur_ax.set_ylabel('Requests\nMeeting SLO')
    if model_size:
        cur_ax.text(
            0.38, 0.96,
            '↑ is better',
            transform=cur_ax.transAxes,
            fontsize=24,
            ha='left',
            va='top',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='grey',
                linestyle='dotted',
            ))
    cur_ax.set_ylabel('SLO-met Req')
    # cur_ax.set_ylim(1200, 9600)
    # cur_ax.set_yticks([2000, 5000, 8000])
    # cur_ax.set_yticklabels(['2K', '5K', '8K'])
    for idx, now_system in enumerate(reversed(global_systems)):
        good_put_data = systems_throughput[now_system]['goodput']
        total_fired_data = systems_throughput[now_system]['total_fired']
        # if now_system == 'sllm':
        #     print(total_fired_data)
        # print(now_system, good_put_data)
        if now_system == 'Ours':
            cur_ax.plot(total_fired_data, color='black', linestyle='dotted', linewidth=3, marker='.', markersize=12,
                        label='Total Req', zorder=4)
        cur_ax.plot(good_put_data, color=systems_color[now_system], zorder=3-idx, linewidth=3,
                    marker=systems_marker[now_system], markersize=12 if now_system != 'Ours' else 15,
                    label=f'{systems_label_translation[now_system]}')
        # print(now_system, good_put_data, total_fired_data)
    cur_ax.legend(ncol=1, loc='upper left', fontsize=24, handlelength=2, columnspacing=0.5, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2)

for ax in [axes[-4]]:
    if model_size:
        ax.text(
            0.06, 0.5,
            '↖ is better',
            transform=ax.transAxes,
            fontsize=24,
            ha='left',
            va='top',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='grey',
                linestyle='dotted',
            ))
    ax.set_xticks([])
    x_ticks = []
    xticklabels = []
    for x_group in range(x_cnt):
        x_st = x_group - 0.5
        x_ed = x_group + 0.5
        max_value = 8
        for i in range(0, 8, 2):
            x_ticks.append(i / max_value + x_st)
            xticklabels.append(i)
        for system in global_systems:
            cur_TTFTs = systems_throughput[system]['TTFTs'][x_group]
            dropped_reqs = systems_throughput[system]['total_fired'][x_group] - \
                           systems_throughput[system]['served_cnt'][x_group]
            # print(f'{x_group} {system} dropped {dropped_reqs}')
            # print(f'{x_group} {system} max TTFTs {max(cur_TTFTs)}')
            x_data, y_data = make_cdf_data(cur_TTFTs, max_value, dropped_reqs)
            x_data = np.array(x_data)
            x_data = x_data / max_value + x_st
            # print(max(x_data), min(x_data))
            zorder = 2
            if systems_linestyle[system] != '-':
                zorder = 3
            if x_group == 0:
                ax.plot(x_data, y_data, zorder=zorder, color=systems_color[system], linewidth=3, linestyle=systems_linestyle[system],
                        label=systems_label_translation[system])
            else:
                ax.plot(x_data, y_data, color=systems_color[system], linewidth=3, linestyle=systems_linestyle[system])
    ax.legend(ncol=4, loc='lower left', fontsize=24, handlelength=1, columnspacing=0.5, handletextpad=0.3,
              borderaxespad=0.2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('TTFT (s)')
    ax.set_ylabel('TTFT\nCDF')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    for now_y in np.arange(0, 1.01, 0.5):
        ax.axhline(y=now_y, color='tab:grey', linestyle='--', alpha=0.5)

gs.update(hspace=0)
print('saving figure pdf to ' + 'figures/' + f'e2e_{model_size}' +'.pdf')
plt.savefig('figures/' + f'e2e_{model_size}' +'.pdf', bbox_inches='tight')
plt.show()