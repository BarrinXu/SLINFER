# SLINFER Setup and Experiment Guide

SLINFER can run on any number of CPU and GPU machines. The instructions below are based on the following hardware setup:

- **1 GPU machine** with 4 × NVIDIA A100-80GB GPUs  
- **4 CPU machines**, each equipped with a 4th-generation (or newer) Intel Xeon processor with 32+ cores

> **Note**: SLINFER also supports GPU-only deployment. If you wish to test in a GPU-only environment, you may skip all steps related to CPU machines.

Ensure that ports in the range **7000–8999** are **free** on all machines to avoid port conflicts.

---

## 1. Environment Setup

On **every machine** (1 GPU + 4 CPU machines):

1. Clone or download the SLINFER project.
2. Set the absolute path to the project root as an environment variable:

   ```bash
   export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
   ```
3. Create and activate a Conda virtual environment:
   * On GPU machines:
      ```bash
      conda create -n SLINFER-GPU python=3.11
      conda activate SLINFER-GPU
      ```
   * On GPU machines:
      ```bash
      conda create -n SLINFER-CPU python=3.11
      conda activate SLINFER-CPU
      ```
> All subsequent steps must be performed within the activated Conda environment.

## 2. Software Installation
### For GPU Machines
**Prerequisite:** Ensure nvcc version is **12.4** (higher versions may work but are untested).
```bash
nvcc --version  # Should output 12.4
```
1. Install ServerlessLLM model loader:
    ```bash
    cd $PROJECT_BASE/ServerlessLLM_modify/sllm_store
    rm -rf build
    pip install .
    ```
   Verify installation:
    ```bash
    pip list | grep serverless-llm-store
    ```
2. Install modified vLLM:
    ```bash
    cd $PROJECT_BASE/vLLM_modify
    pip install -e .
    ```
   > You may see dependency conflicts warnings, which can be safely ignored.
3. Install compatible dependencies and plotting tools:
    ```bash
    pip install torch==2.3.1 transformers==4.46.3
    pip uninstall pyairports -y
    pip install git+https://github.com/ozeliger/pyairports.git
    pip install matplotlib seaborn 
   ```
    > You may see dependency conflicts warnings again, which can be safely ignored.

GPU machine setup is now complete.

### For CPU Machines

1. Install vLLM with OpenVINO support:
    ```bash
    cd $PROJECT_BASE/vLLM_modify
    pip install -r requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
    PIP_PRE=1 PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly/" VLLM_TARGET_DEVICE=openvino python -m pip install -v -e .
    ```
CPU machine setup is now complete.

## 3. Model Preparation
### Download Models (on all machines)
Download the following models from Hugging Face into `$PROJECT_BASE/huggingface_models/`:

* `Llama-3.2-3B-Instruct`
* `Llama-2-7b-chat-hf`
* `Llama-2-13b-chat-hf`
> Tip: To save time, you may download only one model (e.g., Llama-2-7b-chat-hf) for partial testing.

Expected directory structure:
```
$PROJECT_BASE/
└── huggingface_models/
    ├── Llama-3.2-3B-Instruct/
    │   └── *.safetensors
    ├── Llama-2-7b-chat-hf/
    │   └── *.safetensors
    └── Llama-2-13b-chat-hf/
        └── *.safetensors
```

### Convert Models
#### On GPU Machine
Copy Llama-3.2-3B, convert Llama-2-7b and Llama-2-13b (skip 3B due to compatibility issues with ServerlessLLM):
```bash
$PROJECT_BASE/huggingface_models/export_gpu_models.sh
```
Resulting structure:
```
$PROJECT_BASE/
└── gpu_models/
    ├── Llama-3.2-3B-Instruct/     # (just copy, not converted)
    │   └── *.safetensors
    ├── Llama-2-7b-chat-hf/
    │   └── rank_0/
    └── Llama-2-13b-chat-hf/
        └── rank_0/
```
#### On each CPU Machine
Convert all models for OpenVINO:
```bash
$PROJECT_BASE/huggingface_models/export_cpu_models.sh
```
Resulting structure:
```
$PROJECT_BASE/
└── cpu_models/
    ├── Llama-3.2-3B-Instruct/
    │   └── openvino_model.bin
    ├── Llama-2-7b-chat-hf/
    │   └── openvino_model.bin
    └── Llama-2-13b-chat-hf/
        └── openvino_model.bin
```

## 4. Experiment Preparation
### Architecture Overview
* GPU machine runs:
  * A ServerlessLLM model loader service
  * A root gateway that receive and schedule requests
  * One wrapper of inference instances per GPU
* Each CPU machine runs:
  * A distributed gateway (receives requests from root gateway)
  * A wrapper of inference instances
> Skip CPU-related steps if running in GPU-only mode.
### Step-by-Step Setup
#### 1. Configure Network (Skip for GPU-only)
Edit the following config files to specify IP addresses of your CPU machines:
* `scheduler/config_template/pools_info_template_3B_4C4G.py` → lines 83, 97, 111, 125
* `scheduler/config_template/pools_info_template_7B_4C4G.py` → lines 67, 79, 91, 103
* `scheduler/config_template/pools_info_template_13B_4C4G.py` → lines 59, 70, 81, 92
#### 2. (Optional) Enable NVIDIA MPS on GPU Machine
```bash
nvidia-cuda-mps-control -d
```
#### 3. GPU Machine: Launch Terminals
Open **7 persistent terminals** (e.g., using tmux or screen). In each, activate the environment and set `PROJECT_BASE`.

On Window-0 (GPU-0's instances wrapper):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
export OMP_NUM_THREADS=4
cd $PROJECT_BASE/SLINFER_core/tools
```
On Window-1 (GPU-1's instances wrapper):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
export OMP_NUM_THREADS=4
cd $PROJECT_BASE/SLINFER_core/tools
```
On Window-2 (GPU-2's instances wrapper):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
export OMP_NUM_THREADS=4
cd $PROJECT_BASE/SLINFER_core/tools
```
On Window-3 (GPU-3's instances wrapper):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
export OMP_NUM_THREADS=4
cd $PROJECT_BASE/SLINFER_core/tools
```
On Window-loader (ServerlessLLM model loader):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
sllm-store-server --storage_path $PROJECT_BASE/gpu_models --mem_pool_size 64
```
Please wait until `sllm-store-server` outputs `"Server listening on 0.0.0.0:8073"`.

On Window-gateway (root gateway):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
cd $PROJECT_BASE/SLINFER_core/scheduler
```
On Window-test (later will run test script):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
cd $PROJECT_BASE/SLINFER_core/tools/test
```

#### 4. CPU Machines: Launch Terminals (Skip for GPU-only)

On **each CPU machine**, open **2 terminals**:

On Window-0 (CPU's instances wrapper):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
cd $PROJECT_BASE/SLINFER_core/tools
```
On Window-dist_gateway (CPU's distributed gateway):
```bash
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
cd $PROJECT_BASE/SLINFER_core/scheduler
```

## 5. Running Experiments

Three experiments are provided: **3B**, **7B**, and **13B** models (corresponding to Fig.22 a/b/c).

Each experiment includes test scripts with different durations:
* `*_full.py`: 33 min per load level (32/64/128), 4 systems → ~396 min total
* `*_lite.py`: 13 min per load level (32/64/128), 4 systems → ~156 min total
* `*_ultra_lite.py`: 13 min, 1 load level, 4 systems → ~52 min
* `*_extreme_lite.py`: 13 min, 1 load level, 2 systems → **~26 min** (recommended for quick tests)
> If you only downloaded one model (e.g., 7B), only run the corresponding experiment.
### Before Each Experiment
#### Terminate all running processes:
* On GPU machine: `Ctrl+C` in Window-0,1,2,3,gateway
* On CPU machines: `Ctrl+C` in Window-0 and Window-dist_gateway

### A. 3B Model Experiment
1. **Choose config:**
```bash
# With 4 CPU machines
cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_3B_4C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py

# GPU-only
cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_3B_0C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py
```
2. Start GPU instances wrapper (in Windows 0–3):
```bash
# Window-0
python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8000 --gpu 0
# Window-1
python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8100 --gpu 1
# Window-2
python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8200 --gpu 2
# Window-3
python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8300 --gpu 3
```
**Wait ~1 minute for initialization.**

3. **(Skip if GPU-only) On each CPU machine:**
```bash
# Window-0
python vllm_batch_starter.py --model llama-3.2-3b --device aliyun --worker_num 4 --port 8000 --cpu_kv_gb 16
# Window-dist_gateway
python dist_gateway.py --port 7999
```
**Wait ~1 minute for initialization.**

4. **On GPU machine, start root gateway** (Window-gateway):
```bash
python gateway.py
```
Wait for `"Start-up complete"` output (~few minutes).

5. **On GPU machine, run test** in `Window-test`:
```bash
python test_3B_extreme_lite.py   # recommended
# OR
python test_3B_ultra_lite.py
python test_3B_lite.py
python test_3B_full.py
```
6. **On GPU machine, Generate plots:**
```bash
cd $PROJECT_BASE/SLINFER_core/tools/draw
python draw.py # The script will ask for GPU number (number of GPU cards), CPU number (number of CPU machines).
# The script will generate a pdf figure and print the file path.
```

### B. 7B Model Experiment
1. **Choose config:**
```bash
# With 4 CPU machines
cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_7B_4C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py

# GPU-only
cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_7B_0C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py

# Debug (1 GPU only)
cp ..._7B_0C1G_debug.py ...
```
2. Start GPU instances wrapper (in Windows 0–3):
```bash
# Window-0
python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8000 --gpu 0
# Window-1
python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8100 --gpu 1
# Window-2
python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8200 --gpu 2
# Window-3
python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8300 --gpu 3
```
**Wait ~1 minute for initialization.**

3. **(Skip if GPU-only) On each CPU machine:**
```bash
# Window-0
python vllm_batch_starter.py --model llama-2-7b --device aliyun --worker_num 2  --port 8000 --cpu_kv_gb 32
# Window-dist_gateway
python dist_gateway.py --port 7999
```
**Wait ~1 minute for initialization.**

4. **On GPU machine, start root gateway** (Window-gateway):
```bash
python gateway.py
```
Wait for `"Start-up complete"` output (~few minutes).

5. **On GPU machine, run test** in `Window-test`:
```bash
python test_7B_extreme_lite.py   # recommended
# OR
python test_7B_ultra_lite.py
python test_7B_lite.py
python test_7B_full.py
```
6. **On GPU machine, Generate plots:**
```bash
cd $PROJECT_BASE/SLINFER_core/tools/draw
python draw.py # The script will ask for GPU number (number of GPU cards), CPU number (number of CPU machines).
# The script will generate a pdf figure and print the file path.
```

### C. 13B Model Experiment
1. **Choose config:**
```bash
# With 4 CPU machines
cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_13B_4C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py

# GPU-only
cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_13B_0C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py
```
2. Start GPU instances wrapper (in Windows 0–3):
```bash
# Window-0
python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8000 --gpu 0
# Window-1
python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8100 --gpu 1
# Window-2
python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8200 --gpu 2
# Window-3
python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8300 --gpu 3
```
**Wait ~1 minute for initialization.**

3. **(Skip if GPU-only) On each CPU machine:**
```bash
# Window-0
python vllm_batch_starter.py --model llama-2-13b --device aliyun --worker_num 1  --port 8000 --cpu_kv_gb 32
# Window-dist_gateway
python dist_gateway.py --port 7999
```
**Wait ~1 minute for initialization.**

4. **On GPU machine, start root gateway** (Window-gateway):
```bash
python gateway.py
```
Wait for `"Start-up complete"` output (~few minutes).

5. **On GPU machine, run test** in `Window-test`:
```bash
python test_13B_extreme_lite.py   # recommended
# OR
python test_13B_ultra_lite.py
python test_13B_lite.py
python test_13B_full.py
```
6. **On GPU machine, Generate plots:**
```bash
cd $PROJECT_BASE/SLINFER_core/tools/draw
python draw.py # The script will ask for GPU number (number of GPU cards), CPU number (number of CPU machines).
# The script will generate a pdf figure and print the file path.
```