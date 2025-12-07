
SLINFER可以运行在任意数量的CPU和GPU上，接下来的步骤教程基于的硬件环境为，有一个GPU机器，包含4张A100-80GB GPU；此外还有四台CPU机器，每台配备32核或更多 4代或更新的至强处理器。
请确保每台机器的端口号，在7000-8999范围内是未被占用的，否则可能出现端口冲突。
SLINFER也可以运行在只有GPU的环境下，若想测试只有GPU的情况，请忽略下文中关于CPU机器的配置步骤。

请在每台机器（在本案例中，1台4卡的GPU机器，4台32核的CPU机器）上下载SLIFNER的代码，随后请把项目所在的根目录的绝对路径设置为环境变量PROJECT_BASE：
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT

使用conda创建虚拟环境，并激活环境
conda create -n SLINFER-GPU python=3.11 （对于GPU机器而言）
conda create -n SLINFER-CPU python=3.11 （对于CPU机器而言）

conda activate SLINFER-GPU （对于GPU机器而言）
conda activate SLINFER-CPU （对于CPU机器而言）

接下来的每步操作，请都确保在conda的虚拟环境之中。


第一部分：安装软件

对于GPU机器：

前置要求：你需要检查/设定nvcc的版本，我们使用的是12.4，更高的版本理论上也可以工作，但未经过测试。
检查：执行nvcc --version 输出为12.4

1. 进行ServerlessLLM的模型加载器的安装
cd $PROJECT_BASE/ServerlessLLM_modify/sllm_store
rm -rf build
pip install .

检查，执行pip list，输出中有serverless-llm-store

2. 进行vLLM的安装
cd $PROJECT_BASE/vLLM_modify
pip install -e .

提示：此处会提示 serverless-llm-store的依赖需求存在冲突，例如下面的信息，请忽略。
serverless-llm-store 0.6.0 requires torch==2.3.0, but you have torch 2.3.1 which is incompatible.
serverless-llm-store 0.6.0 requires transformers==4.42.0, but you have transformers 4.57.3 which is incompatible.

3. 进行依赖包版本的修改，并进行画图相关包的安装
pip install torch==2.3.1 transformers==4.46.3
pip uninstall pyairports -y
pip install git+https://github.com/ozeliger/pyairports.git
pip install matplotlib seaborn
提示：此处会再次提示依赖需求存在冲突，请忽略。



到此，GPU机器的安装完成

对于每个CPU机器：

1. 进行vLLM的安装
cd $PROJECT_BASE/vLLM_modify
pip install -r requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
PIP_PRE=1 PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly/" VLLM_TARGET_DEVICE=openvino python -m pip install -v -e .

到此，CPU机器的安装完成

第二部分：准备模型

在每台机器上：请从HuggingFace上下载Llama-3.2-3B-Instruct, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf这三个模型，存储到$PROJECT_BASE/huggingface_models/下。
为了节约时间，你可以只下载一个模型，例如Llama-2-7b-chat-hf，并进行部分测试，而不是下载三个模型。

下载完成后，应当满足的层级结构和文件名称为：
$PROJECT_BASE
	huggingface_models
		Llama-3.2-3B-Instruct
			*.safetensors
			...
		Llama-2-7b-chat-hf
			*.safetensors
			...
		Llama-2-13b-chat-hf
			*.safetensors
			...

对于GPU节点，为了加速GPU的模型加载，我们将Llama-2-7b-chat-hf与Llama-2-13b-chat-hf转换成ServerlessLLM的loader所支持的格式（Llama-3.2-3B-Instruct与ServerlessLLM存在兼容性问题，故跳过），请在GPU机器上执行
$PROJECT_BASE/huggingface_models/export_gpu_models.sh
执行完成后，应当满足的层级结构和文件名称为：
$PROJECT_BASE
	gpu_models
		Llama-3.2-3B-Instruct
			rank_0
			...
		Llama-2-7b-chat-hf
			rank_0
			...
		Llama-2-13b-chat-hf
			rank_0
			...


对于CPU节点，为了使用Intel OPENVINO的推理后端，我们将模型转换成其所支持的格式，请在CPU机器上执行
$PROJECT_BASE/huggingface_models/export_cpu_models.sh
执行完成后，应当满足的层级结构和文件名称为：
$PROJECT_BASE
	cpu_models
		Llama-3.2-3B-Instruct
			openvino_model.bin
			...
		Llama-2-7b-chat-hf
			openvino_model.bin
			...
		Llama-2-13b-chat-hf
			openvino_model.bin
			...


第三部分：准备实验


整体介绍：在GPU机器上，将运行一个ServerlessLLM的加载器服务，运行一个接受调度请求的根gateway，还需要对每张卡预启动一些推理实例。
在CPU机器上，将运行一个接收来自根gateway转发的请求的分布式的gateway，还需要预启动一些推理实例。


1. 配置网络信息
	注：如果你只需要测试仅GPU的情况，你可以跳过这一小步。
	由于运行在GPU机器上的根gateway可能需要转发请求到CPU机器上的分布式gateway，GPU机器需要知道CPU机器的ip（并可以连接）。因此，你需要编辑以下三个配置文件，并指定CPU机器的IP。
	具体来说：
		scheduler/config_template/pools_info_template_3B_4C4G.py
		代码行83，97，111，125
		scheduler/config_template/pools_info_template_7B_4C4G.py
		代码行67，79，91，103
		scheduler/config_template/pools_info_template_13B_4C4G.py
		代码行59，70，81，92

2. 在GPU机器上：启用nvidia-mps（推荐，可选）
	执行 nvidia-cuda-mps-control -d

3. 在GPU机器上：
	创建6个窗口（例如，通过tmux、screen），以确保不会被中断。请确保每个窗口激活了SLINFER-GPU虚拟环境，且定义了PROJECT_BASE环境变量。
	以下是6个窗口的职责：
	窗口-0：负责运行GPU-0的推理实例
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	export OMP_NUM_THREADS=4
	cd $PROJECT_BASE/SLINFER_core/tools
	窗口-1：负责运行GPU-1的推理实例
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	export OMP_NUM_THREADS=4
	cd $PROJECT_BASE/SLINFER_core/tools
	窗口-2：负责运行GPU-2的推理实例
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	export OMP_NUM_THREADS=4
	cd $PROJECT_BASE/SLINFER_core/tools
	窗口-3：负责运行GPU-3的推理实例
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	export OMP_NUM_THREADS=4
	cd $PROJECT_BASE/SLINFER_core/tools
	窗口-loader：负责运行ServerlessLLM的模型加载服务
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	sllm-store-server --storage_path $PROJECT_BASE/gpu_models --mem_pool_size 64
	请并等待server初始化完成（输出"Server listening on 0.0.0.0:8073"）
	窗口-gateway：负责运行SLINFER的根gateway
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	cd $PROJECT_BASE/SLINFER_core/scheduler
	窗口-test：负责运行测试脚本
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	cd $PROJECT_BASE/SLINFER_core/tools/test

	对这些窗口的后续操作将在后文展开

4. 在每个CPU机器上：
	注：如果你只需要测试仅GPU的情况，你可以跳过这一小步。
	创建2个窗口（例如，通过tmux、screen），以确保不会被中断。请确保每个窗口激活了SLINFER-CPU虚拟环境，且定义了PROJECT_BASE环境变量。
	以下是2个窗口的职责：
	窗口-0：负责运行当前CPU机器的推理实例
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	cd $PROJECT_BASE/SLINFER_core/tools
	窗口-dist_gateway：负责运行SLINFER的分布式gateway
	执行
	export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER/PROJECT/ROOT
	cd $PROJECT_BASE/SLINFER_core/scheduler

	对这些窗口的后续操作将在后文展开


第四部分：开展实验

整体介绍：本部分介绍如何运行实验，主要分为三组实验，分别测试不同sized的模型。
对于每组实验，我们提供三个测试脚本。
对于脚本的文件名，以full后缀结尾的，会运行3种不同的数量组合（32，64，128），且每种数量测试30分钟；以lite结尾的，会运行3种不同的数量组合（32，64，128），但每种数量只测试10分钟；以ultra_lite结尾的，只会运行1种数量组合，且只测试10分钟。
我们推荐运行后缀为extreme_lite的测试脚本以节省时间。

前置说明：在运行每组测试时，请执行以下初始化操作：
在GPU机器上，对于窗口-0,1,2,3，以及窗口-gateway，按ctrl-c，来终止对应的进程。
在CPU机器上，对于窗口-0和窗口-dist_gateway，按ctrl-c，来终止对应的进程。

提示：如果你在此前为了节约时间只下载并转换了一个模型，例如，你只下载了Llama-2-7B，则你只能开展第二个实验，即测试7B规模的模型。

1. 测试3B规模的模型
	(a)运行前置说明里的初始化操作。
	(b)
	以下命令二选一：
	# 如果你有4个CPU机器
	cp scheduler/config_template/pools_info_template_3B_4C4G.py scheduler/config_template/pools_info_template.py
	# 如果你只考虑GPU的情况
	cp scheduler/config_template/pools_info_template_3B_0C4G.py scheduler/config_template/pools_info_template.py
	(c)
	在GPU机器上：
	进入窗口0，运行
	python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8000 --gpu 0
	进入窗口1，运行
	python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8100 --gpu 1
	进入窗口2，运行
	python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8200 --gpu 2
	进入窗口3，运行
	python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8300 --gpu 3

	请等待1分钟左右，因为vllm实例初始化需要时间，否则执行步骤e时可能出错。
	(d)
	如果你只考虑GPU的情况，可跳过这一步
	在每个CPU机器上：
	进入窗口-0，运行
	python vllm_batch_starter.py --model llama-3.2-3b --device aliyun --worker_num 4  --port 8000 --cpu_kv_gb 16
	进入窗口-dist_gateway，运行：
	python dist_gateway.py --port 7999
	请等待1分钟左右，因为vllm实例初始化需要时间，否则执行步骤e时可能出错。
	(e)
	在GPU机器上：
	进入窗口-gateway，运行：
	python gateway.py
	请等待gateway初始化完成（输出Start-up complete"，可能需要几分钟）
	(f)
	在GPU机器上：
	进入窗口-test，以下命令选择其一：
	# 33分钟/个测试 * 4种系统 * 3种模型数量 = 396分钟
	python test_3B_full.py
	# 13分钟/个测试 * 4种系统 * 3种模型数量 = 156分钟
	python test_3B_lite.py
	# 13分钟/个测试 * 4种系统 * 1种模型数量 = 52分钟
	python test_3B_ultra_lite.py
	# 13分钟/个测试 * 2种系统 * 1种模型数量 = 26分钟
	python test_7B_extreme_lite.py

	等待测试脚本运行完成
	(g)
	在GPU机器上，运行以下命令来画图分析，命令会输出图片保存路径
	cd $PROJECT_BASE/SLINFER_core/tools/draw
	python draw.py

2. 测试7B规模的模型
	(a)运行前置说明里的初始化操作。
	(b)
	以下命令二选一：
	# 如果你有4个CPU机器
	cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_7B_4C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py
	# 如果你只考虑GPU的情况
	cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_7B_0C4G.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py
	# 如果debug
	cp $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template_7B_0C1G_debug.py $PROJECT_BASE/SLINFER_core/scheduler/config_template/pools_info_template.py
	(c)
	在GPU机器上：
	进入窗口0，运行
	python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8000 --gpu 0
	进入窗口1，运行
	python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8100 --gpu 1
	进入窗口2，运行
	python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8200 --gpu 2
	进入窗口3，运行
	python vllm_batch_starter.py --model llama-2-7b --device gpu --worker_num 4 --port 8300 --gpu 3
	请等待1分钟左右，因为vllm实例初始化需要时间，否则执行步骤e时可能出错。
	(d)
	如果你只考虑GPU的情况，可跳过这一步
	在每个CPU机器上：
	进入窗口-0，运行
	python vllm_batch_starter.py --model llama-2-7b --device aliyun --worker_num 2  --port 8000 --cpu_kv_gb 32
	进入窗口-dist_gateway，运行：
	python dist_gateway.py --port 7999
	请等待1分钟左右，因为vllm实例初始化需要时间，否则执行步骤e时可能出错。
	(e)
	在GPU机器上：
	进入窗口-gateway，运行：
	python gateway.py
	请等待gateway初始化完成（输出Start-up complete"，可能需要几分钟）
	(f)
	在GPU机器上：
	进入窗口-test，以下命令选择其一：
	# 33分钟/个测试 * 4种系统 * 3种模型数量 = 396分钟
	python test_7B_full.py
	# 13分钟/个测试 * 4种系统 * 3种模型数量 = 156分钟
	python test_7B_lite.py
	# 13分钟/个测试 * 4种系统 * 1种模型数量 = 52分钟
	python test_7B_ultra_lite.py
	# 13分钟/个测试 * 2种系统 * 1种模型数量 = 26分钟
	python test_7B_extreme_lite.py

	等待测试脚本运行完成
	(g)
	在GPU机器上，运行以下命令来画图分析，命令会输出图片保存路径
	cd $PROJECT_BASE/SLINFER_core/tools/draw
	python draw.py

3. 测试13B规模的模型
	(a)运行前置说明里的初始化操作。
	(b)
	以下命令二选一：
	# 如果你有4个CPU机器
	cp scheduler/config_template/pools_info_template_13B_4C4G.py scheduler/config_template/pools_info_template.py
	# 如果你只考虑GPU的情况
	cp scheduler/config_template/pools_info_template_13B_0C4G.py scheduler/config_template/pools_info_template.py
	(c)
	在GPU机器上：
	进入窗口0，运行
	python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8000 --gpu 0
	进入窗口1，运行
	python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8100 --gpu 1
	进入窗口2，运行
	python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8200 --gpu 2
	进入窗口3，运行
	python vllm_batch_starter.py --model llama-2-13b --device gpu --worker_num 2 --port 8300 --gpu 3
	请等待1分钟左右，因为vllm实例初始化需要时间，否则执行步骤e时可能出错。
	(d)
	如果你只考虑GPU的情况，可跳过这一步
	在每个CPU机器上：
	进入窗口-0，运行
	python vllm_batch_starter.py --model llama-2-13b --device aliyun --worker_num 1  --port 8000 --cpu_kv_gb 32
	进入窗口-dist_gateway，运行：
	python dist_gateway.py --port 7999
	请等待1分钟左右，因为vllm实例初始化需要时间，否则执行步骤e时可能出错。
	(e)
	在GPU机器上：
	进入窗口-gateway，运行：
	python gateway.py
	请等待gateway初始化完成（输出Start-up complete"，可能需要几分钟）
	(f)
	在GPU机器上：
	进入窗口-test，以下命令选择其一：
	# 33分钟/个测试 * 4种系统 * 3种模型数量 = 396分钟
	python test_13B_full.py
	# 13分钟/个测试 * 4种系统 * 3种模型数量 = 156分钟
	python test_13B_lite.py
	# 13分钟/个测试 * 4种系统 * 1种模型数量 = 52分钟
	python test_13B_ultra_lite.py
	# 13分钟/个测试 * 2种系统 * 1种模型数量 = 26分钟
	python test_13B_extreme_lite.py

	等待测试脚本运行完成
	(g)
	在GPU机器上，运行以下命令来画图分析，命令会输出图片保存路径
	cd $PROJECT_BASE/SLINFER_core/tools/draw
	python draw.py