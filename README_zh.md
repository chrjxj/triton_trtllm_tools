# TensorRT-LLM及其Triton Backend自动化部署和测试工具

[English Version](./README.md)

## 1.用户需求

在当前的TensorRT-LLM和对应的Triton Backend版本中，模型的下载，量化，校准，TensorRT引擎编译，Triton配置文件的设置等步骤比较多，手动配置比较繁琐重复。

用户希望能端到端的，使用真实业务数据(prompts)测试真实LLM的性能，统计TTFT，分别计算Input/Output token throughput，以及计算真实业务数据下的的准确率指标


## 2.编译容器

- 先根据 https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#option-2-build-via-docker tensorrtllm_backend 中的步骤构建容器，以 TAG v0.11.0为例，把容器镜像命名为 `nvcr.io/nvidian/sae/customer_triton_trt_llm:v0.11.0`

- 使用 `Dockerfile.trtllm_v0.11.0`，基于 `nvcr.io/nvidian/sae/customer_triton_trt_llm:v0.11.0`构建容器，把相关的代码和脚本打包到容器中方便后续使用: `docker build -t triton_trt_llm:v0.11.0 -f Dockerfile.trtllm_v0.11.0 . `

以下实验在 `triton_trt_llm:v0.11.0` 容器内完成。

## 3. 自动化部署工具

**编辑配置**

根据实验需要，编辑 `inference_deploy/deploy_qwen.sh` 中的配置

```bash
(...skip...)

# example, batch_size_list="1 4 8 16 32 64 128 256"
batch_size_list="4 8"

#input_len_output_len_list="128,2048 2048,128 2048,2048, 8192,4096"
input_len_output_len_list="2048,1024"
# tesorrt-llm triton backend's decoupled_mode 
triton_streaming_mode=True

function set_param_model_qwen2_7b {
    model_family="qwen"
    hf_name="Qwen/Qwen2-7B"
    hf_model_dir_base="Qwen2-7B"
    num_gpus=1
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
}

(...skip...)

for model_size in  qwen2_7b qwen2_72b qwen2_72b_tp4pp2 qwen2_72b_tp8pp1; do
    for precision in fp8 fp16 int8_wo; do   
      ...

```

说明: 

- batch_size_list: TensorRT engine max_batch_size的列表; 也是Triton中 dynamic_batching配置中的 triton_max_batch_size
- input_len_output_len_list: input seqence length和 output seqence length组合的列表
- triton_streaming_mode: tesorrt-llm triton backend的 decoupled_mode, True为streaming模式
- 根据实验需要，修改 model_size 和 precision的循环列表。可选的列表需要有对应的 `set_param_model_`函数， `set_param_quant_`函数

**运行自动化脚本**

- 如果对应的Huggingface LLM模型没有下载到本地文件夹，首先设置HF_TOKEN环境变量: `export HF_TOKEN=<YOUR_HF_TOKEN>`

- 运行 `/opt/inference_deploy/deploy_qwen.sh` 来执行模型的下载，量化，校准，TensorRT引擎编译，Triton配置文件的设置的步骤

**运行Triton Server**

示例:

```
python3 /opt/scripts/launch_triton_server.py  \
    --model_repo=/workspace/ensemble_models/qwen2_7b_fp8_tp1_pp1_isl2048_osl1024_bs8 \
    --world_size 1 --grpc_port 18001 --http_port 18000
```



## 4. A Python Perf tool for TRT-LLM and its Triton Backend

### 背景

用户希望使用真实世界的数据（提示词）并测试真实模型（而不是随机权重）。

Triton的`perf_analyzer`是通用的，适用于任何模型，但并非专为大语言模型（LLM）设计。

Triton的`GenAI-Perf`是在perf_analyzer基础上构建的，更接近用户需求，但仍然缺少一些功能支持和灵活性，例如计算token数量和吞吐、支持流式模式等。

考虑到LLM推理的延迟，Python的I/O性能在客户端发送/生成请求时不太可能引入显著的开销。

### 数据格式

将你的提示数据准备在一个典型的"jsonl"文件中，每行包含一个"input"键：

```
{"input": "prompt 1...", "output": "..."}
{"input": "prompt 2...", "output": "..."}
...
```

### 命令使用说明

命令使用的例子:

```bash
python3 llm_perf.py \
        -u localhost:9001 -i grpc -m ensemble \
        --concurrency 2,4,8,16 \
        --input-data prompts.jsonl --tokenizer-path /models/Qwen2-7B

```

命令行参数说明: 

```bash
-u <URL for inference service>
-i <Protocol used to communicate with inference service>, "http" or "grpc"
-m: This is a required argument and is used to specify the triton model name against which to run
--concurrency a list of concurrency on client side, e.g. "1,4,8"
--input-data <<file path>>
--profile-export-file <<file path>>
```

### 对比测试结果

为了验证工具的结果可靠，可以使用Triton自带的 `perf_analyzer` 或 `GenAI-Perf` 作为对比验证。

使用 `perf_analyzer` 验证，列如:

```bash

perf_analyzer  -u localhost:9001 -i grpc     -m ensemble --async --input-data prompts_perfanalyzer.json     --measurement-interval 10000      --profile-export-file qwen2-7b-1024_1.json      --service-kind triton      -v --endpoint v1/chat/completions      --concurrency-range 6:8:1      --stability-percentage 999      --streaming --shape max_tokens:1 --shape text_input:1

```
