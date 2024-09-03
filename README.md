# Automated Deployment and Testing Tool for TensorRT-LLM and its Triton Backend

[中文版](./README_zh.md)

## 1.User Requirements

In the current versions of TensorRT-LLM and its corresponding Triton Backend, there are multiple steps involved in model downloading, quantization, calibration, TensorRT engine building, and Triton configuration file setup. Manual configuration of these steps can be cumbersome and repetitive.


Users perfer an end-to-end solution to test the performance of real LLM models using actual enterprise data (prompts), calculate TTFT, separately compute Input/Output token throughput, and measure accuracy metrics under real business data conditions.


## 2.Building the Container

- First, build the  docker image for Triton TensorRT-LLM backend following the steps in https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#option-2-build-via-docker. 
    * For example, using TAG v0.11.0, name/tag the docker image to: `nvcr.io/nvidian/sae/customer_triton_trt_llm:v0.11.0`

- Use `Dockerfile.trtllm_v0.11.0` in this folder to build another docker image based on `nvcr.io/nvidian/sae/customer_triton_trt_llm:v0.11.0`, packaging relevant code and scripts into the docker image for convenient future use: `docker build -t triton_trt_llm:v0.11.0 -f Dockerfile.trtllm_v0.11.0 .`


The following experiments run within the triton_trt_llm:v0.11.0 container.


## 3. Automated Deployment Tool

Edit the configuration in `inference_deploy/deploy_qwen.sh` according to experimental needs.

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

Notes:

* hf_name: model's huggingface name; 
* hf_model_dir_base: local foler path to store your huggingface model
* batch_size_list: A list of TensorRT engine max_batch_size; also triton_max_batch_size in Triton's dynamic_batching configuration.
input_len_output_len_list: A list of input sequence length and output sequence length combinations.
* triton_streaming_mode: The decoupled_mode of the tesorrt-llm Triton backend, where True indicates streaming mode.
* Modify the loop lists of model_size and precision according to experimental needs. The selectable lists need to have corresponding set_param_model_ and set_param_quant_ functions.

**Run Deployment Script**

* If the corresponding Huggingface LLM model has not been downloaded to the local folder, first set the HF_TOKEN environment variable: `export HF_TOKEN=<YOUR_HF_TOKEN>`

* Run `/opt/inference_deploy/deploy_qwen.sh` to perform the steps of model downloading, quantization, calibration, TensorRT engine compilation, and Triton configuration file setup.


**Run Triton Server**

Example:

```
python3 /opt/scripts/launch_triton_server.py  \
    --model_repo=/workspace/ensemble_models/qwen2_7b_fp8_tp1_pp1_isl2048_osl1024_bs8 \
    --world_size 1 --grpc_port 18001 --http_port 18000
```



## 4. A Python Perf tool for TRT-LLM and its Triton Backend

### Background and Motiviations

Customer want to use real world data (prompts) and test real model (instead of random weights).

Triton's `perf_analyzer` is general for any request, not designed for LLM. 
Triton's `GenAI-Perf` is built on top of `perf_analyzer`, closer to the user needs, but still miss some function support and flexibility, e.g. count tokens, supporting streaming mode. 

Givn the LLM inference latency, python's I/O performance won't likely inject signficant overhead on client side when sending/generating requests.


### Prompt data format

Prepare your prompt data in a typical "jsonl" file, with "input" key in each line:

```
{"input": "prompt 1...", "output": "..."}
{"input": "prompt 2...", "output": "..."}
...
```

### Command line Usage

Command line usage example:

```bash
python3 llm_perf.py \
        -u localhost:9001 -i grpc -m ensemble \
        --concurrency 2,4,8,16 \
        --input-data prompts.jsonl --tokenizer-path /models/Qwen2-7B

```

About command line arguments: 

```bash
-u <URL for inference service>
-i <Protocol used to communicate with inference service>, "http" or "grpc"
-m: This is a required argument and is used to specify the triton model name against which to run
--concurrency a list of concurrency on client side, e.g. "1,4,8"
--input-data <<file path>>
--profile-export-file <<file path>>
```

### Cross check the perf result 

To verify the reliability of the tool's results, Triton's built-in `perf_analyzer` or `GenAI-Perf` can be used for comparison and validation.


**Result Using Triton's perf_analyzer**

Run `perf_analyzer` to cross check perf results. For example:

```bash

perf_analyzer  -u localhost:9001 -i grpc     -m ensemble --async --input-data prompts_perfanalyzer.json     --measurement-interval 10000      --profile-export-file qwen2-7b-1024_1.json      --service-kind triton      -v --endpoint v1/chat/completions      --concurrency-range 6:8:1      --stability-percentage 999      --streaming --shape max_tokens:1 --shape text_input:1

```