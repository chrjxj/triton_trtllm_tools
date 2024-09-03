#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#
# user to modify following according to your path inside the docker container
#

# root dir of your code inside the docker container 
CODE_DIR=/opt/inference_deploy
# root dir of your huggingface model folders and tensorrt engine folders, inside the docker container 
MODEL_DIR=/workspace/models
# root dir of your triton model repos
triton_repo_dir=/workspace/ensemble_models
mkdir -p ${MODEL_DIR} ${triton_repo_dir}


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

function set_param_model_qwen2_72b {
    model_family="qwen"
    hf_name="Qwen/Qwen2-72B-Instruct"
    hf_model_dir_base="Qwen2-72B-Instruct"
    num_gpus=4
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
}


function set_param_model_qwen2_72b_tp8pp1 {
    model_family="qwen"
    hf_name="Qwen/Qwen2-72B-Instruct"
    hf_model_dir_base="Qwen2-72B-Instruct"
    num_gpus=8
    pp_size=1
    tp_size=$(( num_gpus / pp_size ))
}

function set_param_model_qwen2_72b_tp4pp2 {
    model_family="qwen"
    hf_name="Qwen/Qwen2-72B-Instruct"
    hf_model_dir_base="Qwen2-72B-Instruct"
    num_gpus=8
    pp_size=2
    tp_size=$(( num_gpus / pp_size ))
}


function set_param_quant_fp16 {
    echo ""
}

function set_param_quant_fp8 {
    quant_algo="fp8"
    kv_cache_quant_algo="fp8"
    use_fp8_context_fmha="disable"  #need disable on pre-hopper gpu
    calib_size=32
    calib_dataset="cnn_dailymail"          # use local folder if need
}



function set_param_quant_int8_wo {
    qformat="int8_wo"
    kv_cache_dtype="int8"
    calib_dataset="cnn_dailymail"          # use local folder if need
}


# not real quantize, just convert checkpoint
function quantize_model_fp16 {

    checkpoint_dir=${MODEL_DIR}/${hf_model_dir_base}/temp_checkpoint
    mkdir -p ${checkpoint_dir} && rm -rf ${checkpoint_dir}/*

    # TODO: check /opt/tensorrt_llm/examples/${model_family}
    python /opt/tensorrt_llm/examples/${model_family}/convert_checkpoint.py \
            --model_dir ${MODEL_DIR}/${hf_model_dir_base} \
            --output_dir ${checkpoint_dir} \
            --tp_size ${tp_size} \
            --pp_size ${pp_size} \
            --dtype float16 
}

function quantize_model_fp8 {

    checkpoint_dir=${MODEL_DIR}/${hf_model_dir_base}/temp_checkpoint
    mkdir -p ${checkpoint_dir} && rm -rf ${checkpoint_dir}/*

    python /opt/tensorrt_llm/examples/quantization/quantize.py \
            --model_dir ${MODEL_DIR}/${hf_model_dir_base} \
            --output_dir ${checkpoint_dir} \
            --tp_size ${tp_size} \
            --pp_size ${pp_size} \
            --dtype float16 \
            --qformat fp8 \
            --kv_cache_dtype fp8 \
            --calib_size ${calib_size:-32} \
            --calib_dataset ${calib_dataset:-cnn_dailymail}

}


function quantize_model_int8_wo {

    checkpoint_dir=${MODEL_DIR}/${hf_model_dir_base}/temp_checkpoint
    mkdir -p ${checkpoint_dir} && rm -rf ${checkpoint_dir}/*

    python /opt/tensorrt_llm/examples/quantization/quantize.py \
            --model_dir ${MODEL_DIR}/${hf_model_dir_base} \
            --output_dir ${checkpoint_dir} \
            --tp_size ${tp_size} \
            --pp_size ${pp_size} \
            --dtype float16 \
            --qformat ${qformat:int8_wo} \
            --kv_cache_dtype ${kv_cache_dtype:int8} \
            --calib_size ${calib_size:-32} \
            --calib_dataset ${calib_dataset:-cnn_dailymail}

}

function build_engine {
    # note for v0.11.0 or above, "--max_output_len has been deprecated in favor of --max_seq_len"
    trtllm-build --checkpoint_dir ${checkpoint_dir} \
                --output_dir ${engine_dir} \
                --workers ${num_gpus} \
                --max_input_len ${max_input_len} \
                --max_output_len ${max_output_len} \
                --max_batch_size ${max_batch_size} \
                --paged_kv_cache ${paged_kv_cache:-enable} \
                --use_custom_all_reduce ${use_custom_all_reduce:-enable} \
                --context_fmha ${context_fmha:-enable} \
                --use_paged_context_fmha ${use_paged_context_fmha:-disable} \
                --use_fp8_context_fmha ${use_fp8_context_fmha:-disable} \
                --gpt_attention_plugin float16 \
                --gemm_plugin float16 \
                --enable_xqa ${enable_xqa:-enable} \
                --multi_block_mode ${enable_longcontext:-disable}
}

function fill_triton_repo() {

    echo "----------------------------------"
    echo " config ensemble_models repo for triton server "
    echo "----------------------------------"

    local HF_MODEL_DIR=$1
    local TRT_ENGINE_DIR=$2
    local MAX_BATCH_SIZE=$3
    local CODE_DIR=$4
    local TARGET_DIR=$5
    local base_name=$6

    # clean up old folder
    rm -rf ${TARGET_DIR}/${base_name} # && mkdir -p ${TARGET_DIR}/${base_name}
    # copy template to target
    cp -rf ${CODE_DIR}/ensemble_models_template/ifb ${TARGET_DIR}/${base_name}

    echo "----------------------------------"
    echo " fill_triton_repo_streaming params "
    echo " HF_MODEL_DIR ${HF_MODEL_DIR} "
    echo " TRT_ENGINE_DIR ${TRT_ENGINE_DIR} "
    echo " MAX_BATCH_SIZE ${MAX_BATCH_SIZE} "
    echo " TARGET_DIR ${TARGET_DIR}/${base_name}"
    echo "----------------------------------"

    # run fill_template and modify config.pbtxt
    python3 ${CODE_DIR}/scripts/fill_template.py -i ${TARGET_DIR}/${base_name}/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1,add_special_tokens:True
    python3 ${CODE_DIR}/scripts/fill_template.py -i ${TARGET_DIR}/${base_name}/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1,skip_special_tokens:True
    python3 ${CODE_DIR}/scripts/fill_template.py -i ${TARGET_DIR}/${base_name}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${triton_streaming_mode},bls_instance_count:1,accumulate_tokens:False
    python3 ${CODE_DIR}/scripts/fill_template.py -i ${TARGET_DIR}/${base_name}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
    python3 ${CODE_DIR}/scripts/fill_template.py -i ${TARGET_DIR}/${base_name}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${triton_streaming_mode},max_beam_width:1,engine_dir:${TRT_ENGINE_DIR},max_tokens_in_paged_kv_cache:8560,max_attention_window_size:8560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0

}



function download_hf_model() {
    local HF_MODEL=$1
    local HF_MODEL_DIR=$2
    # Check if the file exists
    if [ -f "$HF_MODEL_DIR" ] && [ "$(ls -A $HF_MODEL_DIR)" ]; then
        echo "Folder exists: $HF_MODEL_DIR; skip download."
    else
        echo "----------------------------------"
        echo " download HF model $HF_MODEL to folder $HF_MODEL_DIR "
        echo "----------------------------------"
        python3 ${CODE_DIR}/scripts/download_hf.py $HF_MODEL $HF_MODEL_DIR
    fi
}

for model_size in  qwen2_7b; do
# for model_size in  qwen2_7b qwen2_72b qwen2_72b_tp4pp2 qwen2_72b_tp8pp1; do

    for precision in fp8; do   
        set_param_model_${model_size}
        set_param_quant_${precision}
        
        hf_model_dir=${MODEL_DIR}/${hf_model_dir_base}
        download_hf_model hf_name $hf_model_dir

        output_dir_base=${model_size}_${precision}_tp${tp_size}_pp${pp_size}
        output_dir=${MODEL_DIR}/${output_dir_base}
        
        for input_len_output_len in ${input_len_output_len_list}; do

            IFS=, read -r max_input_len max_output_len <<< ${input_len_output_len}

            for max_batch_size in ${batch_size_list}; do
                engine_base_name=isl${max_input_len}_osl${max_output_len}_bs${max_batch_size}
                engine_dir=${output_dir}/${engine_base_name}
                mkdir -p ${engine_dir}
                config_file_path=${engine_dir}/config.json
                echo "----------------------------------"
                echo " Processing for ${output_dir_base}"
                echo "----------------------------------"
                
                if [[ ! -f ${config_file_path} ]]; then
                    echo "running ${precision} quantization for ${output_dir_base}"
                    quantize_model_${precision}
                    echo "building TRT engine for ${engine_base_name}"
                    build_engine
                fi

                echo "generating triton repo config for ${engine_base_name}"
                fill_triton_repo ${hf_model_dir} ${engine_dir}  ${max_batch_size} ${CODE_DIR} ${triton_repo_dir} ${engine_base_name}

            done
        done
    done
done
