#!/usr/bin/env python3
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

"""
This tool can perf the LLM inference service hosted by
Triton Inference Server (https://github.com/triton-inference-server/tensorrtllm_backend).

Usage example:

    python3 llm_perf.py \
            -u localhost:9001 -i grpc -m ensemble \
            --concurrency 2,4,8,16 \
            --tokenizer-path /models/Qwen2-7B \
            --input-data prompts.jsonl

Command line args:
    -u <URL for inference service>
    -i <Protocol used to communicate with inference service>
    -m: This is a required argument and is used to specify the model name
    --concurrency a list of concurrency on client side, e.g. "1,4,8"
    --input-data <<file path>>
    --tokenizer-path <<tokenizer folder path>>
    --profile-export-file <<file path, save profile results>>
"""

import os
import time
import random
import argparse
import sys
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Union, Optional, List
from pydantic import BaseModel
from transformers import (
    AutoTokenizer
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from trt_llm import HttpTritonClient, GrpcTritonClient


class Payload(BaseModel):
    prompt: List[List[str]]
    tokens:  int = 255
    temperature:  float = 1.0
    top_k:  int = 1
    top_p:  float = 0
    beam_width:  int = 1
    repetition_penalty:  float = 1.0
    length_penalty: float = 1.0


def send_request(payload, triton_url="", protocol="grpc", model_name="ensemble"):

    perf_result = dict(
        output_text="",
        ttft=-1,
        e2e_latency=-1,
    )

    if isinstance(payload, dict):
        pass
    elif isinstance(payload, Payload):
        payload = payload.dict()
    else:
        raise TypeError

    if protocol.lower() == "grpc":
        triton_client = GrpcTritonClient(triton_url)
        triton_client.load_model(model_name)

        start_time = time.time()
        tokens_generated = 0
        output_text = ""
        for val in triton_client.request_streaming(model_name, request_id=str(random.getrandbits(64)), **payload):
            tokens_generated += 1
            output_text += val
            if tokens_generated == 1:
                ttft = time.time() - start_time

        perf_result["output_text"] = output_text
        perf_result["ttft"] = ttft
        perf_result["e2e_latency"] = time.time() - start_time
        #perf_result["e2e_latency"] = ttft if perf_result["e2e_latency"] <= ttft else perf_result["e2e_latency"]

    elif protocol.lower() == "http":
        triton_client = HttpTritonClient(triton_url)
        triton_client.load_model(model_name)

        start_time = time.time()
        res = triton_client.request(model_name, **payload)
        perf_result["e2e_latency"] = time.time() - start_time
        perf_result["output_text"] = res

        total_time = time.time() - start_time
        # print(f"\n--- used {total_time} seconds ---")

    else:
        raise ValueError

    return perf_result


def load_payload_from_json(json_file):

    df = pd.read_json(json_file, lines=True)

    if "text_input" not in df.columns:
        raise ValueError("Json file must provide prompts (input text) using **text_input** flied. but got: {}".format(df.columns))

    payloads = [
        {
            'prompt': [[prompt]],
            'tokens': 300,
            'temperature': 1.0,
            'top_k': 1,
            'top_p': 0,
            'beam_width': 1,
            'repetition_penalty': 1.0,
            'length_penalty': 1.0
        }
        for prompt in df.text_input
    ]
    return payloads


def calculate_perf_stats(perf_results, decimal_digits = 2):

    # # calc stats
    # metrics = {
    #     'Metrics': ["Average", "P99", "P90", "P75"],
    #     'TTFT (sec)': [
    #             round(np.average([item["ttft"] for item in perf_results]), decimal_digits),
    #             round(np.percentile([item["ttft"] for item in perf_results], 99), decimal_digits),
    #             round(np.percentile([item["ttft"] for item in perf_results], 90), decimal_digits),
    #             round(np.percentile([item["ttft"] for item in perf_results], 75), decimal_digits),

    #             ],
    #     'E2E_Latency (sec)': [
    #             round(np.average([item["e2e_latency"] for item in perf_results]), decimal_digits),
    #             round(np.percentile([item["e2e_latency"] for item in perf_results], 99), decimal_digits),
    #             round(np.percentile([item["e2e_latency"] for item in perf_results], 90), decimal_digits),
    #             round(np.percentile([item["e2e_latency"] for item in perf_results], 75), decimal_digits),
    #             ],
    #     'TPOT (sec)': [
    #             round(np.average([item["TPOT"] for item in perf_results]), decimal_digits),
    #             round(np.percentile([item["TPOT"] for item in perf_results], 99), decimal_digits),
    #             round(np.percentile([item["TPOT"] for item in perf_results], 90), decimal_digits),
    #             round(np.percentile([item["TPOT"] for item in perf_results], 75), decimal_digits),
    #             ],
    # }

    # calc stats
    metrics = {
        'Metrics': ["Average", "Min", "Max", "P99", "P90", "P75"],
        'Time to first token (ms)': [
                round(np.average([item["ttft"]    *1000 for item in perf_results]), decimal_digits),
                round(min([item["ttft"]    *1000 for item in perf_results]), decimal_digits),
                round(max([item["ttft"]    *1000 for item in perf_results]), decimal_digits),
                round(np.percentile([item["ttft"] *1000 for item in perf_results], 99), decimal_digits),
                round(np.percentile([item["ttft"] *1000 for item in perf_results], 90), decimal_digits),
                round(np.percentile([item["ttft"] *1000 for item in perf_results], 75), decimal_digits),

                ],
        'Inter token latency (ms)': [
                round(np.average([item["TPOT"]    *1000 for item in perf_results]), decimal_digits),
                round(min([item["TPOT"]    *1000 for item in perf_results]), decimal_digits),
                round(max([item["TPOT"]    *1000 for item in perf_results]), decimal_digits),
                round(np.percentile([item["TPOT"] *1000 for item in perf_results], 99), decimal_digits),
                round(np.percentile([item["TPOT"] *1000 for item in perf_results], 90), decimal_digits),
                round(np.percentile([item["TPOT"] *1000 for item in perf_results], 75), decimal_digits),
                ],
        'Request latency (ms)': [
                round(np.average([item["e2e_latency"]    *1000 for item in perf_results]), decimal_digits),
                round(min([item["e2e_latency"]    *1000 for item in perf_results]), decimal_digits),
                round(max([item["e2e_latency"]    *1000 for item in perf_results]), decimal_digits),
                round(np.percentile([item["e2e_latency"] *1000 for item in perf_results], 99), decimal_digits),
                round(np.percentile([item["e2e_latency"] *1000 for item in perf_results], 90), decimal_digits),
                round(np.percentile([item["e2e_latency"] *1000 for item in perf_results], 75), decimal_digits),
                ],
    }

    df = pd.DataFrame(metrics)
    df_indexed = df.set_index("Metrics")
    print(df_indexed.T.to_markdown())

    return metrics


def list_of_ints(arg):
	return list(map(int, arg.split(',')))

def main(args):

    triton_url = args.uri
    model_name = args.triton_model
    protocol = args.protocol.lower()
    print(f"triton_url: {triton_url}, protocol: {protocol}, model: {model_name}")

    assert(protocol in ["http", "grpc"])
    assert(isinstance(args.concurrency, list))

    send_request_ensemble = partial(
        send_request, triton_url=triton_url, protocol=protocol, model_name=model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False

    payloads = load_payload_from_json(args.input_data)

    full_results = dict()

    for con in args.concurrency:
        perf_results = []
        print("-"*60)
        print(f"Request concurrency: {con}")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=con) as executor:
            perf_results += list(executor.map(send_request_ensemble, payloads))

        # the end-to-end latency between the first request and the last response of the last request.
        total_time = time.time() - start_time

        # post process
        for payload, result in zip(payloads, perf_results):
            result["input_tokens"] = len(tokenizer.encode(payload["prompt"][0][0]))
            result["output_tokens"] = len(tokenizer.encode(result["output_text"]))
            #  time per output token (TPOT). or avg. Inter-token Latency (ITL)
            result["TPOT"] = (result["e2e_latency"] - result["ttft"]) / (result["output_tokens"]-1)

            # result["input_token_per_sec"] = perf_results["input_tokens"] / perf_results["e2e_latency"]
            # result["output_token_per_sec"] = perf_results["output_tokens"] / perf_results["e2e_latency"]

        metrics_stats = calculate_perf_stats(perf_results)

        # Total TPS per system represents the total output tokens per seconds throughput, accounting for all the requests happening simultaneously.
        total_output_tps_in_system = sum([item["output_tokens"] for item in perf_results]) / total_time
        total_input_tps_in_system = sum([item["input_tokens"] for item in perf_results]) / total_time

        # TODO: use actual completed
        # average number of requests that can be successfully completed by the system
        request_per_sec = len(perf_results) / total_time

        full_results[con] = dict(
            concurrency = con,
            client_requests=len(payloads),
            perf_results=perf_results,
            metrics_stats=metrics_stats,
            total_output_tps_in_system=total_output_tps_in_system,
            total_input_tps_in_system=total_input_tps_in_system,
            request_per_sec=request_per_sec,
        )

        print(f"num of payload: {len(payloads)}; completed results: {len(perf_results)}")
        print(f"request per sec (throughput in system): {round(request_per_sec,4)} infer/sec")
        print(f"total input token per sec (in system): {round(total_input_tps_in_system,4)} token/sec")
        print(f"total output token per sec (in system): {round(total_output_tps_in_system,4)} token/sec")
        print("\n")
        print("-"*60)

        time.sleep(args.measurement_interval)

    # after all case, save results to a text file
    with open(args.profile_export_file, "w") as fp:
        json.dump(full_results, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('-u', '--uri', type=str, default='localhost:8001',
                        help='URL for inference service; host name or IP address and port')
    parser.add_argument('-i', "--protocol", type=str, default='grpc',
                        help='Protocol used to communicate with inference service. http or grpc')
    parser.add_argument('-m', '--triton-model', type=str, default="ensemble", help='specify the triton model against which to run')
    parser.add_argument('--concurrency', type=list_of_ints, default="1,4,8",
                        help='num of concurrency requests to send out')
    parser.add_argument('--measurement-interval', type=int, default=30,
                        help='measurement interval in sec')
    parser.add_argument('--input-data', type=str,  help='path to input prompts/jsonl file')
    parser.add_argument('--tokenizer-path', type=str,  help='path to tokenizer folder')
    parser.add_argument('--profile-export-file', type=str, default='profiling_results.json', help='dump profiling results to output file')

    args = parser.parse_args()

    main(args)

    print("\nNote: Metrics definition in the guide for details: https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html")
    print("-"*60)
    print("\n")
