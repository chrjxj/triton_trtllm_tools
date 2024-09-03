# A Python Perf tool for TRT-LLM and its Triton Backend

### Background and Motiviations

Customer want to use real world data (prompts) and test real model (instead of random weights).

Triton's `perf_analyzer` is general for any request, not designed for LLM. 
Triton's `GenAI-Perf` is built on top of `perf_analyzer`, closer to the user needs, but still miss some function support and flexibility, e.g. count tokens, supporting streaming mode. 

Givn the LLM inference latency, python's I/O performance won't likely inject signficant overhead on client side when sending/generating requests.


### Prompt data format

a typical "jsonl" file, with "input" key in each line:

```
{"input": "prompt 1...", "output": "..."}
{"input": "prompt 2...", "output": "..."}
...
```

### Command line Usage

### Cross check the perf result 

**Result Using this Tool**

command: 

```bash

python3 llm_perf.py    \
    -u localhost:9001 -i grpc -m ensemble    \
    --concurrency 1,4,8  --input-data ./prompts.jsonl  \
    --tokenizer-path /workspace/models/Qwen2-7B
```

Output:

```bash

triton_url: localhost:9001, protocol: grpc, model: ensemble
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
------------------------------------------------------------
Request concurrency: 1
average_input_token_per_sec: 7830.079494413578
average_output_token_per_sec: 3.7117065991177514
average_ttft: 0.03435562758944755
average_latency: 0.03720159222220266
p99_ttft: 0.03717458248138428
p99_latency: 0.10233128070831299
payload and results: 1376 1376
throughput: 23.4745319777834 infer/sec.
------------------------------------------------------------
Request concurrency: 4

average_input_token_per_sec: 2399.1026486293395
average_output_token_per_sec: 1.137250923088528
average_ttft: 0.11386719952489055
average_latency: 0.12141682415507561
p99_ttft: 0.12180376052856445
p99_latency: 0.2988126873970032
payload and results: 1376 1376
throughput: 32.68018036492472 infer/sec.
------------------------------------------------------------
Request concurrency: 8
average_input_token_per_sec: 1243.584730614387
average_output_token_per_sec: 0.589498695955345
average_ttft: 0.21896617253159367
average_latency: 0.23423528549976128
p99_ttft: 0.2427530288696289
p99_latency: 0.6333500146865845
payload and results: 1376 1376
throughput: 33.87773848529009 infer/sec.

```



**Result Using Triton's perf_analyzer**


command: 

```bash

perf_analyzer  -u localhost:9001 -i grpc     -m ensemble --async --input-data prompts_perfanalyzer.json     --measurement-interval 10000      --profile-export-file qwen2-7b-1024_1.json      --service-kind triton      -v --endpoint v1/chat/completions      --concurrency-range 6:8:1      --stability-percentage 999      --streaming --shape max_tokens:1 --shape text_input:1

```

Output:

```bash

 Successfully read data for 1 stream/streams with 1376 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: TRITON
  Using "time_windows" mode for stabilization
  Stabilizing using average throughput
  Measurement window: 10000 msec
  Latency limit: 0 msec
  Concurrency limit: 8 concurrent requests
  Using asynchronous calls for inference
  Detected decoupled model, using the first response for measuring latency

Request concurrency: 6
  Pass [1] throughput: 33.8321 infer/sec. Avg latency: 176620 usec (std 60100 usec).
  Pass [2] throughput: 33.5781 infer/sec. Avg latency: 178174 usec (std 40172 usec).
  Pass [3] throughput: 33.1613 infer/sec. Avg latency: 179927 usec (std 71863 usec).
  Client:
    Request count: 1207
    Throughput: 33.5238 infer/sec
    Response Throughput: 38.7177 infer/sec
    Avg client overhead: 0.01%
    Avg latency: 178229 usec (standard deviation 58760 usec)
    p50 latency: 177093 usec
    p90 latency: 178921 usec
    p95 latency: 179616 usec
    p99 latency: 543558 usec

  Server:
    Inference count: 1207
    Execution count: 1207
    Successful request count: 1207
    Avg request latency: 177344 usec (overhead 29 usec + queue 1368 usec + compute 175947 usec)

  Composing models:
  postprocessing, version: 1
      Inference count: 1394
      Execution count: 1394
      Successful request count: 1394
      Avg request latency: 721 usec (overhead 3 usec + queue 481 usec + compute input 24 usec + compute infer 156 usec + compute output 56 usec)

  preprocessing, version: 1
      Inference count: 1213
      Execution count: 1213
      Successful request count: 1213
      Avg request latency: 1511 usec (overhead 3 usec + queue 838 usec + compute input 21 usec + compute infer 584 usec + compute output 64 usec)

  tensorrt_llm, version: 1
      Inference count: 1207
      Execution count: 1207
      Successful request count: 1207
      Avg request latency: 175090 usec (overhead 1 usec + queue 49 usec + compute input 22 usec + compute infer 174927 usec + compute output 89 usec)

Request concurrency: 7
  Pass [1] throughput: 33.9992 infer/sec. Avg latency: 206523 usec (std 77695 usec).
  Pass [2] throughput: 33.9111 infer/sec. Avg latency: 206535 usec (std 19095 usec).
  Pass [3] throughput: 33.4111 infer/sec. Avg latency: 208196 usec (std 87954 usec).
  Client:
    Request count: 1216
    Throughput: 33.7738 infer/sec
    Response Throughput: 38.6344 infer/sec
    Avg client overhead: 0.01%
    Avg latency: 207079 usec (standard deviation 68494 usec)
    p50 latency: 206087 usec
    p90 latency: 207785 usec
    p95 latency: 208575 usec
    p99 latency: 542962 usec

  Server:
    Inference count: 1216
    Execution count: 1216
    Successful request count: 1216
    Avg request latency: 206149 usec (overhead 22 usec + queue 1370 usec + compute 204757 usec)

  Composing models:
  postprocessing, version: 1
      Inference count: 1396
      Execution count: 1396
      Successful request count: 1396
      Avg request latency: 732 usec (overhead 3 usec + queue 495 usec + compute input 21 usec + compute infer 158 usec + compute output 54 usec)

  preprocessing, version: 1
      Inference count: 1217
      Execution count: 1217
      Successful request count: 1217
      Avg request latency: 1499 usec (overhead 3 usec + queue 848 usec + compute input 21 usec + compute infer 566 usec + compute output 60 usec)

  tensorrt_llm, version: 1
      Inference count: 1216
      Execution count: 1216
      Successful request count: 1216
      Avg request latency: 203903 usec (overhead 1 usec + queue 27 usec + compute input 22 usec + compute infer 203765 usec + compute output 87 usec)

Request concurrency: 8
  Pass [1] throughput: 33.9156 infer/sec. Avg latency: 236511 usec (std 84874 usec).
  Pass [2] throughput: 33.9105 infer/sec. Avg latency: 235124 usec (std 58805 usec).
  Pass [3] throughput: 33.6609 infer/sec. Avg latency: 238354 usec (std 86544 usec).
  Client:
    Request count: 1218
    Throughput: 33.829 infer/sec
    Response Throughput: 38.4951 infer/sec
    Avg client overhead: 0.01%
    Avg latency: 236659 usec (standard deviation 77709 usec)
    p50 latency: 234745 usec
    p90 latency: 236506 usec
    p95 latency: 237074 usec
    p99 latency: 685800 usec

  Server:
    Inference count: 1218
    Execution count: 1218
    Successful request count: 1218
    Avg request latency: 235734 usec (overhead 33 usec + queue 1659 usec + compute 234042 usec)

  Composing models:
  postprocessing, version: 1
      Inference count: 1381
      Execution count: 1381
      Successful request count: 1381
      Avg request latency: 830 usec (overhead 3 usec + queue 603 usec + compute input 20 usec + compute infer 147 usec + compute output 55 usec)

  preprocessing, version: 1
      Inference count: 1219
      Execution count: 1219
      Successful request count: 1219
      Avg request latency: 1637 usec (overhead 3 usec + queue 1016 usec + compute input 19 usec + compute infer 547 usec + compute output 51 usec)

  tensorrt_llm, version: 1
      Inference count: 1218
      Execution count: 1218
      Successful request count: 1218
      Avg request latency: 233241 usec (overhead 1 usec + queue 40 usec + compute input 18 usec + compute infer 233090 usec + compute output 91 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 6, throughput: 33.5238 infer/sec, latency 178229 usec
Concurrency: 7, throughput: 33.7738 infer/sec, latency 207079 usec
Concurrency: 8, throughput: 33.829 infer/sec, latency 236659 usec
```

