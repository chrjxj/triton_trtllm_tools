# A Python Perf tool for TRT-LLM and its Triton Backend

### Background and Motiviations

Customer want to use real world data (prompts) and test real model (instead of random weights).

Triton's `perf_analyzer` is general for any request, not designed for LLM.
Triton's `GenAI-Perf` is built on top of `perf_analyzer`, closer to the user needs

Givn the LLM inference latency, python's I/O performance won't likely inject signficant overhead on client side when sending/generating requests.


### Prompt data format

a typical "jsonl" file, with "input" key in each line:

```
{"input_text": "prompt 1..."}
{"input_text": "prompt 2..."}
...
```

the following sanity check uses `./data/openorca_10.jsonl` for testing.

### Command line Usage

### Cross check the perf result

**Result Using this Tool**

command:

```bash
python3 llm_perf.py    \
    -u localhost:18001 -i grpc -m ensemble   \
    --concurrency 1,2,4 \
    --tokenizer-path /workspace/models/Qwen2-7B \
    --input-data ./data/openorca_10.jsonl

```

Output:

```bash
------------------------------------------------------------
Request concurrency: 1
|                          |   Average |    Min |     Max |     P99 |     P90 |     P75 |
|:-------------------------|----------:|-------:|--------:|--------:|--------:|--------:|
| Time to first token (ms) |     68.03 |  15.61 |  485.8  |  447.55 |  103.32 |   18.48 |
| Inter token latency (ms) |     11.81 |  11.27 |   14.16 |   13.98 |   12.31 |   11.95 |
| Request latency (ms)     |   1344.75 | 196.37 | 3393.54 | 3392.83 | 3386.39 | 1975.16 |
num of payload: 10; completed results: 10
request per sec (throughput in system): 0.7426 infer/sec
total input token per sec (in system): 139.5388 token/sec
total output token per sec (in system): 83.8421 token/sec

```


**Result Using Triton's genai-perf**

command:

```bash

export TRITON_URI=127.0.0.1:18001

genai-perf profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file ./data/openorca_10.jsonl \
  --streaming \
  --output-tokens-mean 300 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer /home/lukex/workspace/models/Qwen2-7B \
  --concurrency 1 \
  --measurement-interval 10000 \
  --profile-export-file qwen2-7b_results_test.json \
  --url ${TRITON_URI}
```

output:

```bash

2024-09-20 07:12 [INFO] genai_perf.parser:90 - Profiling these models: ensemble
2024-09-20 07:12 [INFO] genai_perf.wrapper:147 - Running Perf Analyzer : 'perf_analyzer -m ensemble --async --input-data artifacts/ensemble-triton-tensorrtllm-concurrency1/llm_inputs.json -i grpc --streaming --shape max_tokens:1 --shape text_input:1 --concurrency-range 1 --service-kind triton -u 127.0.0.1:18001 --measurement-interval 10000 --stability-percentage 999 --profile-export-file artifacts/ensemble-triton-tensorrtllm-concurrency1/qwen2-7b_results_test.json'
                                         LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to first token (ms) │    52.10 │    14.55 │   335.78 │   311.03 │    88.31 │    17.47 │
│ Inter token latency (ms) │    11.41 │    11.24 │    11.69 │    11.68 │    11.68 │    11.46 │
│     Request latency (ms) │ 3,426.12 │ 3,373.51 │ 3,753.69 │ 3,729.47 │ 3,511.51 │ 3,382.11 │
│   Output sequence length │   296.70 │   289.00 │   300.00 │   300.00 │   300.00 │   299.75 │
│    Input sequence length │   187.90 │    49.00 │   941.00 │   870.98 │   240.80 │   155.50 │
└──────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 86.59
Request throughput (per sec): 0.29

2024-09-20 07:12 [INFO] genai_perf.export_data.json_exporter:58 - Generating artifacts/ensemble-triton-tensorrtllm-concurrency1/qwen2-7b_results_test_genai_perf.json
2024-09-20 07:12 [INFO] genai_perf.export_data.csv_exporter:69 - Generating artifacts/ensemble-triton-tensorrtllm-concurrency1/qwen2-7b_results_test_genai_perf.csv
```


**Difference**

* `Inter token latency` and `output token per sec `: two approaches have similiar results
* `Request latency` and `Request throughput`: big difference between two approach, mainly because genai-perf force a fixed num of output tokens, while `llm_perf.py` stop earlier.
* `Time to first token`: some difference between the two approaches.
