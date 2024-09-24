
export TRITON_URI=127.0.0.1:8001
export TOKENIZER_DIR= /path/to/Qwen2-7B

genai-perf profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file ./data/openorca_10.jsonl \
  --streaming \
  --output-tokens-mean 300 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer ${TOKENIZER_DIR} \
  --concurrency 1 \
  --measurement-interval 10000 \
  --profile-export-file results_test.json \
  --url ${TRITON_URI}