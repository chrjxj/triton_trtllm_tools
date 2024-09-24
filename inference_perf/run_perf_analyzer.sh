
export TRITON_URI=127.0.0.1:8001

# HTTP
# perf_analyzer -m ensemble --async --input-data llm_inputs.json \
#     --measurement-interval 600000  \
#     --profile-export-file qwen2-7b-1024_1.json  \
#     --service-kind openai  \
#     -v --endpoint v1/chat/completions  \
#     -u ${TRITON_URI}  -i http \
#     --concurrency-range 100  \
#     --stability-percentage 999  \
#     --streaming --shape max_tokens:100 --shape text_input:1

# streaming/grpc
perf_analyzer -m ensemble --async --input-data ./data/openorca_10_perfanalyzer.json \
    --measurement-interval 10000  \
    --profile-export-file perf_analyzer_results.json  \
    --service-kind triton  \
    -v --endpoint v1/chat/completions  \
    -u ${TRITON_URI} -i grpc \
    --concurrency-range 1:5:1  \
    --stability-percentage 999  \
    --streaming --shape max_tokens:1 --shape text_input:1
