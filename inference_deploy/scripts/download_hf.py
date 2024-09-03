#! /usr/bin/env python3
"""
Usage:

    export HF_TOKEN=<YOUR-Huggingface-Token>
    python3 download_hf.py meta-llama/Llama-2-13b-chat-hf  /models/llama-2-13b-chat-hf 
    python3 download_hf.py Qwen/Qwen2-7B  /models/Qwen2-7B 

"""

import os
import sys
from huggingface_hub import snapshot_download


def main():
    token = os.environ.get('HF_TOKEN', "")
    if not token:
        raise RuntimeError("Please set HF_TOKEN Environment Variable to your huggingface token")

    snapshot_download(
        repo_id=sys.argv[1],
        local_dir=sys.argv[2],
        local_dir_use_symlinks=False,
        token=token,
    )


if __name__ == '__main__':
    main()
