#!/bin/bash

DATA_PATH='data/v2/test'

# DeepSeek-Coder-V2-Lite-Base model
MODEL_DIR='deepseek-ai/deepseek-coder-v2-lite-base'

python inference/infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'tablebench'  \
    --temperature 0 \
    --sample_n 1 



