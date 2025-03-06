#!/bin/bash

DATA_PATH='data/v2/test'

# Use DeepSeek-Coder-V2-Lite-Instruct model
MODEL_DIR='deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'

# Run this before executing your script
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create output directory if it doesn't exist
mkdir -p outputs/deepseek

python inference/infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'tablebench' \
    --temperature 0 \
    --sample_n 1 \
    --model_max_length 8192 \
    --outdir "outputs/deepseek"


