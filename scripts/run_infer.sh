# Define the path to test data
DATA_PATH='data/v2/test'

# Define the model
MODEL_DIR='google/flan-t5-base'

# Run inference
python inference/infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --temperature 0 \
    --sample_n 1 \
    --batch_size 8 \
    --device "cuda" \
    --max_output_length 256 \
    --measure_flops \
    --outdir "outputs/flan-t5-base"


