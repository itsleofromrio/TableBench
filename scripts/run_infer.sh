DATA_PATH='data/v2/test'

# Change to FLAN-T5-base
MODEL_DIR='google/flan-t5-base'

# Run this before executing your script
export PYTHONPATH=$PYTHONPATH:$(pwd)

python inference/infer.py \
    --data_path $DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'tablebench' \
    --temperature 0 \
    --sample_n 1 \
    --outdir "outputs/flan-t5"


