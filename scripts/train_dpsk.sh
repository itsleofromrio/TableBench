# Get the absolute path to the project root
PROJECT_ROOT=$(pwd)

export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

MODEL_PATH='deepseek-ai/deepseek-coder-v2-lite-base'

# Use absolute paths
DATA_PATH="${PROJECT_ROOT}/data/v2/training/data-analysis-tcot-instructions.json"
SAVE_PATH="${PROJECT_ROOT}/ckpt/dpsk-lite-TCoT-DataAnalysis/"

# Create DeepSpeed config file with auto values
cat > ds_config.json << EOF
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": false
  }
}
EOF

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    ${PROJECT_ROOT}/train/train_dpsk.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --trust_remote_code True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --use_cot false \
    --save_steps 10000 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --deepspeed ds_config.json