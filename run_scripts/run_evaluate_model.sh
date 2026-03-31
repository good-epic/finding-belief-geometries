#!/usr/bin/bash

MODEL_CKPT="YOUR_MODEL_CKPT"
OUTPUT_DIR="YOUR_OUTPUT_DIR"

python -u scratch/evaluate_model_accuracy.py \
    --model_ckpt "${MODEL_CKPT}" \
    --process_config "process_configs.json" \
    --process_config_name "3xmess3_2xtquant_002" \
    --output_dir "${OUTPUT_DIR}" \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 3 \
    --n_ctx 16 \
    --d_head 32 \
    --act_fn "relu" \
    --batch_size 256 \
    --n_batches 100 \
    --seq_len 16 \
    --seed 43 \
    --device "cuda"
