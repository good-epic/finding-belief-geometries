#!/bin/bash

# Top predicted tokens by vertex (1b) on broad_2 clusters
# Requires GPU + model. Run on RunPod.
# Run from project root: ./run_scripts/run_top_tokens_broad_2.sh

export PYTHONPATH=.

PREPARED_SAMPLES_DIR="YOUR_PREPARED_SAMPLES_DIR"
OUTPUT_DIR="YOUR_OUTPUT_DIR"
HF_CACHE_DIR="YOUR_HF_CACHE_DIR"
NEW_CLUSTERS="512_181,768_140,512_67,768_596"

echo "Running top tokens by vertex on broad_2 clusters..."
python -u validation/top_tokens_by_vertex.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${NEW_CLUSTERS}" \
    --model_name "gemma-2-9b" \
    --device "cuda" \
    --cache_dir "${HF_CACHE_DIR}" \
    --hf_token ${HF_TOKEN} \
    --top_k 20 \
    --max_samples_per_vertex 200 \
    --batch_size 16

echo ""
echo "Done. Results in ${OUTPUT_DIR}/"
