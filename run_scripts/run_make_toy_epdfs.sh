#!/usr/bin/env bash
# Generate per-cluster EPDF figures for multipartite_003e (layer 1, TopK k=12).
# Uses imshow+pixel-grid KDE (triangle for Mess3, disk for Tom Quantum).
# Run from project root: bash run_scripts/run_make_toy_epdfs.sh

set -euo pipefail
export PYTHONPATH=.
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu

CLUSTER_SUMMARY="YOUR_CLUSTER_SUMMARY"
MODEL_CKPT="YOUR_MODEL_CKPT"
SAE_PATH="YOUR_SAE_PATH"
OUTPUT_DIR="YOUR_OUTPUT_DIR"

python -u make_toy_epdfs.py \
    --cluster_summary "${CLUSTER_SUMMARY}" \
    --model_ckpt     "${MODEL_CKPT}" \
    --sae_path       "${SAE_PATH}" \
    --output_dir     "${OUTPUT_DIR}" \
    --n_sequences    10000 \
    --batch_size     512 \
    --grid_size      80 \
    --base_opacity   0.55 \
    --dpi            150 \
    --ext            png

echo ""
echo "Done. Figures in ${OUTPUT_DIR}/"
echo "  cluster_*_all.png        — all-latents overlay per cluster"
echo "  cluster_*/latent_*.png   — solo per-latent figures"
