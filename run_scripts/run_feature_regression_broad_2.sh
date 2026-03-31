#!/bin/bash

# Feature regression (1c) on broad_2 clusters
# Run from project root: ./run_scripts/run_feature_regression_broad_2.sh

export PYTHONPATH=.

PREPARED_SAMPLES_DIR_NW="YOUR_PREPARED_SAMPLES_DIR_NO_WHITESPACE"
PREPARED_SAMPLES_DIR="YOUR_PREPARED_SAMPLES_DIR"
OUTPUT_DIR_NW="YOUR_OUTPUT_DIR_NO_WHITESPACE"
OUTPUT_DIR="YOUR_OUTPUT_DIR"
NEW_CLUSTERS="512_181,768_140,512_67,768_596"

echo "Running feature regression on broad_2 clusters (no whitespace)..."
python -u validation/feature_regression.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR_NW}" \
    --output_dir "${OUTPUT_DIR_NW}" \
    --clusters "${NEW_CLUSTERS}"

echo ""
echo "Running feature regression on broad_2 clusters (with whitespace)..."
python -u validation/feature_regression.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${NEW_CLUSTERS}"

echo ""
echo "Done. Results in ${OUTPUT_DIR} and ${OUTPUT_DIR_NW}/"
