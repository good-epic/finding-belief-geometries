#!/bin/bash

# Null control: collect simplex interior samples and run barycentric coordinate
# vs. single-latent predictive power on null AANet clusters.
#
# Step 1 (GPU): collect simplex interior samples for null clusters.
# Step 2 (CPU): prepare vertex sample JSON files from the null cluster manifest.
# Step 3 (GPU): run latent_vs_barycentric.py on NV + SI samples.
#
# Priority null clusters: 512_138, 512_345, 768_310
#
# Run from project root: ./run_scripts/run_null_latent_vs_barycentric.sh
#
# Prerequisites:
#   - selected_null_clusters/ must exist with vertex_samples.jsonl files
#   - null_clusters/ must contain AANet .pt models and consolidated_metrics CSVs

set -e

export PYTHONPATH=.

NULL_CLUSTERS="512_138,512_345,768_310"
CLUSTERS_512="512:138,345"
CLUSTERS_768="768:310"
MANUAL_K_512="512:138=3,345=3"
MANUAL_K_768="768:310=3"

NULL_DIR="YOUR_NULL_DIR"
MANIFEST="YOUR_MANIFEST"
PREPARED_SAMPLES_DIR="YOUR_PREPARED_SAMPLES_DIR"
SIMPLEX_DIR="YOUR_SIMPLEX_DIR"
SOURCE_DIR="YOUR_SOURCE_DIR"
CSV_DIR="YOUR_CSV_DIR"
OUTPUT_DIR="YOUR_OUTPUT_DIR"
HF_CACHE_DIR="YOUR_HF_CACHE_DIR"

echo "============================================================"
echo "NULL CONTROL: SIMPLEX SAMPLES + BARYCENTRIC REGRESSION"
echo "============================================================"
echo "Null clusters:      ${NULL_CLUSTERS}"
echo "Simplex samples:    ${SIMPLEX_DIR}"
echo "Prepared samples:   ${PREPARED_SAMPLES_DIR}"
echo "AANet/CSV dir:      ${CSV_DIR}"
echo "Output dir:         ${OUTPUT_DIR}"
echo ""

# Step 1a: collect simplex interior samples for 512 null clusters
# echo "--- Step 1a: collecting simplex samples (512 null clusters) ---"
# python -u real_data_tests/collect_simplex_samples.py \
#     --n_clusters_list "512" \
#     --source_dir "${SOURCE_DIR}" \
#     --csv_dir "${NULL_DIR}" \
#     --save_dir "${SIMPLEX_DIR}" \
#     --manual_cluster_ids "${CLUSTERS_512}" \
#     --manual_k "${MANUAL_K_512}" \
#     --n_random_controls 0 \            #rng = np.random.default_rng(seed=42)

#     --n_simplex_samples 5000 \
#     --n_position_buckets 10 \
#     --skip_docs 2000000 \
#     --search_batch_size 32 \
#     --max_inputs_per_cluster 2000000 \
#     --model_name "gemma-2-9b" \
#     --sae_release "gemma-scope-9b-pt-res" \
#     --sae_id "layer_20/width_16k/average_l0_68" \
#     --device "cuda" \
#     --cache_dir "${HF_CACHE_DIR}" \
#     --hf_token "${HF_TOKEN}" \
#     --hf_dataset "HuggingFaceFW/fineweb" \
#     --hf_subset_name "sample-10BT" \
#     --dataset_split "train" \
#     --activity_seq_len 256 \
#     --seed 44 \
#     --aanet_prefetch_size 1024 \
#     --shuffle_buffer_size 50000 \
#     --max_doc_tokens 3000 \
#     --aanet_layer_widths 64 32 \
#     --aanet_simplex_scale 1.0 \
#     --aanet_noise 0.05

# # Step 1b: collect simplex interior samples for 768 null clusters
# echo ""
# echo "--- Step 1b: collecting simplex samples (768 null clusters) ---"
# python -u real_data_tests/collect_simplex_samples.py \
#     --n_clusters_list "768" \
#     --source_dir "${SOURCE_DIR}" \
#     --csv_dir "${NULL_DIR}" \
#     --save_dir "${SIMPLEX_DIR}" \
#     --manual_cluster_ids "${CLUSTERS_768}" \
#     --manual_k "${MANUAL_K_768}" \
#     --n_random_controls 0 \
#     --n_simplex_samples 5000 \
#     --n_position_buckets 10 \
#     --skip_docs 2500000 \
#     --search_batch_size 32 \
#     --max_inputs_per_cluster 2000000 \
#     --model_name "gemma-2-9b" \
#     --sae_release "gemma-scope-9b-pt-res" \
#     --sae_id "layer_20/width_16k/average_l0_68" \
#     --device "cuda" \
#     --cache_dir "${HF_CACHE_DIR}" \
#     --hf_token "${HF_TOKEN}" \
#     --hf_dataset "HuggingFaceFW/fineweb" \
#     --hf_subset_name "sample-10BT" \
#     --dataset_split "train" \
#     --activity_seq_len 256 \
#     --seed 45 \
#     --aanet_prefetch_size 1024 \
#     --shuffle_buffer_size 50000 \
#     --max_doc_tokens 3000 \
#     --aanet_layer_widths 64 32 \
#     --aanet_simplex_scale 1.0 \
#     --aanet_noise 0.05

# # Step 2: prepare vertex sample JSON files (CPU-only)
# echo ""
# echo "--- Step 2: preparing vertex samples ---"
# python -u interpretation/prepare_vertex_samples.py \
#     --manifest "${MANIFEST}" \
#     --output_dir "${PREPARED_SAMPLES_DIR}" \
#     --max_samples_per_vertex 200 \
#     --min_samples_per_vertex 1

# Step 3: latent vs. barycentric regression (NV + SI)
echo ""
echo "--- Step 3: latent vs. barycentric regression ---"
python -u validation/latent_vs_barycentric.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}" \
    --simplex_dir "${SIMPLEX_DIR}" \
    --source_dir "${SOURCE_DIR}" \
    --csv_dir "${CSV_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${NULL_CLUSTERS}" \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "${HF_CACHE_DIR}" \
    --hf_token "${HF_TOKEN}" \
    --batch_size 8 \
    --n_top_tokens 50 \
    --cv_folds 5 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05 \
    --max_samples_per_vertex 200 \
    --max_simplex_samples 500 \
    --skip_vertex

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "============================================================"
