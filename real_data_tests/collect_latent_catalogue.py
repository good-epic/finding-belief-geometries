"""collect_latent_catalogue.py

Build a JSON catalogue of every latent in the 13 priority clusters, merging:
  - Latent index and cluster membership
  - Global activity rate (fraction of 5000 simplex samples where the latent fired)
  - Per-vertex mean activation and pie-chart share in near-vertex samples
  - Active-vertex labels from our sonnet_broad_2 synthesis (with confidence + consistency)
  - Neuronpedia gpt-4o-mini auto-interpretation

Output: outputs/latent_catalogue.json

Usage:
    python real_data_tests/collect_latent_catalogue.py [--output outputs/latent_catalogue.json]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRIORITY_CLUSTERS = [
    ("512", "17"),
    ("512", "22"),
    ("512", "67"),
    ("512", "181"),
    ("512", "229"),
    ("512", "261"),
    ("512", "471"),
    ("512", "504"),
    ("768", "140"),
    ("768", "210"),
    ("768", "306"),
    ("768", "581"),
    ("768", "596"),
]

NEURONPEDIA_API = (
    "https://neuronpedia.org/api/feature/gemma-2-9b/20-gemmascope-res-16k/{}"
)
# Pie chart threshold: latent must have mean_act >= 1% of vertex total to appear
PIE_THRESH_FRAC = 0.01
# Delay between Neuronpedia requests (seconds)
REQUEST_DELAY = 0.4


# ---------------------------------------------------------------------------
# Local data loading
# ---------------------------------------------------------------------------

def load_spatial_stats(spatial_stats_dir: Path, n_clusters: str, cluster_id: str):
    """Return dict with latent_indices, n_total_samples, per_latent list."""
    key = f"{n_clusters}_{cluster_id}"
    path = spatial_stats_dir / f"cluster_{key}_spatial_stats.json"
    return json.loads(path.read_text())


def load_synthesis(synthesis_dir: Path, n_clusters: str, cluster_id: str):
    """Return synthesis sub-dict with consolidated_vertex_labels, confidence,
    per_vertex_consistency."""
    key = f"{n_clusters}_{cluster_id}"
    path = synthesis_dir / f"{key}_synthesis.json"
    data = json.loads(path.read_text())
    return data["synthesis"]


def iter_vertex_acts(with_acts_path: Path):
    """Yield records from vertex_samples_with_acts.jsonl."""
    with open(with_acts_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_with_acts_path(selected_dir: Path, n_clusters: str, cluster_id: str):
    """Glob for the vertex_samples_with_acts.jsonl file."""
    pattern = f"n{n_clusters}/cluster_{cluster_id}_k*_category*_vertex_samples_with_acts.jsonl"
    matches = list(selected_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No vertex_samples_with_acts.jsonl found for {n_clusters}_{cluster_id} "
            f"in {selected_dir}"
        )
    return matches[0]


def compute_per_vertex_mean_acts(with_acts_path: Path, k: int, n_latents: int):
    """
    Returns:
        mean_acts: np.ndarray shape (k, n_latents) — mean activation per vertex per latent
        counts:    np.ndarray shape (k,) — number of near-vertex samples per vertex
    """
    sum_acts = np.zeros((k, n_latents), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for rec in iter_vertex_acts(with_acts_path):
        v = rec.get("vertex_id")
        if v is None or v >= k:
            continue
        la = rec.get("latent_acts")
        if la is None:
            continue
        # latent_acts is list[list[float]] (one per trigger); average over triggers
        arr = np.array(la, dtype=np.float64)  # (n_triggers, n_latents)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[-1] != n_latents:
            continue
        sum_acts[v] += arr.mean(axis=0)
        counts[v] += 1
    mean_acts = np.where(
        counts[:, None] > 0,
        sum_acts / np.maximum(counts[:, None], 1),
        0.0,
    )
    return mean_acts, counts


# ---------------------------------------------------------------------------
# Neuronpedia fetching
# ---------------------------------------------------------------------------

def fetch_neuronpedia(latent_idx: int, session: requests.Session):
    """Return gpt-4o-mini description string, or None on failure."""
    url = NEURONPEDIA_API.format(latent_idx)
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for exp in data.get("explanations", []):
            if exp.get("explanationModelName") == "gpt-4o-mini":
                return exp.get("description")
        # Fall back to first available explanation if no gpt-4o-mini entry
        exps = data.get("explanations", [])
        if exps:
            return exps[0].get("description")
        return None
    except Exception as exc:
        print(f"  WARNING: failed to fetch latent {latent_idx}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_catalogue(
    spatial_stats_dir: Path,
    synthesis_dir: Path,
    selected_dir: Path,
) -> dict:
    catalogue = {}

    session = requests.Session()
    session.headers["User-Agent"] = "latent-catalogue-builder/1.0"

    # Collect all unique latent indices first for progress reporting
    total_latents = 0
    for n_clusters, cluster_id in PRIORITY_CLUSTERS:
        stats = load_spatial_stats(spatial_stats_dir, n_clusters, cluster_id)
        total_latents += len(stats["latent_indices"])
    print(f"Total latents to process: {total_latents}")

    latent_count = 0
    for n_clusters, cluster_id in PRIORITY_CLUSTERS:
        key = f"{n_clusters}_{cluster_id}"
        print(f"\n{'='*50}")
        print(f"Cluster {key}")

        # --- spatial stats ---
        stats = load_spatial_stats(spatial_stats_dir, n_clusters, cluster_id)
        latent_indices = stats["latent_indices"]
        n_latents = len(latent_indices)
        k = stats["k"]
        n_total = stats["n_total_samples"]

        # global activity rate per latent (from 5000 simplex samples)
        global_act_rates = {}
        for entry in stats["per_latent"]:
            idx = entry["latent_idx"]
            global_act_rates[idx] = entry["n_active"] / n_total if n_total > 0 else 0.0

        # --- synthesis ---
        syn = load_synthesis(synthesis_dir, n_clusters, cluster_id)
        vertex_labels = syn["consolidated_vertex_labels"]  # list[str], length k
        confidence = syn["confidence"]
        per_vertex_consistency = syn["per_vertex_consistency"]  # list[str], length k

        # --- per-vertex mean activations (pie chart data) ---
        with_acts_path = find_with_acts_path(selected_dir, n_clusters, cluster_id)
        mean_acts, vertex_counts = compute_per_vertex_mean_acts(
            with_acts_path, k, n_latents
        )
        # mean_acts shape: (k, n_latents)

        # Compute vertex totals for pie share calculation
        vertex_totals = mean_acts.sum(axis=1)  # (k,)

        # --- assemble latent entries ---
        latent_entries = []
        for li, latent_idx in enumerate(latent_indices):
            latent_count += 1
            print(
                f"  [{latent_count}/{total_latents}] Latent {latent_idx} ...",
                end=" ",
                flush=True,
            )

            # Per-vertex data — only include vertices where the latent appears
            # in the pie chart (mean_act >= 1% of that vertex's total)
            active_vertices = {}
            for v in range(k):
                act = float(mean_acts[v, li])
                total_v = float(vertex_totals[v])
                if total_v > 0 and act >= PIE_THRESH_FRAC * total_v:
                    pie_share = act / total_v
                    active_vertices[str(v)] = {
                        "mean_act": round(act, 6),
                        "pie_share": round(pie_share, 4),
                        "vertex_label": vertex_labels[v] if v < len(vertex_labels) else None,
                        "vertex_consistency": per_vertex_consistency[v] if v < len(per_vertex_consistency) else None,
                    }

            # Neuronpedia
            interp = fetch_neuronpedia(latent_idx, session)
            print(f"interp={'OK' if interp else 'NONE'}")
            time.sleep(REQUEST_DELAY)

            latent_entries.append({
                "latent_idx": latent_idx,
                "global_activity_rate": round(
                    global_act_rates.get(latent_idx, 0.0), 6
                ),
                "active_vertices": active_vertices,
                "neuronpedia_interp": interp,
            })

        catalogue[key] = {
            "k": k,
            "n_latents": n_latents,
            "synthesis_confidence": confidence,
            "vertex_labels": vertex_labels,
            "per_vertex_consistency": per_vertex_consistency,
            "vertex_sample_counts": [int(c) for c in vertex_counts],
            "latents": latent_entries,
        }

    return catalogue


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/validation/latent_catalogue.json",
        help="Output JSON path (default: outputs/validation/latent_catalogue.json)",
    )
    parser.add_argument(
        "--spatial_stats_dir",
        default="outputs/validation/latent_spatial",
    )
    parser.add_argument(
        "--synthesis_dir",
        default="outputs/interpretations/sonnet_broad_2/synthesis",
    )
    parser.add_argument(
        "--selected_dir",
        default="outputs/selected_clusters_broad_2",
    )
    args = parser.parse_args()

    catalogue = build_catalogue(
        spatial_stats_dir=Path(args.spatial_stats_dir),
        synthesis_dir=Path(args.synthesis_dir),
        selected_dir=Path(args.selected_dir),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalogue, indent=2))
    print(f"\nSaved {len(catalogue)} clusters to {out_path}")

    # Quick summary
    total = sum(len(v["latents"]) for v in catalogue.values())
    with_interp = sum(
        1
        for v in catalogue.values()
        for lat in v["latents"]
        if lat["neuronpedia_interp"]
    )
    print(f"Total latents: {total}, with Neuronpedia interp: {with_interp}/{total}")


if __name__ == "__main__":
    main()
