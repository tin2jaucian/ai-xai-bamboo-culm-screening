#!/usr/bin/env python3
"""
TERP Full Workflow Script
This script runs the TERP medoid and centroid workflows for all clusters and targets,
and computes the canonical local fidelity (|ρ|) between black-box predictions and the linear surrogate.

# Adapted in part from TERP code by the Tiwary Research Group
# Original source: https://github.com/tiwarylab/TERP/tree/main
# Licensed under the MIT License
If you use the TERP method, please cite:
Mehdi, S., & Tiwary, P. (2024). *Thermodynamics-inspired explanations of artificial intelligence*. Nature Communications, 15(1), 7859.
"""


import os
import argparse
import subprocess
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Paths to TERP utilities
NEIGHBORHOOD_SCRIPT = os.path.join("TERP_pys", "TERP_neighborhood_generator.py")
OPTIMIZER_STEP1     = os.path.join("TERP_pys", "updated_TERP_optimizer_01_pca.py")
OPTIMIZER_STEP2     = os.path.join("TERP_pys", "updated_TERP_optimizer_02_pca2.py")


def process_cluster(
    pipeline,
    file_path: str,
    target_name: str,
    cluster_label: str,
    output_dir: str,
    mode: str,
    seed: int,
    num_samples: int,
    cutoff: int,
    feature_csv: str,
    targets_list: list
) -> (dict, pd.DataFrame, str):
    """
    Run TERP neighborhood + optimizers + weight plotting for one cluster file,
    then compute local fidelity |ρ| with a linear surrogate.
    Returns (summary_dict, scan_log_df, idx_path) or None if failed.
    """
    data = np.load(file_path)
    n_rows = data.shape[0]

    # determine index of interest (instance to be explained)
    if mode == 'pure':
        centroid = data.mean(axis=0, keepdims=True)
        distances = cdist(data, centroid, metric='euclidean').flatten()
        idx = int(np.argmin(distances))
    else:
        idx = n_rows - 1

    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    # 1) Generate synthetic neighborhood
    subprocess.run([
        'python', NEIGHBORHOOD_SCRIPT,
        '-seed', str(seed),
        '--progress_bar',
        '-input_numeric', file_path,
        '-num_samples', str(num_samples),
        '-index', str(idx)
    ], check=True)

    # 2) Predict on neighborhood
    neigh_file = os.path.join('DATA', 'make_prediction_numeric.npy')
    perturbed   = np.load(neigh_file)
    preds_all   = pipeline.predict(perturbed)
    y_true      = preds_all[:, targets_list.index(target_name)]
    step1_path = os.path.join(
        mode_dir,
        f"pred_{target_name}_cluster_{cluster_label}_{mode}_step1.npy"
    )
    np.save(step1_path, y_true)

    # 3) Initial TERP optimizer
    subprocess.run([
        'python', OPTIMIZER_STEP1,
        '-TERP_input', os.path.join('DATA', 'TERP_numeric.npy'),
        '-blackbox_prediction', step1_path,
        '-cutoff', str(cutoff)
    ], check=True)

    # 4) Regenerate neighborhood with selected features
    subprocess.run([
        'python', NEIGHBORHOOD_SCRIPT,
        '-seed', str(seed),
        '--progress_bar',
        '-input_numeric', file_path,
        '-num_samples', str(num_samples),
        '-index', str(idx),
        '-selected_features', os.path.join('TERP_results', 'selected_features.npy')
    ], check=True)

    # 5) Predict on regenerated neighborhood
    neigh2_file = os.path.join('DATA_2', 'make_prediction_numeric.npy')
    perturbed2  = np.load(neigh2_file)
    preds2_all  = pipeline.predict(perturbed2)
    y_step2     = preds2_all[:, targets_list.index(target_name)]
    step2_path = os.path.join(
        mode_dir,
        f"pred_{target_name}_cluster_{cluster_label}_{mode}_step2.npy"
    )
    np.save(step2_path, y_step2)

    # 6) Final TERP optimizer
    subprocess.run([
        'python', OPTIMIZER_STEP2,
        '-TERP_input', os.path.join('DATA_2', 'TERP_numeric.npy'),
        '-blackbox_prediction', step2_path,
        '-selected_features', os.path.join('TERP_results', 'selected_features.npy')
    ], check=True)

    # 7) Load optimal feature weights
    weight_file = os.path.join('TERP_results_2', 'optimal_feature_weights.npy')
    if not os.path.exists(weight_file):
        return None
    w = np.load(weight_file)
        # 7.1) Save copy of weights for specific cluster
    np.save(
        os.path.join(mode_dir,
                     f"weights_{target_name}_cluster_{cluster_label}_{mode}.npy"),
        w
    )
    sel_idxs = np.where(np.abs(w) > 1e-5)[0]
    if sel_idxs.size == 0:
        return None

    # 8) Compute local fidelity
    y_bb       = pipeline.predict(perturbed2)[:, targets_list.index(target_name)]
    X_sur      = perturbed2[:, sel_idxs]
    lr         = LinearRegression().fit(X_sur, y_bb)
    y_hat      = lr.predict(X_sur)

    rho, _     = pearsonr(y_bb, y_hat)
    fidelity   = abs(rho)
    U          = 1 - fidelity
    rmse       = np.sqrt(mean_squared_error(y_bb, y_hat))
    w_sel      = np.abs(w[sel_idxs])
    S          = -np.sum(w_sel * np.log(w_sel + 1e-12))
    intercept  = lr.intercept_

    summary = {
        'target':      target_name,
        'cluster':     cluster_label,
        'mode':        mode,
        'seed':        seed,
        'num_samples': num_samples,
        'n_features':  sel_idxs.size,
        'fidelity':    fidelity,
        'U':           U,
        'entropy_S':   S,
        'rmse':        rmse,
        'intercept':   intercept
    }

    # 9) Read the θ-scan log
    scan_log_path = os.path.join('TERP_results_2', 'terp_scan_log.csv')
    scan_df = pd.read_csv(scan_log_path)
    scan_df['target']  = target_name
    scan_df['cluster'] = cluster_label
    scan_df['mode']    = mode

    # 10) Save optimal feature indices
    idx_path = os.path.join(
        mode_dir,
        f"optimal_feature_indices_{target_name}_cluster_{cluster_label}_{mode}.npy"
    )
    np.save(idx_path, sel_idxs)

    return summary, scan_df, idx_path


def run_all_targets(args) -> dict:
    pipeline = joblib.load(args.pipeline_path)
    all_summaries, all_scans = [], []
    result_map = {}

    for tgt in args.targets:
        result_map[tgt] = {}
        for mode, suffix in [('pure','_pure.npy'), ('augmented','_augmented.npy')]:
            result_map[tgt][mode] = {}
            files = sorted(
                f for f in os.listdir(args.cluster_dir)
                if f.startswith(f"{tgt}_cluster_") and f.endswith(suffix)
            )
            for f in files:
                label = f.split('_')[2]
                path  = os.path.join(args.cluster_dir, f)
                ret = process_cluster(
                    pipeline, path, tgt, label,
                    args.output_dir, mode,
                    args.seed, args.num_samples,
                    args.cutoff, args.feature_csv,
                    args.targets
                )
                if ret is None:
                    continue
                summary, scan_df, idx_path = ret
                all_summaries.append(summary)
                all_scans.append(scan_df)
                result_map[tgt][mode][label] = idx_path

    # Dump CSVs
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(all_summaries).to_csv(
        os.path.join(args.output_dir, 'terp_summary.csv'), index=False)
    pd.concat(all_scans, ignore_index=True).to_csv(
        os.path.join(args.output_dir, 'terp_scan_details.csv'), index=False)

    return result_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Full TERP Medoid & Centroid Workflow'
    )
    parser.add_argument('--targets',      nargs='+', required=True,
                        help='Targets list, e.g. CS SL EC')
    parser.add_argument('--pipeline-path', required=True,
                        help='Path to trained pipeline .pkl')
    parser.add_argument('--cluster-dir',  required=True,
                        help='Directory with *_pure.npy & *_augmented.npy')
    parser.add_argument('--output-dir',   default='TERP_Results',
                        help='Where to write TERP outputs')
    parser.add_argument('--feature-csv',  required=True,
                        help='CSV of training features (for plot labels)')
    parser.add_argument('--seed',         type=int,   default=0,
                        help='Random seed')
    parser.add_argument('--num-samples',  type=int,   default=5000,
                        help='Synthetic sample count')
    parser.add_argument('--cutoff',       type=int,   default=12,
                        help='Initial optimizer cutoff')
    args = parser.parse_args()

    results = run_all_targets(args)
    print("\nDone. Optimal indices saved:\n", results)
