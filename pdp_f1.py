import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import glob

target_units = {
    "BS": "MPa",
    "moe": "MPa",
    "EC": "CO₂eq",
}
regression_targets = ["BS", "moe", "EC"]

def generate_pdp_for_all_targets(targets, pipeline, X_train, feature_names,
                                 cluster_dir, optimal_indices_dirs, pdp_out_dir):
    os.makedirs(pdp_out_dir, exist_ok=True)

    # Ensure it's a list
    if isinstance(optimal_indices_dirs, str):
        optimal_indices_dirs = [optimal_indices_dirs]

    for target in targets:
        print(f"\nGenerating PDPs for target: {target}")
        target_index = regression_targets.index(target)

        cluster_files = sorted([
            f for f in os.listdir(cluster_dir)
            if f.startswith(f"{target}_cluster_") and (f.endswith("_pure.npy") or f.endswith("_augmented.npy"))
        ])

        for fname in cluster_files:
            parts = fname.split("_")
            cluster_label = parts[2]
            mode = "pure" if fname.endswith("_pure.npy") else "augmented"

            # Flexible lookup for the index file
            optimal_path = None
            for folder in optimal_indices_dirs:
                pattern = os.path.join(folder, f"optimal_feature_indices_{target}_cluster_{cluster_label}_{mode}.npy")
                print(f"Looking in: {pattern}")
                matches = glob.glob(pattern)
                if matches:
                    optimal_path = matches[0]
                    break

            if optimal_path is None:
                print(f"  Missing optimal index file for {target} cluster {cluster_label}. Skipping.")
                continue

            optimal_feature_indices = np.load(optimal_path)
            if optimal_feature_indices.size == 0:
                print(f"  Empty index set for {target} cluster {cluster_label} ({mode}). Skipping.")
                continue

            print(f"  Cluster {cluster_label} ({mode}) → features {optimal_feature_indices}")

            fig, ax = plt.subplots(figsize=(12, 8))
            pd_style = dict(color="red", linewidth=3.5, label="PDP (mean)")
            ice_style = dict(color="tab:blue", alpha=0.10, linewidth=0.8)

            try:
                cluster_X = np.load(os.path.join(cluster_dir, fname)) # shape = (n_samples, n_features)
                print(f"{fname} → shape {cluster_X.shape}")
                pd_display = PartialDependenceDisplay.from_estimator(
                    pipeline,
                    cluster_X,
                    features=optimal_feature_indices,
                    target=target_index,
                    grid_resolution=100,
                    kind="both",
                    feature_names=feature_names,
                    ax=ax,
                    pd_line_kw=pd_style,
                    ice_lines_kw=ice_style,
                )

                # flatten the axes array it created:
                axes = np.array(pd_display.axes_).ravel()

                # annotate each axis with sample count & feature range
                for feature_pos, feature_idx in enumerate(optimal_feature_indices):
                    subax = axes[feature_pos]
                    col = cluster_X[:, feature_idx]
                    n, lo, hi = len(col), col.min(), col.max()
                    subax.text(
                        0.98, 0.02,
                        f"n={n}, range={lo:.0f}–{hi:.0f}",
                        transform=subax.transAxes,
                        ha="right", va="bottom",
                        fontsize=8, color="gray"
                    )

                unit = target_units.get(target, "")
                ylabel = f"Partial dependence of {target} ({unit})" if unit else f"Partial dependence of {target}"
                for subax in fig.axes:
                    subax.set_ylabel(ylabel)

                fig.suptitle(
                    f"PDP for {target} – Cluster {cluster_label} ({mode})\n"
                    f"indices: {optimal_feature_indices}",
                    fontsize=14
                )
                fig.tight_layout()

                out_fname = f"PDP_{target}_cluster_{cluster_label}_{mode}.png"
                fig.savefig(os.path.join(pdp_out_dir, out_fname), dpi=150, bbox_inches="tight")
                plt.close(fig)

            except Exception as e:
                print(f"  PDP failed for {target} cluster {cluster_label} ({mode}): {e}")

        print(f"Completed PDPs for all clusters of target: {target}")
