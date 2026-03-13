import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Set a random seed for reproducibility
rseed = 0
np.random.seed(rseed)

def compute_target_clusters(
    shap_values: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int = 0
):
    """
    Finds optimal k via silhouette, then returns (best_k, labels, ks, silhouette_scores, inertias).
    """
    silhouette_scores = []
    inertias = []
    ks = list(range(min_k, max_k + 1))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(shap_values)
        inertias.append(km.inertia_)
        if k > 1:
            score = silhouette_score(shap_values, labels)
            silhouette_scores.append(score)
            print(f"  k={k}: silhouette={score:.3f}, inertia={km.inertia_:.1f}")
        else:
            silhouette_scores.append(np.nan)
            print(f"  k={k}: inertia={km.inertia_:.1f}")

    best_k = ks[int(np.nanargmax(silhouette_scores))]
    print(f"→ Chosen k = {best_k}\n")
    km_best = KMeans(n_clusters=best_k, random_state=random_state)
    return best_k, km_best.fit_predict(shap_values), ks, silhouette_scores, inertias


def cluster_shap(
    shap_dict: dict,
    original_features: np.ndarray,
    out_dir: str = "Clusters",
    plot_dir: str = "Cluster_Plots",
    min_k: int = 2,
    max_k: int = 10,
    seed: int = rseed
) -> dict:
    """
    For each target in shap_dict:
      - find optimal k via silhouette
      - run KMeans(labels)
      - save `shap_clusters_<target>.csv`
      - save per-cluster pure & augmented .npy in out_dir
      - save PCA scatter, silhouette, and elbow plots in plot_dir

    Returns:
      mapping of { target: [cluster_labels] }
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    target_to_labels = {}
    targets = list(shap_dict.keys())

    for target in targets:
        print(f"\n=== Clustering SHAP for target '{target}' ===")
        shap_vals = shap_dict[target]
        print(f"SHAP array shape: {shap_vals.shape}")

        # 1) find best k and labels
        best_k, labels, ks, sil_scores, inertias = compute_target_clusters(
            shap_vals, min_k, max_k, random_state=seed
        )
        target_to_labels[target] = labels

        # 2) save labels to CSV
        df_lbl = pd.DataFrame({"cluster": labels})
        csv_path = os.path.join(out_dir, f"shap_clusters_{target}.csv")
        df_lbl.to_csv(csv_path, index=False)
        print(f"→ Saved cluster labels: {csv_path}")

        # 3a) Silhouette plot
        plt.figure(figsize=(7, 4))
        plt.plot(ks, sil_scores, marker="o")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette Scores – {target}")
        sil_path = os.path.join(plot_dir, f"Silhouette_{target}.png")
        plt.savefig(sil_path, bbox_inches="tight")
        plt.close()
        print(f"→ Saved Silhouette plot: {sil_path}")

        # 3b) Elbow plot
        plt.figure(figsize=(7, 4))
        plt.plot(ks, inertias, marker="o", color="coral")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia")
        plt.title(f"Elbow Plot – {target}")
        elbow_path = os.path.join(plot_dir, f"Elbow_{target}.png")
        plt.savefig(elbow_path, bbox_inches="tight")
        plt.close()
        print(f"→ Saved Elbow plot: {elbow_path}")

        # 3c) PCA plot
        pca = PCA(n_components=2, random_state=seed)
        pcs = pca.fit_transform(shap_vals)
        df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        df_pca["cluster"] = labels

        #tag source model
        df_pca["target"] = target

        #save PCA data to csv for unified plotting

        pca_csv_path = os.path.join(out_dir,f"pca_data_{target}.csv")
        df_pca.to_csv(pca_csv_path, index=False)
        print(f"→ Saved PCA data: {pca_csv_path}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_pca,
            x="PC1", y="PC2",
            hue="cluster", palette="viridis",
            legend="full", s=50
        )
        plt.title(f"PCA of SHAP – {target}")
        plot_path = os.path.join(plot_dir, f"PCA_SHAP_{target}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"→ Saved PCA plot: {plot_path}")

        # 4) per-cluster pure & augmented
        for cl in range(best_k):
            idx = np.where(labels == cl)[0]
            cluster_data = original_features[idx, :]
            centroid = cluster_data.mean(axis=0, keepdims=True)

            # pure
            pure_path = os.path.join(out_dir, f"{target}_cluster_{cl}_pure.npy")
            np.save(pure_path, cluster_data)

            # augmented
            aug = np.vstack([cluster_data, centroid])
            aug_path = os.path.join(out_dir, f"{target}_cluster_{cl}_augmented.npy")
            np.save(aug_path, aug)

            print(
                f"→ Cluster {cl}: {len(idx)} pts, "
                f"saved pure→'{pure_path}' and aug→'{aug_path}'"
            )

    print("\nFinished clustering all targets.")
    return target_to_labels
