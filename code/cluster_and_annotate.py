import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load embeddings
# ----------------------------
df = pd.read_csv("esm2_embeddings.csv")
ids = df["id"].values
X = df.drop(columns=["id"])

# ----------------------------
# Step 2: Handle NaN values
# ----------------------------
if X.isnull().values.any():
    print("⚠️ NaN values found in embeddings, replacing with column means...")
    X = X.fillna(X.mean())

X = X.values  # convert to NumPy array

# ----------------------------
# Step 3: Dimensionality Reduction
# ----------------------------
print("Reducing dimensions with UMAP...")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
X_umap = umap_model.fit_transform(X)

# ----------------------------
# Step 4: Clustering
# ----------------------------
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, metric='euclidean')
labels = clusterer.fit_predict(X_umap)

# ----------------------------
# Step 5: Save results
# ----------------------------
df_result = pd.DataFrame({
    "id": ids,
    "cluster": labels,
    "umap_x": X_umap[:, 0],
    "umap_y": X_umap[:, 1]
})
df_result.to_csv("sequence_clusters.csv", index=False)
print(f"✅ Saved clustering results to sequence_clusters.csv")

# ----------------------------
# Step 6: Visualization
# ----------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='Spectral', s=3)
plt.title("Sequence Embedding Clusters (UMAP + HDBSCAN)")
plt.colorbar(scatter, label="Cluster ID")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.savefig("clusters_plot.png", dpi=300)
plt.show()
