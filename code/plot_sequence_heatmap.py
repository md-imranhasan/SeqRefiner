import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# --------------------------
# Load Data
# --------------------------
df_umap = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")
df = df_umap.merge(df_pred[["id", "predicted_amr_class"]], on="id", how="left")

# --------------------------
# Get AMR Classes and Sequences
# --------------------------
amr_classes = df["predicted_amr_class"].unique()  # All AMR classes
sequences = df["id"].values  # Sequence IDs

# --------------------------
# Load Sequence Embeddings
# --------------------------
df_embeddings = pd.read_csv("esm2_embeddings.csv")

# Check that all sequence IDs from `df` exist in the `df_embeddings`
missing_ids = [seq for seq in sequences if seq not in df_embeddings["id"].values]
if missing_ids:
    print(f"Warning: The following IDs are missing from embeddings: {missing_ids}")

# Filter embeddings for sequences that exist in the dataframe
df_embeddings = df_embeddings[df_embeddings["id"].isin(sequences)]
sequence_embeddings = df_embeddings.drop(columns=["id"]).values  # Get embeddings only

# --------------------------
# Handle NaN values: Fill with column mean
# --------------------------
sequence_embeddings = np.nan_to_num(sequence_embeddings, nan=np.nanmean(sequence_embeddings))

# --------------------------
# Cluster Sequences using KMeans (reduce the number of sequences for similarity calc)
# --------------------------
n_clusters = 100  # Number of clusters (adjust as needed)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(sequence_embeddings)

# --------------------------
# Compute Cosine Similarity (in batches)
# --------------------------
# Placeholder: Load or compute actual embeddings for AMR classes (use CARD/ResFinder representative genes)
amr_class_embeddings = np.random.rand(len(amr_classes), sequence_embeddings.shape[1])  # Replace with real data

# Initialize an empty similarity matrix
similarity_matrix = np.zeros((n_clusters, len(amr_classes)))

# Calculate similarity within each cluster
for cluster_id in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_embeddings = sequence_embeddings[cluster_indices]
    similarity_matrix[cluster_id, :] = cosine_similarity(cluster_embeddings, amr_class_embeddings).mean(axis=0)

# --------------------------
# Plot the Heatmap
# --------------------------
plt.figure(figsize=(16, 12))  # Adjusted figure size for better space

# Plot heatmap for AMR class similarity within clusters
sns.set(font_scale=0.8)
ax = sns.heatmap(similarity_matrix, cmap="YlGnBu", annot=False, fmt=".2f", 
                 linewidths=0.5, cbar_kws={'label': 'Cosine Similarity'}, 
                 xticklabels=amr_classes, yticklabels=[f"Cluster {i}" for i in range(n_clusters)])

# Rotate tick labels to avoid overlap
plt.xticks(rotation=90, ha="right")
plt.yticks(rotation=0)  # Keep the cluster labels readable

# Add a title
ax.set_title("Cluster-to-AMR Class Similarity Heatmap", fontsize=16)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig("cluster_to_amr_similarity_heatmap_fixed.png", dpi=300)
plt.show()
