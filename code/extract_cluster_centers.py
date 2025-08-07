import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Load clustered data and original embeddings
df_cluster = pd.read_csv("sequence_clusters.csv")
df_embed = pd.read_csv("esm2_embeddings.csv")

# Merge both on ID
df = df_cluster.merge(df_embed, on="id")

# Show overall NaN count
total_nan = df.isna().sum().sum()
print(f"🔍 Total NaN values before processing: {total_nan}")

# Select valid clusters (excluding noise = -1)
clusters = sorted([c for c in df["cluster"].unique() if c != -1])
print(f"🔎 Found {len(clusters)} valid clusters")

center_seqs = []

# Process each cluster
for c in clusters:
    group = df[df["cluster"] == c]
    feature_cols = [col for col in df.columns if col.startswith("f")]
    
    # Extract features and clean per-cluster
    features = group[feature_cols]
    nan_count = features.isna().sum().sum()
    
    if nan_count > 0:
        print(f"⚠️ Cluster {c}: {nan_count} NaNs found, replacing with column means...")
        features = features.fillna(features.mean())

    features_np = features.values
    mean_vector = np.mean(features_np, axis=0).reshape(1, -1)

    # Skip if mean still has NaN
    if np.isnan(mean_vector).any() or np.isnan(features_np).any():
        print(f"❌ Cluster {c}: still has NaNs after cleaning, skipping...")
        continue

    # Compute cosine distances
    dists = cosine_distances(features_np, mean_vector)
    min_idx = np.argmin(dists)
    center_seq_id = group.iloc[min_idx]["id"]
    center_seqs.append((c, center_seq_id))

# Save final results
df_centers = pd.DataFrame(center_seqs, columns=["cluster", "id"])
df_centers.to_csv("cluster_centers.csv", index=False)
print("✅ Cluster center IDs saved to cluster_centers.csv")
