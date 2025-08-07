import pandas as pd

# Load cluster + AMR prediction
df_cluster = pd.read_csv("sequence_clusters.csv")
df_trace = pd.read_csv("amr_trace_results.csv")
df = df_cluster.merge(df_trace, on="id", how="left")

# Group by cluster
summary = []

for cluster_id, group in df.groupby("cluster"):
    if cluster_id == -1:  # skip noise
        continue

    class_counts = group["matched_amr_class"].value_counts().head(5)
    for i, (amr_class, count) in enumerate(class_counts.items()):
        summary.append({
            "cluster": cluster_id,
            "rank": i + 1,
            "amr_class": amr_class,
            "count": count
        })

df_summary = pd.DataFrame(summary)
df_summary.to_csv("cluster_amr_summary.csv", index=False)
print("✅ Top 5 AMR classes per cluster saved to cluster_amr_summary.csv")
