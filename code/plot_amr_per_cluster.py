import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("amr_trace_results.csv")
df_cluster = pd.read_csv("sequence_clusters.csv")
df = df.merge(df_cluster, on="id", how="left")

df = df[df["cluster"] != -1]  # exclude noise

# Top 10 clusters by size
top_clusters = df["cluster"].value_counts().nlargest(10).index
df_top = df[df["cluster"].isin(top_clusters)]

plt.figure(figsize=(14, 6))
sns.countplot(
    data=df_top,
    y="matched_amr_class",
    hue="cluster",
    order=df_top["matched_amr_class"].value_counts().iloc[:15].index
)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Top AMR Classes in Top 10 Clusters")
plt.xlabel("Count")
plt.ylabel("AMR Class")
plt.tight_layout()
plt.savefig("barplot_amr_per_cluster.png", dpi=300)
plt.show()
