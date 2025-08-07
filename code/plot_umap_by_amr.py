import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load data
df_umap = pd.read_csv("sequence_clusters.csv")
df_amr = pd.read_csv("esm2_with_amr_prediction.csv")
df = df_umap.merge(df_amr[["id", "amr_prediction"]], on="id", how="left")
df["amr_prediction"] = df["amr_prediction"].fillna("unknown")

# Count unique AMR labels
unique_labels = sorted(df["amr_prediction"].unique())
n_labels = len(unique_labels)

# Dynamic color palette
palette = sns.color_palette("hsv", n_labels)

# Build label-to-color map
label2color = dict(zip(unique_labels, palette))
df["color"] = df["amr_prediction"].map(label2color)

# Create plot
plt.figure(figsize=(14, 10))
for label in unique_labels:
    subset = df[df["amr_prediction"] == label]
    plt.scatter(subset["umap_x"], subset["umap_y"], label=label, s=10, color=label2color[label], alpha=0.8)

# Dynamic legend column count
ncols = math.ceil(n_labels / 30)

plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.01, 1),
    title="AMR Class",
    fontsize="small",
    title_fontsize="medium",
    ncol=ncols,
    borderaxespad=0.2,
    frameon=False
)

plt.title("UMAP Clusters Colored by Predicted AMR Class")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("umap_clusters_amr_full_legend.png", dpi=300, bbox_inches='tight')
plt.show()
