import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load UMAP + Novelty data
# ----------------------------
df_umap = pd.read_csv("sequence_clusters.csv")
df_novel = pd.read_csv("novelty_scores.csv")

# Merge on 'id'
df = df_umap.merge(df_novel, on="id", how="left")

# Fill missing novelty
df["novelty_label"] = df["novelty_label"].fillna("Unknown")

# Define palette including 'Unknown'
palette = {
    "High Novelty": "red",
    "Moderate Novelty": "orange",
    "Low Novelty": "yellow",
    "Unknown": "lightblue"
}

# ----------------------------
# Plot UMAP colored by novelty label
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="umap_x",
    y="umap_y",
    hue="novelty_label",
    palette=palette,
    alpha=0.6,
    s=8,
    linewidth=0
)

plt.title("UMAP Highlighting Novel Sequences")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Novelty", loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("umap_novelty_highlighted.png", dpi=300, bbox_inches='tight')
plt.show()
