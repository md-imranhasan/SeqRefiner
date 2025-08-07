import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")
df = df.merge(df_pred[["id", "predicted_amr_class"]], on="id", how="left")

# ----------------------------
# Bin UMAP coordinates
# ----------------------------
df["x_bin"] = (df["umap_x"] * 10).round().astype(int)
df["y_bin"] = (df["umap_y"] * 10).round().astype(int)

# Group by grid cell, assign majority AMR class
majority_class_per_bin = (
    df.groupby(["y_bin", "x_bin"])["predicted_amr_class"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

# Map AMR class to integers for heatmap
amr_classes = sorted(majority_class_per_bin["predicted_amr_class"].unique())
amr_to_int = {amr: i for i, amr in enumerate(amr_classes)}
majority_class_per_bin["amr_int"] = majority_class_per_bin["predicted_amr_class"].map(amr_to_int)

# Pivot to 2D matrix
pivot = majority_class_per_bin.pivot(index="y_bin", columns="x_bin", values="amr_int")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    pivot,
    cmap="tab20",  # or any suitable discrete colormap
    cbar=False,
    square=True,
    linewidths=0.05
)

# Build custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=plt.cm.tab20(amr_to_int[amr] / max(amr_to_int.values())), label=amr)
    for amr in amr_classes
]
plt.legend(handles=legend_elements, title="AMR Class", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.title("Non-overlapping UMAP Grid Colored by AMR Class")
plt.xlabel("UMAP X (binned)")
plt.ylabel("UMAP Y (binned)")
plt.tight_layout()
plt.savefig("umap_binned_heatmap_amr.png", dpi=300, bbox_inches='tight')
plt.show()
