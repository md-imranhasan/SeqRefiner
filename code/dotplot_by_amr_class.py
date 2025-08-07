import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df = pd.read_csv("amr_ensemble_predictions.csv")

# Limit to top N classes
top_classes = df["predicted_amr_class"].value_counts().head(15).index
df_top = df[df["predicted_amr_class"].isin(top_classes)].copy()

# Assign x positions for each AMR class
df_top["position"] = df_top.groupby("predicted_amr_class").cumcount()

# Plot
plt.figure(figsize=(12, 6))
sns.stripplot(
    data=df_top,
    x="position",
    y="predicted_amr_class",
    hue="confidence",
    jitter=False,
    dodge=True,
    palette={"High": "green", "Medium": "orange", "Low": "red"},
    size=3,
    alpha=0.7,
    linewidth=0
)

plt.xlabel("Sequence Index (non-meaningful)")
plt.ylabel("Predicted AMR Class")
plt.title("Dot Plot of Sequences Grouped by Predicted AMR Class")
plt.legend(title="Confidence", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("dotplot_amr_classes.png", dpi=300, bbox_inches='tight')
plt.show()
