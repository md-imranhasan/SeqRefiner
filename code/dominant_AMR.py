import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cluster summary
df_summary = pd.read_csv("cluster_amr_summary_filtered.csv")

# Plot the distribution of AMR classes per cluster (Horizontal Barplot)
plt.figure(figsize=(22, 16))  # Further increased figure size to fit labels better
sns.countplot(data=df_summary, y="cluster", hue="dominant_amr_class", palette="Set2")

# Add title and labels
plt.title("AMR Class Distribution by Cluster", fontsize=20)
plt.xlabel("Number of Sequences", fontsize=14)
plt.ylabel("Cluster ID", fontsize=14)

# Rotate y-axis labels for cluster IDs to prevent overlap
plt.yticks(rotation=0, fontsize=12)

# Rotate the AMR class labels (x-axis) for better readability
plt.xticks(rotation=45, ha="right", fontsize=12)

# Adjust layout to ensure everything fits
plt.tight_layout()

# Save and show the plot
plt.savefig("amr_class_distribution_by_cluster_horizontal.png", dpi=300)
plt.show()
