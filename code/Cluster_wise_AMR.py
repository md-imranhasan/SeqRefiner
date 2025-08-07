import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")

# Merge data
df = df.merge(df_pred[["id", "predicted_amr_class"]], on="id", how="left")

# Group data by cluster and AMR class
cluster_summary = df.groupby(['cluster', 'predicted_amr_class']).size().reset_index(name='count')

# Plot the bar plot
fig = px.bar(cluster_summary, x="cluster", y="count", color="predicted_amr_class",
             title="AMR Class Distribution Across Clusters",
             labels={'count': 'Number of Sequences', 'cluster': 'Cluster ID'},
             barmode='stack')

fig.update_layout(
    title="Interactive Cluster-wise AMR Class Distribution",
    xaxis_title="Cluster ID",
    yaxis_title="Count of Sequences"
)

# Save and show the figure
fig.write_html("interactive_cluster_amr_summary.html")
fig.show()
