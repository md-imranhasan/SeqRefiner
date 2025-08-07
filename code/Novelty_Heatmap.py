import pandas as pd
import plotly.express as px

# Load and process the data
df = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")
df_novel = pd.read_csv("novelty_scores.csv")

# Merge dataframes
df = df.merge(df_pred[["id", "predicted_amr_class", "confidence"]], on="id", how="left")
df = df.merge(df_novel[["id", "novelty_label"]], on="id", how="left")
df["novelty_label"] = df["novelty_label"].fillna("Unknown")

# Group data by AMR class and novelty
heatmap_df = df.groupby(["predicted_amr_class", "novelty_label"]).size().reset_index(name='count')

# Plot the heatmap
fig = px.density_heatmap(
    heatmap_df,
    x="predicted_amr_class", 
    y="novelty_label", 
    z="count",
    color_continuous_scale="Viridis",
    title="Novelty Distribution Across AMR Classes"
)

fig.update_layout(
    title="Novelty Score Distribution by AMR Class",
    xaxis_title="AMR Class",
    yaxis_title="Novelty Label"
)

# Save and show the figure
fig.write_html("interactive_novelty_heatmap.html")
fig.show()
