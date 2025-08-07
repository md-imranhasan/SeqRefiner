import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")
df_novel = pd.read_csv("novelty_scores.csv")

# Merge data
df = df.merge(df_pred[["id", "predicted_amr_class", "confidence"]], on="id", how="left")
df = df.merge(df_novel[["id", "novelty_label"]], on="id", how="left")
df["novelty_label"] = df["novelty_label"].fillna("Unknown")

# Create the interactive UMAP plot with novelty highlight
fig = px.scatter(df, x="umap_x", y="umap_y", color="novelty_label", 
                 color_discrete_map={"High Novelty": "red", "Moderate Novelty": "orange", 
                                     "Low Novelty": "gray", "Unknown": "lightblue"},
                 title="Interactive UMAP Highlighting Novel Sequences",
                 hover_data=["id", "predicted_amr_class", "novelty_label", "confidence"])

fig.update_layout(
    title="UMAP with Novelty Sequences Highlighted",
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2"
)

# Save and show the figure
fig.write_html("interactive_umap_novelty_highlighted.html")
fig.show()
