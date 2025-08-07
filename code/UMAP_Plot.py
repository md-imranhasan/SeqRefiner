import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")
df_novel = pd.read_csv("novelty_scores.csv")

# Merge datasets
df = df.merge(df_pred[["id", "predicted_amr_class", "confidence"]], on="id", how="left")
df = df.merge(df_novel[["id", "novelty_label"]], on="id", how="left")
df["novelty_label"] = df["novelty_label"].fillna("Unknown")

# Create interactive UMAP plot
fig = px.scatter(df, x="umap_x", y="umap_y", color="predicted_amr_class", 
                 hover_data=["id", "predicted_amr_class", "novelty_label", "confidence"],
                 title="Interactive UMAP - AMR Class Prediction")

fig.update_layout(
    title="Interactive UMAP of Sequences Colored by AMR Class",
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2"
)

# Save and show the figure
fig.write_html("interactive_umap_amr_class.html")
fig.show()
