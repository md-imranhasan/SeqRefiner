import pandas as pd
import plotly.express as px

# Load merged UMAP + AMR data
df_umap = pd.read_csv("sequence_clusters.csv")
df_amr = pd.read_csv("amr_trace_results.csv")

df = df_umap.merge(df_amr[["id", "matched_amr_class", "matched_amr_gene_id", "similarity_score"]], on="id", how="left")
df["matched_amr_class"] = df["matched_amr_class"].fillna("unknown")

# Interactive UMAP plot
fig = px.scatter(
    df,
    x="umap_x",
    y="umap_y",
    color="matched_amr_class",
    hover_data=["id", "matched_amr_gene_id", "similarity_score"],
    title="UMAP of All Sequences Colored by Predicted AMR Class",
    color_discrete_sequence=px.colors.qualitative.Set3,
    width=1200,
    height=800
)

fig.update_layout(
    legend_title_text="AMR Class",
    margin=dict(l=20, r=20, t=40, b=20),
)

fig.write_html("interactive_umap_amr.html")
fig.show()
