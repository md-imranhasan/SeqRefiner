import pandas as pd
import numpy as np

# Load the data
df_cluster = pd.read_csv("sequence_clusters.csv")
df_pred = pd.read_csv("amr_ensemble_predictions.csv")

# Handle NaN/Null values by assigning 'Medium' confidence in case of missing values
df_pred['predicted_amr_class'].fillna('Medium', inplace=True)

# Merge cluster data with AMR predictions
df = df_cluster.merge(df_pred[['id', 'predicted_amr_class']], on="id", how="left")

# Exclude noise (cluster == -1)
df_filtered = df[df['cluster'] != -1]

# Group by cluster to get the number of sequences and the dominant AMR class
cluster_summary = df_filtered.groupby('cluster').agg(
    num_sequences=('id', 'count'),
    dominant_amr_class=('predicted_amr_class', lambda x: x.mode()[0])
).reset_index()

# Add Purity and Example Members
cluster_summary['purity'] = cluster_summary.apply(lambda row: 
    (df_filtered[df_filtered['cluster'] == row['cluster']]['predicted_amr_class'].value_counts().get(row['dominant_amr_class'], 0) / row['num_sequences']) * 100, axis=1)

# Add a few example sequence IDs from each cluster
def get_example_members(cluster_id, top_n=3):
    example_ids = df_filtered[df_filtered['cluster'] == cluster_id].head(top_n)['id'].tolist()
    return ", ".join(example_ids)

cluster_summary['example_members'] = cluster_summary['cluster'].apply(lambda x: get_example_members(x))

# Save the summary table as a CSV
cluster_summary.to_csv("cluster_amr_summary_filtered.csv", index=False)

# Display the first few rows of the table
print(cluster_summary.head())
