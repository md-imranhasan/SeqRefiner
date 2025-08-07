import pandas as pd

# Load novelty scores and traceability results
df_novelty = pd.read_csv("novelty_scores.csv")
df_trace = pd.read_csv("amr_trace_results.csv")

# Merge novelty scores with traceability
df_novelty = df_novelty.merge(df_trace[["id", "matched_amr_gene_id", "similarity_score", "matched_amr_class"]],
                               on="id", how="left")

# Sort by novelty score (high novelty first)
df_novelty_sorted = df_novelty.sort_values(by="novelty_score", ascending=False).head(20)

# Select relevant columns
table2 = df_novelty_sorted[["id", "novelty_score", "matched_amr_gene_id", "similarity_score", "matched_amr_class"]]

# Save the table
table2.to_csv("top_novel_sequences.csv", index=False)
print("✅ Saved Table 2: top_novel_sequences.csv")
