import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load known AMR embeddings
df_known = pd.read_csv("amr_embeddings.csv")
df_known = df_known.dropna()
X_known = df_known[[col for col in df_known.columns if col.startswith("f")]].fillna(0).values
known_ids = df_known["id"].values
known_classes = df_known["amr_class"].values

# Load query embeddings (your 73K sequences)
df_query = pd.read_csv("esm2_embeddings.csv")
X_query = df_query[[col for col in df_query.columns if col.startswith("f")]].fillna(0).values
query_ids = df_query["id"].values

# Trace match source
trace_results = []

for i in tqdm(range(len(X_query))):
    vec = X_query[i].reshape(1, -1)
    sims = cosine_similarity(vec, X_known)
    top_idx = np.argmax(sims)
    trace_results.append({
        "id": query_ids[i],
        "matched_amr_gene_id": known_ids[top_idx],
        "matched_amr_class": known_classes[top_idx],
        "similarity_score": sims[0, top_idx]
    })

# Save results
df_trace = pd.DataFrame(trace_results)
df_trace.to_csv("amr_trace_results.csv", index=False)
print("✅ Traceability saved to amr_trace_results.csv")
