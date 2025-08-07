import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ----------------------------
# Load labeled AMR embeddings
# ----------------------------
df_known = pd.read_csv("amr_embeddings.csv")
df_known = df_known.dropna(subset=[col for col in df_known.columns if col.startswith("f")])
X_known = df_known[[col for col in df_known.columns if col.startswith("f")]]
y_known = df_known["amr_class"]

# Fill NaNs with column-wise mean (just in case)
X_known = X_known.fillna(X_known.mean())
X_known_np = X_known.values
y_known_np = y_known.values

# ----------------------------
# Load unlabeled embeddings
# ----------------------------
df_query = pd.read_csv("esm2_embeddings.csv")
X_query = df_query[[col for col in df_query.columns if col.startswith("f")]]
X_query = X_query.fillna(X_query.mean())  # fill NaNs with mean
X_query_np = X_query.values
ids_query = df_query["id"].values

# ----------------------------
# Predict AMR class via cosine similarity
# ----------------------------
print("🔍 Predicting AMR by nearest neighbor similarity...")
predicted_labels = []

for i in tqdm(range(len(X_query_np))):
    query_vec = X_query_np[i].reshape(1, -1)
    
    # Skip if still NaN (corrupted row)
    if np.isnan(query_vec).any():
        predicted_labels.append("unknown")
        continue

    sims = cosine_similarity(query_vec, X_known_np)
    top_idx = np.argmax(sims)
    predicted_labels.append(y_known_np[top_idx])

# ----------------------------
# Save results
# ----------------------------
df_query["amr_prediction"] = predicted_labels
df_query["id"] = ids_query
df_query.to_csv("esm2_with_amr_prediction.csv", index=False)
print("✅ Saved predictions to esm2_with_amr_prediction.csv")
