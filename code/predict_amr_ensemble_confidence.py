import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm

# Load known AMR embeddings
df_known = pd.read_csv("amr_embeddings.csv").dropna()
X_known = df_known[[col for col in df_known.columns if col.startswith("f")]].fillna(0).values
known_ids = df_known["id"].values
known_classes = df_known["amr_class"].values

# Load query (your 73K sequences)
df_query = pd.read_csv("esm2_embeddings.csv").dropna()
X_query = df_query[[col for col in df_query.columns if col.startswith("f")]].fillna(0).values
query_ids = df_query["id"].values

# Parameters
TOP_K = 5

results = []

for i in tqdm(range(len(X_query))):
    vec = X_query[i].reshape(1, -1)
    sims = cosine_similarity(vec, X_known)[0]

    top_k_idx = sims.argsort()[-TOP_K:][::-1]
    top_k_ids = known_ids[top_k_idx]
    top_k_classes = known_classes[top_k_idx]
    top_k_sims = sims[top_k_idx]

    # Majority vote with tie-breaking by mean similarity
    counter = Counter(top_k_classes)
    top_class = counter.most_common(1)[0][0]
    class_sim = np.mean([sim for cls, sim in zip(top_k_classes, top_k_sims) if cls == top_class])

    # Confidence scoring
    if class_sim >= 0.92:
        confidence = "High"
    elif class_sim >= 0.85:
        confidence = "Medium"
    else:
        confidence = "Low"

    results.append({
        "id": query_ids[i],
        "predicted_amr_class": top_class,
        "confidence": confidence,
        "avg_class_similarity": class_sim,
        "top_k_ids": ";".join(top_k_ids),
        "top_k_classes": ";".join(top_k_classes),
        "top_k_similarities": ";".join([f"{x:.3f}" for x in top_k_sims])
    })

df_out = pd.DataFrame(results)
df_out.to_csv("amr_ensemble_predictions.csv", index=False)
print("✅ Ensemble predictions with confidence saved to amr_ensemble_predictions.csv")
