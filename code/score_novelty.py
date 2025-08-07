import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load known AMR embeddings
df_known = pd.read_csv("amr_embeddings.csv").dropna()
X_known = df_known[[col for col in df_known.columns if col.startswith("f")]].fillna(0).values

# Load query
df_query = pd.read_csv("esm2_embeddings.csv").dropna()
X_query = df_query[[col for col in df_query.columns if col.startswith("f")]].fillna(0).values
query_ids = df_query["id"].values

novelty_scores = []

for i in tqdm(range(len(X_query))):
    vec = X_query[i].reshape(1, -1)
    sims = cosine_similarity(vec, X_known)[0]
    max_sim = np.max(sims)

    # Novelty scoring based on how far it is from known
    if max_sim < 0.80:
        label = "High Novelty"
    elif max_sim < 0.90:
        label = "Moderate Novelty"
    else:
        label = "Low Novelty"

    novelty_scores.append({
        "id": query_ids[i],
        "novelty_score": 1 - max_sim,
        "novelty_label": label
    })

df_novelty = pd.DataFrame(novelty_scores)
df_novelty.to_csv("novelty_scores.csv", index=False)
print("✅ Novelty scores saved to novelty_scores.csv")
