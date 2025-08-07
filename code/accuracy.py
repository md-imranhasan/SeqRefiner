import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Load known AMR-annotated data
df = pd.read_csv("amr_embeddings.csv").dropna()
X = df[[col for col in df.columns if col.startswith("f")]].fillna(0).values
y = df["amr_class"].values
ids = df["id"].values

y_pred = []

for i in tqdm(range(len(X))):
    query = X[i].reshape(1, -1)
    support = np.delete(X, i, axis=0)
    labels = np.delete(y, i)

    sims = cosine_similarity(query, support)[0]
    top_idx = np.argmax(sims)
    y_pred.append(labels[top_idx])

# Evaluate
print("✅ Self-Consistency Accuracy:")
print(classification_report(y, y_pred))
