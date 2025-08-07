import pandas as pd
from sklearn.metrics import accuracy_score

# Merge predictions with known AMR labels
df_pred = pd.read_csv("amr_ensemble_predictions.csv")
df_known = pd.read_csv("amr_embeddings.csv")[["id", "amr_class"]]
df = df_pred.merge(df_known, on="id", how="inner")

# Accuracy by confidence group
for level in ["High", "Medium", "Low"]:
    sub = df[df["confidence"] == level]
    acc = accuracy_score(sub["amr_class"], sub["predicted_amr_class"])
    print(f"{level} Confidence Accuracy: {acc:.2%} (n={len(sub)})")
