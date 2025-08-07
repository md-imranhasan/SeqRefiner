import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("amr_ensemble_predictions.csv")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="confidence", y="avg_class_similarity", order=["High", "Medium", "Low"], palette="Set3")
plt.title("Similarity Score by Confidence Level")
plt.xlabel("Confidence")
plt.ylabel("Average Cosine Similarity")
plt.tight_layout()
plt.savefig("confidence_vs_similarity.png", dpi=300)
plt.show()
