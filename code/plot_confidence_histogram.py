import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("amr_ensemble_predictions.csv")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="confidence", order=["High", "Medium", "Low"], palette="Set2")
plt.title("Prediction Confidence Levels")
plt.xlabel("Confidence Category")
plt.ylabel("Number of Sequences")
plt.tight_layout()
plt.savefig("confidence_histogram.png", dpi=300)
plt.show()
