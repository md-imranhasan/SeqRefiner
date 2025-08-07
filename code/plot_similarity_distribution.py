import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("amr_trace_results.csv")

plt.figure(figsize=(8, 5))
plt.hist(df["similarity_score"], bins=50, color='royalblue', edgecolor='black')
plt.title("Cosine Similarity to Nearest AMR Gene")
plt.xlabel("Similarity Score")
plt.ylabel("Sequence Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("similarity_score_distribution.png", dpi=300)
plt.show()
