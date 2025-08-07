import os
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import esm
import torch

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "/mnt/d/Bioinformatics/SeqRefiner/ResFinder/resfinder_db"
OUT_CSV = "amr_embeddings.csv"
MAX_SEQ_LENGTH = 1000

# ----------------------------
# Extract FASTA + Labels
# ----------------------------
def load_labeled_sequences(folder, extensions=(".fsa",)):
    data = []
    for file in os.listdir(folder):
        if file.endswith(extensions):
            class_name = file.replace(".fsa", "")
            for record in SeqIO.parse(os.path.join(folder, file), "fasta"):
                seq = str(record.seq)
                if len(seq) <= MAX_SEQ_LENGTH:
                    data.append((record.id, seq, class_name))
    return pd.DataFrame(data, columns=["id", "sequence", "label"])

df = load_labeled_sequences(DATA_DIR)
print(f"✅ Loaded {len(df)} sequences from ResFinder")

# ----------------------------
# Load ESM2 Model
# ----------------------------
print("Loading ESM2...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ----------------------------
# Get Embeddings
# ----------------------------
results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        data = [(row["id"], row["sequence"])]
        labels, strs, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[6])
        rep = out["representations"][6]
        embedding = rep[0, 1:len(row["sequence"])+1].mean(0).cpu().numpy()
        results.append({
            "id": row["id"],
            "label": row["label"],
            **{f"f{i}": v for i, v in enumerate(embedding)}
        })
    except Exception as e:
        print(f"Error on {row['id']}: {e}")

# ----------------------------
# Save CSV
# ----------------------------
df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)
print(f"✅ Embeddings saved to {OUT_CSV}")
