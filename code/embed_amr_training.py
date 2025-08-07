import torch
import esm
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Load ESM2 model
# ----------------------------
print("🔁 Loading ESM2 model...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ----------------------------
# Load labeled AMR sequences
# ----------------------------
df = pd.read_csv("amr_training_sequences.csv")
print(f"🔍 Loaded {len(df)} sequences")

results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    seq_id = row["id"]
    seq = row["sequence"]
    label = row["amr_class"]

    # Skip long sequences
    if len(seq) > 1000:
        continue

    try:
        batch_labels, batch_strs, batch_tokens = batch_converter([(seq_id, seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            output = model(batch_tokens, repr_layers=[6])
            reps = output["representations"][6]
            embedding = reps[0, 1:len(seq)+1].mean(0).cpu().numpy()

        result = {"id": seq_id, "amr_class": label}
        result.update({f"f{i}": val for i, val in enumerate(embedding)})
        results.append(result)

    except Exception as e:
        print(f"⚠️ Skipping {seq_id}: {e}")

# ----------------------------
# Save embeddings
# ----------------------------
df_embed = pd.DataFrame(results)
df_embed.to_csv("amr_embeddings.csv", index=False)
print("✅ Saved embeddings to amr_embeddings.csv")
