import torch
import esm
import os
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "/mnt/d/Bioinformatics/SeqRefiner/data"
OUTPUT_CSV = "esm2_embeddings.csv"
SAVE_FREQ = 1  # Save after every sequence
MAX_SEQ_LENGTH = 1000  # Skip too long

# ----------------------------
# Load ESM2
# ----------------------------
print("Loading ESM2 model...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ----------------------------
# Load already processed IDs (for resuming)
# ----------------------------
if os.path.exists(OUTPUT_CSV):
    processed_ids = set(pd.read_csv(OUTPUT_CSV)["id"])
else:
    processed_ids = set()

# ----------------------------
# Parse sequences lazily
# ----------------------------
def yield_sequences(folder, extensions=(".fasta", ".fa", ".fna")):
    for fname in os.listdir(folder):
        if fname.endswith(extensions):
            fpath = os.path.join(folder, fname)
            for record in SeqIO.parse(fpath, "fasta"):
                if len(record.seq) <= MAX_SEQ_LENGTH:
                    yield (record.id, str(record.seq))

# ----------------------------
# Run embedding & save
# ----------------------------
with open(OUTPUT_CSV, "a") as fout:
    if os.stat(OUTPUT_CSV).st_size == 0:
        # Write header only if new file
        fout.write("id," + ",".join([f"f{i}" for i in range(384)]) + "\n")

    for seq_id, sequence in tqdm(yield_sequences(DATA_DIR), desc="Embedding Sequences"):
        if seq_id in processed_ids:
            continue  # Skip already processed

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter([(seq_id, sequence)])
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                out = model(batch_tokens, repr_layers=[6])
                reps = out["representations"][6]
                embedding = reps[0, 1:len(sequence)+1].mean(0).cpu().numpy()

            # Save line
            emb_str = ",".join([str(x) for x in embedding])
            fout.write(f"{seq_id},{emb_str}\n")
            fout.flush()  # Ensure it's written immediately
        except Exception as e:
            print(f"⚠️ Error processing {seq_id}: {e}")
