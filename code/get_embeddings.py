import torch
import esm
import os
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO

# ----------------------------
# Step 1: Parse Sequences
# ----------------------------
def parse_sequences(folder_path, extensions=(".fasta", ".fa", ".fna")):
    sequences = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extensions):
            file_path = os.path.join(folder_path, file_name)
            for record in SeqIO.parse(file_path, "fasta"):
                # Optional: Skip long sequences
                if len(record.seq) <= 1000:
                    sequences.append((record.id, str(record.seq)))
    return sequences

# ----------------------------
# Step 2: Load ESM2 Model
# ----------------------------
print("Loading model...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ----------------------------
# Step 3: Process Sequences
# ----------------------------
data_dir = "/mnt/d/Bioinformatics/SeqRefiner/data"
sequences = parse_sequences(data_dir)

results = []
save_every = 25  # Save progress every N sequences
output_file = "esm2_embeddings.csv"

for i, (label, seq) in enumerate(tqdm(sequences, desc="Processing Sequences")):
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter([(label, seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results_esm = model(batch_tokens, repr_layers=[6])
            token_representations = results_esm["representations"][6]

        # Mean pooling of token representations (excluding [CLS], [EOS])
        embedding = token_representations[0, 1:len(seq)+1].mean(0).cpu().numpy()
        results.append((label, embedding))

        # Periodic saving
        if (i + 1) % save_every == 0 or (i + 1) == len(sequences):
            print(f"Saving checkpoint at {i+1} sequences...")
            df = pd.DataFrame([
                {"id": r[0], **{f"f{j}": val for j, val in enumerate(r[1])}}
                for r in results
            ])
            df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error processing {label}: {e}")
