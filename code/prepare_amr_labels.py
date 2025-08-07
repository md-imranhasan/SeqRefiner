from Bio import SeqIO
import os
import pandas as pd

# === CARD ===
card_dir = "/mnt/d/Bioinformatics/SeqRefiner/Card"
card_fasta = os.path.join(card_dir, "protein_fasta_protein_homolog_model.fasta")
card_json = os.path.join(card_dir, "card.json")  # optional for metadata

card_labels = []
for record in SeqIO.parse(card_fasta, "fasta"):
    desc = record.description
    seq = str(record.seq)
    # Extract AMR class from description (you can refine this based on CARD JSON)
    if "|" in desc:
        parts = desc.split("|")
        label = parts[1].strip().lower()  # Use resistance class like "beta-lactamase"
        card_labels.append((record.id, seq, label))

df_card = pd.DataFrame(card_labels, columns=["id", "sequence", "amr_class"])
print(f"✅ Parsed {len(df_card)} CARD sequences")

# === ResFinder ===
resfinder_dir = "/mnt/d/Bioinformatics/SeqRefiner/ResFinder/resfinder_db"
resfinder_labels = []

for fname in os.listdir(resfinder_dir):
    if fname.endswith(".fsa"):
        label = fname.replace(".fsa", "").lower()
        fpath = os.path.join(resfinder_dir, fname)
        for record in SeqIO.parse(fpath, "fasta"):
            resfinder_labels.append((record.id, str(record.seq), label))

df_resfinder = pd.DataFrame(resfinder_labels, columns=["id", "sequence", "amr_class"])
print(f"✅ Parsed {len(df_resfinder)} ResFinder sequences")

# === Merge & Save ===
df_all = pd.concat([df_card, df_resfinder], ignore_index=True)
df_all.to_csv("amr_training_sequences.csv", index=False)
print(f"✅ Saved combined AMR training data: amr_training_sequences.csv")
