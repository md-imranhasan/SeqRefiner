from pathlib import Path

def split_fasta(input_fasta, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_fasta, 'r') as f:
        content = f.read().split('>')[1:]

    for seq in content:
        lines = seq.strip().split('\n')
        header = lines[0]
        sequence = '\n'.join(lines[1:])

        # Use second token in ID (e.g., A0A023HIB6 from tr|A0A023HIB6|A0A023HIB6_HV1)
        filename_token = header.split('|')[1] if '|' in header else header.split()[0]
        output_file = output_dir / f"{filename_token}.fasta"

        with open(output_file, 'w') as out_f:
            out_f.write(f">{header}\n{sequence}")

    print(f"✅ Done! {len(content)} sequences written to {output_dir}")

# Run the script
if __name__ == "__main__":
    input_path = "/mnt/d/Bioinformatics/SeqRefiner/data/uniprot.fasta"
    output_path = "/mnt/d/Bioinformatics/SeqRefiner/data/split_sequences"
    split_fasta(input_path, output_path)
