from Bio import SeqIO
import os

def parse_sequences(folder_path, extensions=(".fasta", ".fa", ".fna")):
    sequences = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extensions):
            file_path = os.path.join(folder_path, file_name)
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append((record.id, str(record.seq)))
    return sequences
