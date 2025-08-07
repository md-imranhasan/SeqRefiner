# SeqRefiner
SeqRefiner: A Scalable and Model-Free Transformer-Based Pipeline for Sequence Embedding, Clustering, and Functional Annotation


---

# **SeqRefiner: A Scalable and Model-Free Pipeline for Sequence Embedding, Clustering, and Functional Annotation**

**SeqRefiner** is a robust, scalable, and **model-free bioinformatics pipeline** designed to **embed**, **cluster**, and **annotate biological sequences**. It leverages **transformer-based embeddings** (e.g., **ESM2**) to provide **unsupervised clustering** and **nearest-neighbor AMR classification** without requiring supervised models. Additionally, SeqRefiner offers powerful tools for **novelty detection** and **traceability**, making it ideal for large-scale **antimicrobial resistance (AMR) gene annotation** and functional genomics research.

## Key Features:

* **Model-Free AMR Prediction**: Uses **ESM2 embeddings** for sequence-to-sequence comparisons and nearest-neighbor classification without the need for a supervised model.
* **Unsupervised Clustering**: Leverages **UMAP** and **HDBSCAN** to group sequences by functional similarities without the need for pre-labeled data.
* **Novelty Detection**: Identifies potentially novel sequences that do not match known AMR genes.
* **Confidence Scoring**: Provides **confidence scores** for each annotation, categorizing them as High, Medium, or Low confidence based on similarity to nearest neighbors.
* **Traceability**: Links each sequence prediction to its **top K nearest AMR gene references**, providing full transparency for results.
* **Extensible**: Designed to be adaptable for applications beyond AMR, such as **virulence factors**, **enzyme functions**, and **novel sequence discovery**.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Dependencies](#dependencies)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation

To get started with **SeqRefiner**, clone the repository and install the required dependencies:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/SeqRefiner.git
   cd SeqRefiner
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv seqrefiner_env
   source seqrefiner_env/bin/activate  # On Windows, use `seqrefiner_env\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Prepare Input Data**

* **Unlabeled Dataset**: Protein sequences in **FASTA format**.
* **Reference Datasets**: Known AMR genes from **CARD** and **ResFinder** databases.

### 2. **Run the Pipeline**:

* **Step 1**: Sequence embedding using ESM2

  ```bash
  python get_embeddings_large.py
  ```

* **Step 2**: Dimensionality reduction and clustering

  ```bash
  python cluster_and_annotate.py
  ```

* **Step 3**: AMR annotation via nearest-neighbor classification

  ```bash
  python predict_amr_ensemble_confidence.py
  ```

* **Step 4**: Traceability and novelty detection

  ```bash
  python trace_amr_sources.py
  python score_novelty.py
  ```

* **Step 5**: Cluster-wise AMR summary

  ```bash
  python cluster_amr_summary.py
  ```

* **Step 6**: Generate visualizations

  ```bash
  python plot_umap_by_amr.py
  python plot_confidence_histogram.py
  python plot_confidence_vs_similarity.py
  ```

---

## Features

1. **Transformer-Based Sequence Embedding**: Using **ESM2 embeddings** to capture deep relationships in protein sequences.
2. **Unsupervised Clustering**: Group similar sequences using **UMAP** and **HDBSCAN**.
3. **Nearest-Neighbor AMR Classification**: Classify sequences using **cosine similarity** with **reference AMR genes**.
4. **Confidence Scoring**: Calculate the reliability of each prediction based on top K nearest neighbors.
5. **Novelty Detection**: Flag sequences that are distant from known AMR genes, signaling potentially new or divergent sequences.
6. **Cluster-wise Functional Summaries**: Identify dominant AMR classes in each cluster and generate interpretable summaries.
7. **Visualization Tools**: Includes UMAP plots, confidence histograms, and novelty highlight plots to aid in result interpretation.

---

## Dependencies

* **Python 3.x**
* **ESM2 (Transformer Model)**
* **UMAP**: For dimensionality reduction
* **HDBSCAN**: For clustering
* **Seaborn/Matplotlib**: For visualizations
* **pandas**: For data manipulation
* **scikit-learn**: For metrics and nearest-neighbor search

Install dependencies using:

```bash
pip install -r requirements.txt
```

### **requirements.txt** (Example)

```txt
pandas
seaborn
matplotlib
umap-learn
hdbscan
scikit-learn
plotly
```

---

## Contributing

We welcome contributions to **SeqRefiner**! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-name`).
6. Open a pull request.

---

Feel free to adjust the README based on your project’s specific needs or additional details. If you'd like any section to be enhanced further, let me know!

