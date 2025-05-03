# Measuring Song Lyric Theme Similarity Using Sentence-BERT Embeddings

## Overview

This project investigates whether pre-trained Sentence-BERT (SBERT) embeddings can capture thematic similarity in song lyrics. We use cosine similarity between 768-dimensional SBERT vectors to quantify thematic relatedness, compare within- and across-genre pairs, and evaluate how well unsupervised clustering and simple classification recover genre labels.

## Features

- **Data Preparation:** Filters the Genius Song Lyrics dataset (\~5.1M songs) to English, cleans bracketed markers, and stratified-samples 30K songs (5K per genre).
- **Embedding Computation:** Computes SBERT embeddings (`all-mpnet-base-v2`) on Apple M2 MPS or CPU.
- **Similarity Analysis:** Generates boxplots and bar charts comparing within- vs. across-genre cosine similarities.
- **Clustering:** Applies K-means (k=6), reports Adjusted Rand Index and silhouette score, and visualizes a confusion heatmap.
- **Classification:** Trains k-NN and logistic regression, presents per-genre precision/recall/F1 and overall performance.
- **Visualization:** Projects embeddings into 2D via UMAP, displays side-by-side maps colored by true genre and cluster.

## Project Structure

```
song-theme-similarity/
├─ notebooks/                     # Jupyter notebooks
│  ├─ 01_data_prep.ipynb         # Load, clean, stratified sampling
│  ├─ 02_compute_embeddings.ipynb# Compute and save SBERT embeddings
│  └─ 03_similarity_analysis.ipynb# Similarity, clustering, classification, UMAP
├─ src/                           # Helper modules
│  ├─ data_loader.py             # Chunked load, filter, clean, sample
│  └─ embeddings.py              # SBERT embedding functions
├─ data/                          # Raw & processed datasets
│  ├─ song_lyrics.csv            # Original Genius lyrics
│  ├─ sample_stratified.csv      # 30K stratified sample
├─ results/                       # Analysis outputs
│  ├─ embeddings_stratified.npy  # Saved embeddings
│  ├─ metadata_stratified.csv    # Corresponding genres
│  ├─ analysis_summary.csv       # Summary metrics
│  └─ plots/                      # Generated figures (PDF/PNG)
├─ report.md                      # Project report markdown
├─ poster.pdf                     # Final poster layout (PDF)
├─ README.md                      # This file
└─ requirements.txt               # Python dependencies
```

## Installation

1. **Clone the repo:**

   ```bash
   git clone https://github.com/youruser/song-theme-similarity.git
   cd song-theme-similarity
   ```

2. **Create a virtual environment** (Python 3.9+):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset:** place `song_lyrics.csv` into `data/` (see Kaggle).

## Usage

1. **Prepare data:**

   ```bash
   jupyter lab notebooks/01_data_prep.ipynb
   ```

2. **Compute embeddings:**

   ```bash
   jupyter lab notebooks/02_compute_embeddings.ipynb
   ```

3. **Run analyses & plots:**

   ```bash
   jupyter lab notebooks/03_similarity_analysis.ipynb
   ```

4. **Generate report & poster:**

   - Edit `report.md`, then convert to PDF if desired.
   - Open `poster.pdf` for final poster.

## License

This project is licensed under the MIT License.
