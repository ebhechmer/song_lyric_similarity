# Measuring Song Lyric Theme Similarity Using Sentence-BERT Embeddings

## 1. Project Overview

This project investigates whether Sentence-BERT embeddings of song lyrics can capture thematic similarity—measured via cosine similarity—and how that aligns with genre labels.

## 2. Scientific Questions

1. Do pre-trained Sentence-BERT embeddings reflect the semantic/themes of lyrics?
2. Can cosine similarity between lyrics embeddings serve as a proxy for “theme similarity”?
3. How do embedding-based similarities compare within vs. across the six provided genres?

## 3. Dataset

We use the “Genius Song Lyrics” CSV (~5.1 M rows), filtering to English only.

- **File:** `data/song_lyrics.csv`
- **Key columns:**
  - `tag` (genre)
  - `language`
  - `lyrics`

## 4. Project Structure

```
song-theme-similarity/
├─ notebooks/         ← exploratory & analysis notebooks
│  └─ 01_data_prep.ipynb
├─ src/               ← helper modules
│  └─ data_loader.py
├─ data/              ← raw & processed data
├─ results/           ← embeddings, similarity matrices, plots
├─ README.md
└─ requirements.txt
```

## 5. Getting Started

1. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Download the dataset:** place `song_lyrics.csv` into `data/`.
3. **Launch the first notebook for data preparation:**
   ```bash
   jupyter lab notebooks/01_data_prep.ipynb
   ```

## 6. Next Steps

- **Data Preparation:**
  - Filter to `language == 'en'`
  - Remove bracketed section markers (e.g. `[Chorus]`)
  - Collapse extra newlines
- **Embeddings:** compute SBERT vectors in batches, save to `results/`
- **Analysis:** generate cosine-similarity stats within vs. across genres
- **Visualization:** t-SNE/UMAP plots, similarity heatmaps, boxplots
- **Reporting:** compile findings in `report.md` and design poster
