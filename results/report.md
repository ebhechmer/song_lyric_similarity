# Measuring Song Lyric Theme Similarity Using Sentence-BERT Embeddings

## 1. Introduction

Song lyrics are rich in thematic content, yet traditional classification relies on metadata tags that may not capture nuanced semantic relationships. This project evaluates whether pre-trained Sentence-BERT embeddings can serve as an effective proxy for thematic similarity among songs, using cosine similarity in embedding space to compare lyrics.

## 2. Scientific Questions

- Can SBERT embeddings reflect genre-specific thematic cohesion within lyrics?
- How well does cosine similarity separate within-genre from across-genre lyric pairs?
- Can unsupervised clustering and simple classifiers recover or predict genre labels?

## 3. Methods

1. **Data Preparation**: Filtered \~3.4M English lyrics from the Genius dataset; cleaned bracketed markers; stratified sampled 30k songs (5k per genre).
2. **Embeddings**: Computed 768-dimensional SBERT vectors (all-mpnet-base-v2) on MPS backend.
3. **Analysis**:

   - **Similarity**: Sampled 2000 within- and across-genre pairs per genre; compared distributions and means.
   - **Clustering**: Applied K-means (k=6); evaluated with confusion crosstab and silhouette score.
   - **Classification**: Trained k-NN and logistic regression; evaluated per-genre precision, recall, and F1.

4. **Visualization**: Boxplots, bar charts, UMAP projections.

## 4. Results

### 4.1 Genre Similarity

- **Boxplots** show rap and country have tight within-genre distributions (median \~0.50) with clear gaps over across-genre pairs, while misc and rock overlap heavily.

### 4.2 Clustering

- **Confusion Matrix** indicates Cluster 0 aligns with rap (≈3500/5000), Cluster 2 with rb, and Cluster 4 with rock; some mixing in pop and country clusters.

### 4.3 Classification

```
# (Excerpt of classification_report)
              precision    recall  f1-score   support

         rap       0.70      0.68      0.69      1000
          rb       0.63      0.50      0.56      1000
        rock       0.45      0.44      0.44      1000
      country     0.75      0.80      0.77      1000
         pop       0.55      0.50      0.52      1000
        misc      0.48      0.40      0.43      1000
```

### 4.4 Embedding Space Visualization

Side-by-side UMAP shows distinct "islands" for rap, rb, and country, with pop and misc more intermingled, reflecting thematic overlap.

## 5. Discussion

- SBERT embeddings capture strong genre signals for rap and country, moderate for rb and rock, and weaker for pop and misc.
- Clustering aligns partially with true genres but mixing indicates thematic continuums.
- Classification performance (macro-averaged F1 ≈0.57) underscores varying thematic cohesion across genres.

## 6. Future Work

- Incorporate human similarity judgments to validate embedding-based measures.
- Experiment with alternative embedding models or fine-tuning on lyric data.
- Explore subgenre or era distinctions.

## 7. References

- Reimers and Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Genius Dataset (carlosgdcj/genius-song-lyrics-with-language-information).
