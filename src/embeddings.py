"""
Embedding utilities for song-theme-similarity project.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

import pandas as pd

# Cell X: Load your cleaned sample and build the list
SAMPLE_PATH = "../data/sample_100k.csv"
df = pd.read_csv(SAMPLE_PATH)
print("Sample shape:", df.shape)
# This is the crucial line that defines lyrics_list
lyrics_list = df["clean_lyrics"].tolist()

def embed_lyrics(
    lyrics_list: list[str],
    model_name: str = 'all-mpnet-base-v2',
    batch_size: int = 16,
    device: str = 'cpu',
    truncate_tokens: int = 512
) -> np.ndarray:
    """
    Compute SBERT embeddings for a list of lyrics in batches.
    Truncates long lyrics to the first truncate_tokens words to avoid token-length errors.

    Parameters:
    - lyrics_list: list of cleaned lyric strings
    - model_name: HuggingFace model name
    - batch_size: number of items per batch (reduced to manage memory)
    - device: computation device ('cpu' or 'cuda')
    - truncate_tokens: max number of words per lyric

    Returns:
    - embeddings: numpy array of shape (n_lyrics, embedding_dim)
    """
    # Load the model on specified device
    model = SentenceTransformer(model_name, device=device)
    embeddings = []
    for i in tqdm(range(0, len(lyrics_list), batch_size), desc="Embedding batches"):
        batch = lyrics_list[i : i + batch_size]
        # Truncate each lyric to avoid excessive tokens
        truncated = [' '.join(lyric.split()[:truncate_tokens]) for lyric in batch]
        # Compute embeddings
        emb = model.encode(truncated, convert_to_numpy=True, device=device)
        embeddings.append(emb)
    return np.vstack(embeddings)


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str
) -> None:
    """
    Save embeddings array to a .npy file.
    """
    np.save(output_path, embeddings)
