"""
Helper functions for loading and cleaning the Genius Song Lyrics dataset.
"""
import pandas as pd
import re


def load_lyrics_in_chunks(
    filepath: str,
    usecols: list = None,
    chunksize: int = 100_000
) -> pd.io.parsers.TextFileReader:
    """
    Lazily load CSV file in chunks.

    Parameters:
    - filepath: path to song_lyrics.csv
    - usecols: list of columns to load
    - chunksize: number of rows per chunk

    Returns:
    - an iterator over DataFrame chunks
    """
    return pd.read_csv(
        filepath,
        usecols=usecols,
        chunksize=chunksize,
        iterator=True
    )


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where language is English.

    Assumes a 'language' column with 'en' for English.
    """
    return df[df['language'] == 'en'].copy()


def clean_lyrics_column(df: pd.DataFrame, lyrics_col: str = 'lyrics') -> pd.DataFrame:
    """
    Remove bracketed section markers (e.g. [Chorus]) and collapse newlines.

    Parameters:
    - df: DataFrame containing lyrics
    - lyrics_col: name of the lyrics column

    Returns:
    - DataFrame with a new column 'clean_lyrics'
    """
    # remove [Section] markers
    pattern = re.compile(r"\[.*?\]")
    def _clean(text: str) -> str:
        no_markers = re.sub(pattern, '', text)
        # collapse multiple newlines and whitespace
        collapsed = re.sub(r"\s+", ' ', no_markers).strip()
        return collapsed

    df['clean_lyrics'] = df[lyrics_col].fillna('').apply(_clean)
    return df


def sample_and_save(df: pd.DataFrame, output_path: str, n: int = 100_000) -> None:
    """
    Randomly sample n rows and save to CSV.
    """
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample.to_csv(output_path, index=False)
