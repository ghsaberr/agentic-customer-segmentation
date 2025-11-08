from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os

def safe_load_corpus(path):
    """
    Loads a corpus DataFrame from a Parquet file, checking for existence first.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found at {path}")
    
    try:        
        df = pd.read_parquet(path)
        return df
    except Exception as e:        
        raise IOError(f"Failed to read parquet file at {path}. Error: {e}")

def build_or_load_faiss_index(corpus_df, emb_model_name):
    emb_model = SentenceTransformer(emb_model_name)
    embeddings = emb_model.encode(corpus_df["text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    meta = [{"doc_id": i, "text": corpus_df["text"].iloc[i]} for i in range(len(corpus_df))]
    return index, embeddings, meta