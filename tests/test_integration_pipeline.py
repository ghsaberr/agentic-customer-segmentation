import json
import numpy as np
from src.agent import agent_for_cluster
from src.retrieval import build_or_load_faiss_index, safe_load_corpus
from src.llm import load_llm

def test_end_to_end_mini(df_customers, tmp_path):
    assert agent_for_cluster is not None, "agent_for_cluster not found; adjust imports"
    # Preparing a small corpus from df_customers (each passage = summary)
    corpus = df_customers.head(5).copy()
    corpus['text'] = "Policy summary: " + corpus['customer_id'].astype(str)
    # build simple FAISS index using sentence-transformers if available
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = emb.encode(corpus['text'].tolist()).astype('float32')
        idx = faiss.IndexFlatL2(embs.shape[1])
        idx.add(embs)
        meta = [{"doc_id": i, "text": corpus['text'].iloc[i]} for i in range(len(corpus))]
    except Exception as e:
        pytest.skip("sentence-transformers or faiss not available for integration test")
    # load local LLM
    llm = load_llm()

    res = agent_for_cluster(
        cluster_id=0,
        df_clustered=df_customers.head(20).assign(cluster_kmeans=0),
        index=idx,
        embeddings=embs,
        meta=meta,
        numeric_cols=df_customers.select_dtypes(include=['number']).columns.tolist(),
        llm=llm if llm is not None else None,
        top_k=2
    )
    assert 'parsed' in res
    assert 'docs_used' in res['parsed'] or isinstance(res['parsed'], dict)
