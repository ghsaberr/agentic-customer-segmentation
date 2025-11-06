def test_retrieval_sanity():
    try:
        from __main__ import build_or_load_faiss_index, safe_load_corpus
    except Exception:
        try:
            from src.retrieval import build_or_load_faiss_index, safe_load_corpus
        except Exception:
            build_or_load_faiss_index = None
    if build_or_load_faiss_index is None:
        import pytest
        pytest.skip("retrieval module not found")
    # If local corpus is available, run the test.
    try:
        corpus = safe_load_corpus("data/corpus_passages.parquet")
    except Exception:
        import pytest
        pytest.skip("corpus not available")
    idx, embs, meta = build_or_load_faiss_index(corpus, emb_model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Simple query
    D, I = idx.search(embs[:1], 3)
    assert I.shape[1] == 3
