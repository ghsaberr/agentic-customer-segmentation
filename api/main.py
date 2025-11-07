from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os
import asyncio


from src.cluster_pipeline import run_clustering_pipeline
from src.agent import agent_for_cluster
from src.retrieval import build_or_load_faiss_index, safe_load_corpus
from src.cluster_profiler import cluster_profiler
from src.llm import load_llm
from src.llm_utils import call_llama


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


app = FastAPI(
    title="Customer Intelligence API",
    description="Microservice for clustering, profiling, and LLM-based action generation.",
    version="1.0.0",
)

# ---- Load global resources ----
DATA_PATH = "data/clustered_customers.parquet"
CORPUS_PATH = "data/corpus_passages.parquet"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

index, embeddings, meta = None, None, None
df_clustered = None
numeric_cols = []
llm = None


# ---- Models for input ----
class AgentRequest(BaseModel):
    cluster_id: int
    top_k: Optional[int] = 3


# ---- Utility loaders ----
def load_clustered_df():
    global df_clustered, numeric_cols
    if df_clustered is None:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail="Clustered dataset not found. Run /compute_clusters first.")
        df_clustered = pd.read_parquet(DATA_PATH)
        numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
    return df_clustered


# ---- Endpoints ----

@app.get("/")
def root():
    return {"message": "Customer Intelligence API is running."}


@app.post("/compute_clusters")
def compute_clusters():
    """
    Run the clustering pipeline end-to-end.
    """
    try:
        df, profile = run_clustering_pipeline()
        return JSONResponse({
            "message": "✅ Clustering pipeline executed successfully.",
            "n_rows": len(df),
            "n_clusters": df['cluster_kmeans'].nunique()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster_profile/{cluster_id}")
def get_cluster_profile(cluster_id: int):
    """
    Return summarized statistics for a given cluster.
    """
    try:
        df = load_clustered_df()
        prof = cluster_profiler(cluster_id, df, numeric_cols)
        return prof
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_agent")
def run_agent(request: AgentRequest):
    """
    Run the LLM agent for a given cluster_id.
    """
    try:
        df = load_clustered_df()
        global index, embeddings, meta
        if index is None:
            raise HTTPException(status_code=400, detail="Vector index not loaded. Run /rebuild_index first.")
        result = agent_for_cluster(
            cluster_id=request.cluster_id,
            df_clustered=df,
            index=index,
            embeddings=embeddings,
            meta=meta,
            numeric_cols=numeric_cols,
            llm=llm,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.on_event("startup")
def startup_event():
    """
    Loads LLM at startup to avoid reloading per request.
    """
    global llm
    try:
        llm = load_llm()
        print("✅ LLM loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load LLM: {e}")

INDEX_STATE = {
    "status": "idle",
    "records": 0,
    "embeddings_shape": None,
    "error": None
}


@app.get("/index_status")
async def index_status():

    return INDEX_STATE


def _rebuild_index_sync():

    try:
        INDEX_STATE["status"] = "building"
        INDEX_STATE["error"] = None

        corpus_df = safe_load_corpus("data/corpus_passages.parquet")

        index, embeddings, meta = build_or_load_faiss_index(corpus_df, "all-MiniLM-L6-v2")

        faiss.write_index(index, "data/faiss_index.bin")
        np.save("data/embeddings.npy", embeddings)
        pd.DataFrame(meta).to_json("data/faiss_meta.json", orient="records")

        INDEX_STATE["status"] = "ready"
        INDEX_STATE["records"] = len(corpus_df)
        INDEX_STATE["embeddings_shape"] = list(embeddings.shape)
        print("✅ FAISS index built and saved successfully.")

    except Exception as e:
        INDEX_STATE["status"] = "error"
        INDEX_STATE["error"] = str(e)
        print(f"❌ Error rebuilding index: {e}")


@app.post("/rebuild_index")
async def rebuild_index(background_tasks: BackgroundTasks):

    if INDEX_STATE["status"] == "building":
        return {"status": "busy", "message": "Index rebuild already in progress."}

    background_tasks.add_task(_rebuild_index_sync)
    return {"status": "started", "message": "Index rebuild started in background."}