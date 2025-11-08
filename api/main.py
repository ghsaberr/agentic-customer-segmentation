from dotenv import load_dotenv
load_dotenv()  # this reads .env into os.environ

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from src.security import verify_admin

from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os
import asyncio
import faiss
import numpy as np
import json

from src.cluster_pipeline import run_clustering_pipeline
from src.agent import agent_for_cluster
from src.retrieval import build_or_load_faiss_index, safe_load_corpus
from src.cluster_profiler import cluster_profiler
from src.llm import load_llm


app = FastAPI(
    title="Customer Intelligence API",
    description="Microservice for clustering, profiling, and LLM-based action generation.",
    version="1.0.0",
)

# ---- Global constants ----
DATA_PATH = "data/clustered_customers.parquet"
CORPUS_PATH = "data/corpus_passages.parquet"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FAISS_INDEX_PATH = "data/faiss_index.bin"
EMB_PATH = "data/doc_embeddings.npy"
META_PATH = "data/doc_metadata.json"

# ---- Globals ----
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
        if index is None or embeddings is None or meta is None:
            raise HTTPException(status_code=400, detail="Vector index not loaded. Ensure FAISS files are present or rebuild the index.")
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


# ---- Startup loader ----
@app.on_event("startup")
def startup_event():
    """
    Load LLM and FAISS index at startup (if available).
    """
    global llm, index, embeddings, meta

    # Load LLM
    try:
        llm = load_llm()
        print("✅ LLM loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load LLM: {e}")

    # Load FAISS index + embeddings + metadata
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMB_PATH) and os.path.exists(META_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            embeddings = np.load(EMB_PATH)
            with open(META_PATH, "r", encoding="utf-8-sig") as f:
                meta = json.load(f)

            print(f"✅ FAISS index loaded successfully: {len(meta)} documents, dim={embeddings.shape[1]}")
        else:
            print("⚠️ FAISS index or embedding files not found. Run /rebuild_index if needed.")
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")



# ---- Index rebuild endpoints ----

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
    """
    Rebuild the FAISS index and save it to disk.
    """
    global index, embeddings, meta

    try:
        INDEX_STATE["status"] = "building"
        INDEX_STATE["error"] = None

        corpus_df = safe_load_corpus(CORPUS_PATH)
        index, embeddings, meta = build_or_load_faiss_index(corpus_df, EMB_MODEL_NAME)

        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(EMB_PATH, embeddings)
        pd.DataFrame(meta).to_json(META_PATH, orient="records")

        INDEX_STATE["status"] = "ready"
        INDEX_STATE["records"] = len(corpus_df)
        INDEX_STATE["embeddings_shape"] = list(embeddings.shape)
        print("✅ FAISS index built and saved successfully.")

    except Exception as e:
        INDEX_STATE["status"] = "error"
        INDEX_STATE["error"] = str(e)
        print(f"❌ Error rebuilding index: {e}")


@app.post("/rebuild_index")
async def rebuild_index(background_tasks: BackgroundTasks, authorized: bool = Depends(verify_admin)):
    """
    Trigger background rebuild of FAISS index.
    authorized == True
    """
    if INDEX_STATE["status"] == "building":
        return {"status": "busy", "message": "Index rebuild already in progress."}

    background_tasks.add_task(_rebuild_index_sync)
    return {"status": "started", "message": "Index rebuild started in background."}
