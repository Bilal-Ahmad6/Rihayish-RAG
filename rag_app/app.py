import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

# Import our local config module
import config
settings = config.settings


app = FastAPI(title="Real Estate RAG API (Phase 8)", version="0.2.0")


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    collection: Optional[str] = None


class QueryResult(BaseModel):
    id: str
    distance: float
    text: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    k: int
    collection: str
    results: List[QueryResult]


def get_collection(name: str):
    """Return an existing collection; raise if it does not exist to avoid silent empty queries."""
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    # Attempt strict get; fall back with clear error
    try:
        return client.get_collection(name=name, embedding_function=emb_fn)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found. Embed data first.")


def _price_normalize(meta: Dict[str, Any]) -> None:
    """Ensure price_numeric is set using price_numeric_pkr or parsing textual price if needed."""
    if meta is None:
        return
    if meta.get("price_numeric") is not None:
        return
    # price_numeric_pkr present?
    if meta.get("price_numeric_pkr") is not None:
        meta["price_numeric"] = meta.get("price_numeric_pkr")
        return
    # Try parse textual price fields
    for key in ("price", "price_text"):
        txt = str(meta.get(key) or "").lower()
        if not txt:
            continue
        # Simple crore/lakh parser
        m = re.search(r"(\d+(?:\.\d+)?)\s*(crore|cr)", txt)
        if m:
            meta["price_numeric"] = int(round(float(m.group(1)) * 10_000_000))
            return
        m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac)", txt)
        if m:
            meta["price_numeric"] = int(round(float(m.group(1)) * 100_000))
            return
        # Plain integer
        m = re.search(r"\b(\d{5,})\b", txt)
        if m:
            try:
                meta["price_numeric"] = int(m.group(1))
                return
            except Exception:
                pass


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    collection_name = req.collection or settings.collection_name
    collection = get_collection(collection_name)

    # Perform query; include distances for ranking transparency
    res = collection.query(query_texts=[req.question], n_results=req.k, include=["documents", "metadatas", "distances"])

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results: List[QueryResult] = []
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        # Ensure listing_id present (derive from rid if not)
        if isinstance(meta, dict):
            if not meta.get("listing_id") and rid:
                # rid format may be <listing>_chunk_n
                base_id = rid.split("_chunk_")[0]
                meta["listing_id"] = base_id
            _price_normalize(meta)
        results.append(QueryResult(id=rid, distance=float(dist), text=doc, metadata=meta or {}))

    return QueryResponse(question=req.question, k=req.k, collection=collection_name, results=results)

