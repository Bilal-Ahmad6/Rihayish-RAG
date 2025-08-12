#!/usr/bin/env python3
"""
FastAPI app for PropertyGuru RAG system with lazy loading to fix Render deployment issues.
Addresses "No open ports detected" and "WORKER TIMEOUT" by deferring heavy imports.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (inline to avoid extra imports)
# =============================================================================

class Settings(BaseSettings):
    """Application configuration with env overrides."""
    
    # Paths - use absolute paths resolved at runtime
    @property
    def data_dir(self) -> Path:
        return Path.cwd() / "data"
    
    @property 
    def raw_dir(self) -> Path:
        return Path.cwd() / "data" / "raw"
    
    @property
    def processed_dir(self) -> Path:
        return Path.cwd() / "data" / "processed"
    
    @property
    def chroma_persist_dir(self) -> Path:
        return Path.cwd() / "chromadb_data"

    # Models / RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "rihyesh_listings"

    # HTTP
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    requests_timeout: int = 20

    # Gemini API
    gemini_api_key: str = Field(default="", description="Gemini API key for LLM inference")

    class Config:
        # Align with rest of project / .env
        env_prefix = "GRAANA_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return Settings()

# =============================================================================
# Global variables for lazy loading
# =============================================================================

_chroma_client = None
_embedding_model = None

# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("PropertyGuru RAG API starting up...")
    settings = get_settings()
    logger.info(f"Using embedding model: {settings.embedding_model}")
    logger.info(f"ChromaDB path: {settings.chroma_persist_dir}")
    
    # Check if data directory exists
    if not settings.chroma_persist_dir.exists():
        logger.info(f"ChromaDB directory will be created on first query: {settings.chroma_persist_dir}")
    else:
        logger.info(f"ChromaDB directory found: {settings.chroma_persist_dir}")
    
    yield
    
    # Shutdown
    logger.info("PropertyGuru RAG API shutting down...")

# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="PropertyGuru RAG API",
    version="1.0.0",
    description="Real estate RAG system with lazy loading for deployment stability",
    lifespan=lifespan
)

# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about properties")
    k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    collection: Optional[str] = Field(default=None, description="Collection name (optional)")

class QueryResponse(BaseModel):
    question: str
    k: int
    results: List[Dict[str, Any]]
    collection_used: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

# =============================================================================
# Lazy Loading Functions
# =============================================================================

def get_chroma_client():
    """Lazy load ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        try:
            logger.info("Initializing ChromaDB client...")
            import chromadb
            settings = get_settings()
            _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    return _chroma_client

def get_embedding_model():
    """Lazy load SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            logger.info("Loading SentenceTransformer model...")
            from sentence_transformers import SentenceTransformer
            settings = get_settings()
            _embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding model initialization failed: {str(e)}")
    return _embedding_model

def get_collection(name: str):
    """Get or create a ChromaDB collection with embedding function."""
    try:
        # Import here to avoid loading at module level
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        
        client = get_chroma_client()
        settings = get_settings()
        
        # Create embedding function
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        
        # Get or create collection
        collection = client.get_or_create_collection(name=name, embedding_function=emb_fn)
        
        logger.info(f"Successfully accessed collection: {name}")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to get collection {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Collection access failed: {str(e)}")

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Fast health check endpoint for Render and load balancers."""
    return HealthResponse(
        status="healthy",
        message="Service is running",
        version="1.0.0"
    )

@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "PropertyGuru RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_properties(req: QueryRequest):
    """Query the property database using RAG."""
    start_time = time.time()
    
    try:
        settings = get_settings()
        collection_name = req.collection or settings.collection_name
        
        logger.info(f"Processing query: '{req.question}' (k={req.k}, collection={collection_name})")
        
        # Get collection with lazy loading
        collection = get_collection(collection_name)
        
        # Perform the query
        res = collection.query(
            query_texts=[req.question], 
            n_results=req.k
        )
        
        # Extract results
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        
        # Format results
        results: List[Dict[str, Any]] = []
        for rid, doc, meta in zip(ids, docs, metas):
            results.append({
                "id": rid,
                "text": doc,
                "metadata": meta or {}
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Query completed in {processing_time:.2f}ms, found {len(results)} results")
        
        return QueryResponse(
            question=req.question,
            k=req.k,
            results=results,
            collection_used=collection_name,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/collections")
async def list_collections():
    """List available collections."""
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        return {
            "collections": [col.name for col in collections],
            "count": len(collections)
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.get("/collection/{collection_name}/info")
async def collection_info(collection_name: str):
    """Get information about a specific collection."""
    try:
        collection = get_collection(collection_name)
        count = collection.count()
        return {
            "name": collection_name,
            "count": count
        }
    except Exception as e:
        logger.error(f"Failed to get collection info for {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Collection info failed: {str(e)}")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import time
    
    # Port handling for both local development and production
    port_env = os.environ.get("PORT", "10000")
    
    try:
        # Handle template variables that might not be resolved
        if port_env.startswith("${") and port_env.endswith("}"):
            port = 10000
            logger.warning(f"PORT env var appears to be unresolved template: {port_env}, using default 10000")
        elif port_env.startswith("$"):
            port = 10000  
            logger.warning(f"PORT env var appears to be unresolved: {port_env}, using default 10000")
        else:
            port = int(port_env)
            
        # Validate port range
        if port < 1 or port > 65535:
            port = 10000
            logger.warning(f"Invalid port {port_env}, using default 10000")
            
    except (ValueError, TypeError):
        port = 10000
        logger.warning(f"Invalid PORT value '{port_env}', using default 10000")
    
    logger.info(f"Starting FastAPI server on 0.0.0.0:{port}")
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
