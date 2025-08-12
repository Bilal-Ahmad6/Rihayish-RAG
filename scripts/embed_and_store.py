import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import chromadb
from sentence_transformers import SentenceTransformer  # type: ignore

# Add parent directory to path to import our local config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from scripts.utils import ensure_dirs, load_jsonl


def load_processed_map(processed_path: Path) -> Dict[str, dict]:
    if not processed_path.exists():
        return {}
    data = []
    try:
        with processed_path.open("r", encoding="utf-8") as f:
            import json

            data = json.load(f)
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    for rec in data:
        lid = rec.get("listing_id")
        if lid:
            out[str(lid)] = rec
    return out


def build_payloads(
    chunks_path: Path, processed_map: Dict[str, dict]
) -> Tuple[List[str], List[str], List[dict]]:
    records = load_jsonl(chunks_path)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    # Per-listing chunk numbering
    per_listing_index: Dict[str, int] = {}

    for r in records:
        lid = str(r.get("listing_id"))
        if not lid:
            # skip malformed
            continue
        per_listing_index[lid] = per_listing_index.get(lid, 0) + 1
        idx = per_listing_index[lid]
        cid = f"{lid}_chunk_{idx}"

        processed = processed_map.get(lid, {})
        ids.append(cid)
        text = r.get("text") or ""
        docs.append(text)

        metas.append(
            {
                "listing_id": lid,
                "url": r.get("url"),
                "price_numeric": processed.get("price_numeric"),
                "bedrooms": processed.get("bedrooms"),
                "bathrooms": processed.get("bathrooms"),
                "area_unit": processed.get("area_unit"),
                "chunk_type": r.get("chunk_type"),
                "text": text,
            }
        )

    return ids, docs, metas


def choose_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> List[List[float]]:
    embeddings: List[List[float]] = []
    # Use SentenceTransformers internal batching for efficiency
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=False, show_progress_bar=True)
    # vecs can be a numpy array or list; normalize to list of lists
    for v in vecs:
        if hasattr(v, "tolist"):
            embeddings.append(v.tolist())
        else:
            embeddings.append(list(v))
    return embeddings


def upsert_in_batches(
    collection, ids: List[str], docs: List[str], metas: List[dict], embeddings: List[List[float]], batch_size: int
) -> int:
    total = 0
    n = len(ids)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        chunk_ids = ids[i:j]
        chunk_docs = docs[i:j]
        chunk_metas = metas[i:j]
        chunk_embs = embeddings[i:j]
        # Prefer upsert if available to avoid duplicate ID errors
        if hasattr(collection, "upsert"):
            collection.upsert(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas, embeddings=chunk_embs)
        else:
            # Fallback to add
            collection.add(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas, embeddings=chunk_embs)
        total += len(chunk_ids)
    return total


def query(collection_name: str, question: str, k: int = 5, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = client.get_or_create_collection(name=collection_name)

    device = choose_device()
    model = SentenceTransformer(model_name, device=device)
    q_emb = embed_texts(model, [question], batch_size=1)[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["metadatas", "documents", "distances"])
    return res


def run(
    input_path: Path,
    processed_path: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    upsert_batch_size: int,
) -> None:
    ensure_dirs()

    t0 = time.time()

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = client.get_or_create_collection(name=collection_name)

    processed_map = load_processed_map(processed_path)
    ids, docs, metas = build_payloads(input_path, processed_map)
    if not ids:
        print("No chunks to embed.")
        return

    device = choose_device()
    print(f"Using model {model_name} on device: {device}")
    model = SentenceTransformer(model_name, device=device)

    embeddings = embed_texts(model, docs, batch_size=batch_size)

    total = upsert_in_batches(collection, ids, docs, metas, embeddings, upsert_batch_size)

    elapsed = time.time() - t0
    print(f"Embedded & stored {total} chunks into '{collection_name}' at {settings.chroma_persist_dir} in {elapsed:.2f}s")

    # Verification: 3 example queries
    examples = [
        "3 bed apartment near park in Phase 7",
        "10 marla house corner plot",
        "river hills 2 apartment 2 bed",
    ]
    for q in examples:
        print(f"\nQuery: {q}")
        res = query(collection_name, q, k=3, model_name=model_name)
        docs_out = res.get("documents", [[]])[0]
        metas_out = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, (doc, meta, dist) in enumerate(zip(docs_out, metas_out, dists), start=1):
            print(f" {i}. distance={dist:.4f}")
            if isinstance(meta, dict):
                print(
                    "    meta:",
                    {
                        "listing_id": meta.get("listing_id"),
                        "url": meta.get("url"),
                        "price_numeric": meta.get("price_numeric"),
                        "bedrooms": meta.get("bedrooms"),
                        "bathrooms": meta.get("bathrooms"),
                        "area_unit": meta.get("area_unit"),
                        "chunk_type": meta.get("chunk_type"),
                    },
                )
            # Print a short snippet of the text
            snippet = (doc or "")[:220]
            print(f"    text: {snippet}{'...' if doc and len(doc) > 220 else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and store in ChromaDB")
    parser.add_argument("--input", required=True, type=Path, help="Path to chunks JSONL file")
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/graana_phase8_processed.json"),
        help="Path to processed JSON array for metadata enrichment",
    )
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model to use for embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=256,
        help="Batch size for upserting into Chroma",
    )
    args = parser.parse_args()

    run(args.input, args.processed, args.collection, args.model, args.batch_size, args.upsert_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

