"""
Document indexing module for Healthcare RAG.
Reads medical guidelines and public factsheets, segments text into overlapping chunks,
computes dense vector embeddings with sentence-transformers, and builds a persistent FAISS index.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unnecessary symbols."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks respecting sentence and section boundaries.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append({
                    "chunk_id": f"{source_name}_chunk_{chunk_idx:03d}",
                    "source_name": source_name,
                    "text": current_chunk,
                    "char_count": len(current_chunk),
                })
                chunk_idx += 1
                # Preserve overlap from tail of previous chunk
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = f"{overlap_text}\n\n{para}".strip()
            else:
                # Large single paragraph: split by character limit
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    sub_text = para[i : i + chunk_size].strip()
                    chunks.append({
                        "chunk_id": f"{source_name}_chunk_{chunk_idx:03d}",
                        "source_name": source_name,
                        "text": sub_text,
                        "char_count": len(sub_text),
                    })
                    chunk_idx += 1
                current_chunk = ""

    if current_chunk:
        chunks.append({
            "chunk_id": f"{source_name}_chunk_{chunk_idx:03d}",
            "source_name": source_name,
            "text": current_chunk,
            "char_count": len(current_chunk),
        })

    return chunks


def load_and_chunk_documents(docs_dir: Path = settings.DOCS_DIR) -> List[Dict[str, Any]]:
    """Ingest all text and markdown documents from documentation directory."""
    all_chunks = []
    doc_files = sorted(list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md")))
    
    print(f"Discovered {len(doc_files)} medical reference documents in {docs_dir}")
    for file_path in doc_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        cleaned = clean_text(content)
        source_name = file_path.name
        doc_chunks = chunk_text(cleaned, source_name)
        all_chunks.extend(doc_chunks)
        print(f" - {source_name}: {len(doc_chunks)} chunks generated.")

    return all_chunks


def build_faiss_index(
    chunks: List[Dict[str, Any]],
    output_dir: Path = settings.VECTOR_STORE_DIR,
    model_name: str = settings.EMBEDDING_MODEL_NAME,
):
    """
    Compute dense embeddings for chunks and serialize FAISS index & metadata.
    """
    if not chunks:
        raise ValueError("No text chunks found to index.")

    print(f"\nLoading embedding model: {model_name}...")
    embedder = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks into dense vector representations...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.json"
    npy_embeddings_path = output_dir / "embeddings.npy"

    # Save normalized embeddings for fast fallback / inspection
    np.save(npy_embeddings_path, embeddings)

    # Build FAISS Index with Inner Product (equivalent to Cosine Similarity for normalized vectors)
    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, str(index_path))
        print(f"FAISS index written successfully to {index_path}")
    else:
        print("FAISS not installed; saved raw normalized embeddings for pure-numpy vector search.")

    # Save chunks metadata
    metadata = {
        "embedding_model": model_name,
        "dimension": dim,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata index written successfully to {metadata_path}")
    print("=" * 60)
    print(f"Indexing completed: {len(chunks)} chunks indexed across {dim} dimensions.")
    print("=" * 60)


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    build_faiss_index(chunks)
