"""
embedder.py
-----------
Parses source documents (PDF/TXT/MD), splits them into overlapping chunks,
generates vector embeddings, and persists them into a local ChromaDB collection.

Run directly:
    python embedder.py --source ./data --collection support_kb --reset

Supports two embedding backends (choose via EMBEDDING_BACKEND env var):
    - "openai"     -> OpenAI text-embedding-3-small (requires OPENAI_API_KEY)
    - "huggingface"-> local sentence-transformers model (no API key needed)
"""

import os
import argparse
import hashlib
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import chromadb
from chromadb.config import Settings

load_dotenv()

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "support_kb")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def get_embedding_function():
    """
    Returns a callable embedding function compatible with ChromaDB's
    `embedding_function` interface: __call__(input: List[str]) -> List[List[float]]
    """
    if EMBEDDING_BACKEND == "openai":
        from chromadb.utils import embedding_functions

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai"
            )
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
    else:
        from chromadb.utils import embedding_functions

        # Local, no API key required. all-MiniLM-L6-v2 is small and fast.
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )


def load_documents(source_dir: str) -> List[Document]:
    """Loads all supported files (.pdf, .txt, .md) from a directory into LangChain Documents."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    docs: List[Document] = []
    for file_path in sorted(source_path.rglob("*")):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()  # one Document per page, page metadata included
                for d in loaded:
                    d.metadata["source"] = file_path.name
                    # PyPDFLoader already sets metadata["page"] (0-indexed) -> normalize to 1-indexed
                    d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
                docs.extend(loaded)
            elif suffix in (".txt", ".md"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    d.metadata["source"] = file_path.name
                    d.metadata["page"] = 1
                docs.extend(loaded)
            else:
                continue
        except Exception as e:
            print(f"[embedder] Skipping {file_path.name}: {e}")

    if not docs:
        raise ValueError(f"No supported documents (.pdf/.txt/.md) found in {source_dir}")

    print(f"[embedder] Loaded {len(docs)} raw document sections from {source_dir}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Splits documents into overlapping chunks for retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[embedder] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def _stable_id(text: str, source: str, page: int, idx: int) -> str:
    """Deterministic chunk ID so re-running the embedder is idempotent (no duplicate rows)."""
    raw = f"{source}:{page}:{idx}:{text[:64]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embed_and_store(
    chunks: List[Document],
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = CHROMA_PERSIST_DIR,
    reset: bool = False,
) -> None:
    """Embeds chunks and upserts them into a persistent local ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[embedder] Reset existing collection '{collection_name}'")
        except Exception:
            pass

    embedding_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 1)
        ids.append(_stable_id(chunk.page_content, source, page, idx))
        texts.append(chunk.page_content)
        metadatas.append({"source": source, "page": page, "chunk_index": idx})

    # Batch upsert (Chroma upsert avoids duplicate errors on re-runs with same IDs)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"[embedder] Upserted batch {i // batch_size + 1} "
              f"({min(i + batch_size, len(ids))}/{len(ids)} chunks)")

    print(f"[embedder] Done. Collection '{collection_name}' now has "
          f"{collection.count()} vectors at '{persist_dir}'")


def run_ingestion(source_dir: str, collection_name: str, reset: bool) -> None:
    docs = load_documents(source_dir)
    chunks = chunk_documents(docs)
    embed_and_store(chunks, collection_name=collection_name, reset=reset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB for the support agent.")
    parser.add_argument("--source", default="./data", help="Directory containing source .pdf/.txt/.md files")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the collection from scratch")
    args = parser.parse_args()

    run_ingestion(args.source, args.collection, args.reset)
 