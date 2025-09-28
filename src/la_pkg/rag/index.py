"""Vector index building logic for RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import google.generativeai as genai
import pandas as pd
from chromadb.utils import embedding_functions
from pydantic import BaseModel


class IndexMeta(BaseModel):
    """Metadata about the vector index."""

    n_docs: int
    n_chunks: int
    vector_dim: int
    embed_provider: str
    embed_model: str


def build_index(
    chunks_path: Path,
    index_dir: Path,
    embed_provider: str,
    embed_model: str,
    batch_size: int,
) -> None:
    """Build a vector index from chunks."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    df = pd.read_parquet(chunks_path)
    if df.empty:
        print("No chunks to index.")
        return

    # Configure embeddings
    if embed_provider == "google":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.environ["GOOGLE_API_KEY"], model_name=embed_model
        )
    else:
        raise ValueError(f"Unsupported embed provider: {embed_provider}")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(
        name="papers", embedding_function=embedding_function
    )

    # Add documents to the collection in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]

        # Clean metadata: remove None values
        metadatas = batch.drop(columns=["text", "chunk_id"]).to_dict("records")
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned_meta = {k: v for k, v in meta.items() if v is not None}
            cleaned_metadatas.append(cleaned_meta)

        collection.add(
            ids=batch["chunk_id"].astype(str).tolist(),
            documents=batch["text"].tolist(),
            metadatas=cleaned_metadatas,
        )

    # Save index metadata
    vector_dim = 0
    peek_result = collection.peek()
    if (
        peek_result
        and peek_result["embeddings"]
        and len(list(peek_result["embeddings"])) > 0
    ):  # Explicitly convert to list
        vector_dim = len(peek_result["embeddings"][0])

    meta = IndexMeta(
        n_docs=df["paper_id"].nunique(),
        n_chunks=len(df),
        vector_dim=vector_dim,
        embed_provider=embed_provider,
        embed_model=embed_model,
    )
    meta_path = index_dir / "index_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, indent=2)

    print(f"Index built successfully. Metadata: {meta.model_dump_json(indent=2)}")


def get_index_stats(index_dir: Path) -> IndexMeta:
    """Get stats from an existing index."""
    meta_path = index_dir / "index_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "Index metadata not found. Please build the index first."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return IndexMeta(**data)


def verify_index(index_dir: Path, question_file: Path) -> bool:
    """Verify the index by running smoke test questions."""
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")

    with question_file.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(name="papers")

    all_passed = True
    for item in questions:
        question = item["question"]
        results = collection.query(query_texts=[question], n_results=1)
        if not results["documents"] or not results["documents"][0]:
            print(f"FAIL: No results for question: '{question}'")
            all_passed = False
        else:
            print(f"PASS: Got result for question: '{question}'")

    return all_passed
