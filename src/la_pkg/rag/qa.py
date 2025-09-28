"""QA logic for RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import google.generativeai as genai
import pandas as pd
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


class Citation(BaseModel):
    paper_id: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    quote: str


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class QAResult(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    sources_used: list[str]  # uniikit paper_id:t
    retrieved_k: int
    llm_model: str
    embed_model: str


def retrieve(
    question: str, index_dir: Path, k: int, chunks_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for a question."""
    # 1. BM25 search
    tokenized_corpus = [doc.split(" ") for doc in chunks_df["text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_bm25_indices = bm25_scores.argsort()[-50:][::-1]

    # 2. Vector search on BM25 results
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(name="papers")

    results = collection.query(
        query_texts=[question],
        n_results=k,
        where={"chunk_id": {"$in": top_n_bm25_indices.tolist()}},
    )

    # Combine and rerank (simple approach for now)
    retrieved_chunks = []
    if results["documents"]:
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunk = metadata.copy()
            chunk["text"] = doc
            retrieved_chunks.append(chunk)

    return retrieved_chunks


def run_qa(
    question: str,
    index_dir: Path,
    k: int,
    llm_provider: str,
    llm_model: str,
    out_path: Path,
) -> None:
    """Run the full QA pipeline for a single question."""
    chunks_path = Path("data/cache/chunks.parquet")
    if not chunks_path.exists():
        raise FileNotFoundError(
            "Chunks file not found, please run `la rag chunk` first."
        )
    chunks_df = pd.read_parquet(chunks_path)

    retrieved_chunks = retrieve(question, index_dir, k, chunks_df)

    # Create context for LLM
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    # Configure LLM
    if llm_provider == "google":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(llm_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Create prompt
    prompt = f"""Vastaa vain annetuista katkelmista. Jokaisesta väitteestä anna vähintään 1 suora sitaatti muodossa [paper_id:page_start–page_end]. Älä keksi sivunumeroita. Pyri 2+ eri lähteeseen.

Context:
{context}

Question: {question}

Answer:
"""

    # Generate answer
    response = model.generate_content(prompt)

    # Parse response and create QAResult (simplified for now)
    # In a real implementation, this would involve more sophisticated parsing
    qa_result = QAResult(
        question=question,
        answer=response.text,
        claims=[],  # Placeholder
        sources_used=[],  # Placeholder
        retrieved_k=k,
        llm_model=llm_model,
        embed_model="text-embedding-004",  # Placeholder
    )

    # Save result to JSONL
    with out_path.open("a", encoding="utf-8") as f:
        f.write(qa_result.model_dump_json() + "\n")

    print(f"QA complete for question: '{question}'. Result saved to {out_path}")
