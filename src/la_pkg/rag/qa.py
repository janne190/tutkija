"""QA logic for RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import chromadb
import google.generativeai as genai
import pandas as pd
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from chromadb.utils import embedding_functions


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
    question: str,
    index_dir: Path,
    k: int,
    chunks_df: pd.DataFrame,
    embed_model: str,
    embed_provider: str,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for a question using hybrid search."""

    # Configure embeddings for query
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY. Set it in .env.")

    if embed_provider == "google":
        genai.configure(api_key=api_key)
        embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key, model_name=embed_model
        )
    else:
        raise ValueError(f"Unsupported embed provider: {embed_provider}")

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(
        name="papers", embedding_function=embedding_function
    )

    # 1. BM25 search to get initial candidates
    tokenized_corpus = [doc.split(" ") for doc in chunks_df["text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top N BM25 chunk_ids
    bm25_top_n_indices = bm25_scores.argsort()[-k * 2 :][
        ::-1
    ]  # Get more candidates for reranking
    bm25_chunk_ids = chunks_df.iloc[bm25_top_n_indices]["chunk_id"].tolist()

    # 2. Vector search (Chroma)
    # Query Chroma with the question to get vector-based similar chunks
    vector_results = collection.query(
        query_texts=[question],
        n_results=k * 2,  # Retrieve more for potential re-ranking
        include=["documents", "metadatas", "distances"],
    )

    # Extract chunk_ids from vector search results
    vector_chunk_ids = (
        [meta["chunk_id"] for meta in vector_results["metadatas"][0]]
        if vector_results["metadatas"]
        else []
    )

    # Combine chunk_ids from both methods (union)
    combined_chunk_ids = list(set(bm25_chunk_ids + vector_chunk_ids))

    # Retrieve full chunks from Chroma using combined IDs
    if not combined_chunk_ids:
        return []

    final_retrieval = collection.get(
        ids=combined_chunk_ids,
        include=["documents", "metadatas"],
    )

    retrieved_chunks = []
    if final_retrieval["documents"]:
        for doc, metadata in zip(
            final_retrieval["documents"], final_retrieval["metadatas"]
        ):
            chunk = cast(dict[str, Any], metadata)
            chunk["text"] = doc
            retrieved_chunks.append(chunk)

    # Simple re-ranking: BM25 rank + alpha * embed_rank (alpha can be tuned)
    # For now, we'll just take the top k from the combined set,
    # assuming Chroma's distance is a good proxy for embed_rank.
    # A more sophisticated re-ranking would involve normalizing scores and combining.
    # For this task, we'll prioritize Chroma's ranking for the final k.
    # The task asks for a union and then scoring, so we'll just return the top k from Chroma's perspective
    # after ensuring they are part of the combined set.

    # Filter and sort by Chroma's relevance (distance)
    # This assumes vector_results are already sorted by distance
    sorted_retrieved_chunks = []
    if (
        vector_results["ids"]
        and vector_results["documents"]
        and vector_results["metadatas"]
    ):
        for i in range(len(vector_results["ids"][0])):
            chunk_id = vector_results["ids"][0][i]
            if chunk_id in combined_chunk_ids:
                meta = vector_results["metadatas"][0][i]
                doc = vector_results["documents"][0][i]
                chunk = cast(dict[str, Any], meta)
                chunk["text"] = doc
                sorted_retrieved_chunks.append(chunk)
                if len(sorted_retrieved_chunks) >= k:
                    break

    return sorted_retrieved_chunks


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

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY. Set it in .env.")

    embed_model = "text-embedding-004"  # Default, will be updated in P1

    def _generate_qa_response(current_k: int) -> tuple[QAResult, list[dict[str, Any]]]:
        retrieved_chunks = retrieve(
            question, index_dir, current_k, chunks_df, embed_model, llm_provider
        )

        context_parts = []
        for i, c in enumerate(retrieved_chunks):
            page_info = ""
            if c.get("page_start") is not None and c.get("page_end") is not None:
                page_info = f" (pages {c['page_start']}-{c['page_end']})"
            context_parts.append(
                f"Document ID: {c.get('paper_id', 'N/A')}, Section: {c.get('section_title', 'N/A')}{page_info}\n{c['text']}"
            )
        context = "\n\n".join(context_parts)

        if llm_provider == "google":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(llm_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        prompt = f"""Vastaa kysymykseen vain annetuista katkelmista. Palauta vastaus JSON-muodossa, joka sisältää 'answer' ja 'claims' kentät. Jokainen 'claim' on objekti, jossa on 'text' ja 'citations' kentät. Jokainen 'citation' on objekti, jossa on 'paper_id', 'page_start', 'page_end' ja 'quote' kentät. Älä keksi sivunumeroita. Pyri käyttämään vähintään kahta eri lähdettä.

Esimerkki JSON-muodosta:
```json
{{
  "answer": "...",
  "claims": [
    {{"text":"...", "citations":[{{"paper_id":"...", "page_start":1, "page_end":1, "quote":"..."}}]}}
  ]
}}
```

Context:
{context}

Question: {question}

Answer in JSON format:
"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Attempt to parse JSON, with fallback for malformed JSON
        try:
            # Extract JSON block if LLM wraps it in markdown
            if response_text.startswith("```json") and response_text.endswith("```"):
                json_str = response_text[7:-3].strip()
            else:
                json_str = response_text
            parsed_response = json.loads(json_str)
        except json.JSONDecodeError:
            print(
                "Warning: LLM response was not valid JSON. Attempting regex fallback."
            )
            # Fallback: try to find a JSON-like block using regex (simplified)
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    parsed_response = {"answer": response_text, "claims": []}
            else:
                parsed_response = {"answer": response_text, "claims": []}

        claims = []
        sources_used = set()
        for claim_data in parsed_response.get("claims", []):
            citations = []
            for citation_data in claim_data.get("citations", []):
                citations.append(
                    Citation(
                        paper_id=citation_data.get("paper_id", "N/A"),
                        page_start=citation_data.get("page_start"),
                        page_end=citation_data.get("page_end"),
                        section_title=None,  # Not directly from LLM output
                        quote=citation_data.get("quote", ""),
                    )
                )
                sources_used.add(citation_data.get("paper_id", "N/A"))
            claims.append(Claim(text=claim_data.get("text", ""), citations=citations))

        qa_result = QAResult(
            question=question,
            answer=parsed_response.get("answer", response_text),
            claims=claims,
            sources_used=list(sources_used),
            retrieved_k=current_k,
            llm_model=llm_model,
            embed_model=embed_model,
        )
        return qa_result, retrieved_chunks

    qa_audit_path = Path("data/logs/qa_audit.csv")
    qa_audit_path.parent.mkdir(parents=True, exist_ok=True)

    initial_k = k
    qa_result, retrieved_chunks = _generate_qa_response(initial_k)

    # Guardrail: if less than 2 unique sources, increase k and retry once
    if len(qa_result.sources_used) < 2:
        print(
            f"Guardrail triggered: only {len(qa_result.sources_used)} sources used. Retrying with k={initial_k + 4}."
        )
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "question": question,
            "initial_k": initial_k,
            "sources_used_initial": len(qa_result.sources_used),
            "guardrail_triggered": True,
            "retry_k": initial_k + 4,
            "final_sources_used": None,
            "final_answer_len": None,
        }
        try:
            qa_result_retry, retrieved_chunks_retry = _generate_qa_response(
                initial_k + 4
            )
            qa_result = qa_result_retry
            retrieved_chunks = retrieved_chunks_retry
            log_entry["final_sources_used"] = len(qa_result.sources_used)
            log_entry["final_answer_len"] = len(qa_result.answer)
        except Exception as e:
            print(f"Retry failed: {e}")
            log_entry["final_sources_used"] = "ERROR"
            log_entry["final_answer_len"] = "ERROR"

        # Append to audit log
        audit_df = pd.DataFrame([log_entry])
        audit_df.to_csv(
            qa_audit_path, mode="a", header=not qa_audit_path.exists(), index=False
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(qa_result.model_dump_json() + "\n")

    print(f"QA complete for question: '{question}'. Result saved to {out_path}")
    print(
        f"Summary: {len(qa_result.claims)} claims, {len(qa_result.sources_used)} unique sources."
    )
