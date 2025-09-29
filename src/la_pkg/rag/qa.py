"""QA logic for RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import chromadb
import google.generativeai as genai
import pandas as pd
from chromadb.utils import embedding_functions
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


class MockOpenAIModel:
    """
    A mock OpenAI model for testing purposes.
    Returns a deterministic JSON response with at least two distinct sources.
    """
    def generate_content(self, prompt: str) -> MagicMock:
        # Extract paper_ids and page_starts from the prompt to create deterministic citations
        # This is a simplified extraction for mocking purposes.
        # In a real scenario, you might pass these in the ctor or have a more robust parsing.
        
        # For testing, return a deterministic JSON with at least 2 distinct sources
        # Use retrieved_chunks metadata from the prompt to construct citations
        # This is a simplified approach for the stub.
        
        # Parse the prompt to find paper_ids and page_ranges from the "Context:" section
        # This is a simplified parsing for the mock to make it somewhat dynamic
        citations_from_prompt = []
        context_start = prompt.find("Context:\n")
        if context_start != -1:
            context_text = prompt[context_start + len("Context:\n"):]
            # Regex to find "Document ID: pX, Section: Y (pages A-B)"
            import re
            matches = re.finditer(r"Document ID: (p\d+).*?\(pages (\d+)-(\d+)\)", context_text)
            for match in matches:
                paper_id, page_start, page_end = match.groups()
                citations_from_prompt.append({
                    "paper_id": paper_id,
                    "page_start": int(page_start),
                    "page_end": int(page_end),
                    "quote": f"Mocked quote from {paper_id} pages {page_start}-{page_end}"
                })
                if len(citations_from_prompt) >= 2:
                    break
        
        # Ensure at least two distinct sources, even if prompt parsing fails or yields less
        if len(citations_from_prompt) < 2:
            citations_from_prompt = [
                {"paper_id": "p1", "page_start": 1, "page_end": 1, "quote": "Mocked quote from p1"},
                {"paper_id": "p2", "page_start": 2, "page_end": 2, "quote": "Mocked quote from p2"},
            ]
        
        payload = {
            "answer": "Mocked answer from OpenAI stub. This answer is based on the provided context.",
            "claims": [{
                "text": "This is a mocked claim, citing multiple sources for demonstration.",
                "citations": citations_from_prompt,
            }],
            "sources_used": list(sorted(list(set(c["paper_id"] for c in citations_from_prompt)))),
        }
        return MagicMock(text=json.dumps(payload))


def retrieve(
    question: str,
    index_dir: Path,
    k: int,
    chunks_df: pd.DataFrame,
    embed_model: str,
    embed_provider: str,
    w_vec: float = 1.0,
    w_bm25: float = 0.5,
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

    # Map BM25 scores to chunk_ids and create a rank
    bm25_id_to_score = {
        chunks_df.iloc[i]["chunk_id"]: score
        for i, score in enumerate(bm25_scores)
        if score > 0
    }
    bm25_ranked_ids = sorted(
        bm25_id_to_score.keys(), key=lambda x: bm25_id_to_score[x], reverse=True
    )
    bm25_rank = {
        chunk_id: i for i, chunk_id in enumerate(bm25_ranked_ids)
    }  # 0-based rank

    # 2. Vector search (Chroma)
    vector_results = collection.query(
        query_texts=[question],
        n_results=k * 5,  # Retrieve more for robust hybrid ranking
        include=["metadatas", "distances"],
    )

    vector_id_to_distance = {}
    if vector_results["ids"] and vector_results["distances"]:
        for i, chunk_id in enumerate(vector_results["ids"][0]):
            vector_id_to_distance[chunk_id] = vector_results["distances"][0][i]

    vector_ranked_ids = sorted(
        vector_id_to_distance.keys(), key=lambda x: vector_id_to_distance[x]
    )
    vector_rank = {
        chunk_id: i for i, chunk_id in enumerate(vector_ranked_ids)
    }  # 0-based rank

    # 3. Combine chunk_ids from both methods (union)
    # Use dict.fromkeys to maintain order and get unique elements
    union_ids = list(dict.fromkeys(vector_ranked_ids + bm25_ranked_ids))

    if not union_ids:
        return []

    # 4. Retrieve full chunks from Chroma using combined IDs
    all_retrieved_data = collection.get(
        ids=union_ids,
        include=["documents", "metadatas"],
    )

    chunk_data_by_id = {}
    if all_retrieved_data["documents"]:
        for doc, metadata in zip(
            all_retrieved_data["documents"], all_retrieved_data["metadatas"]
        ):
            chunk = cast(dict[str, Any], metadata)
            chunk["text"] = doc
            chunk_data_by_id[chunk["chunk_id"]] = chunk

    # 5. Formulate scoring and re-rank
    scored_chunks = []
    # Use a large number for "infinity" for ranks not found
    inf_rank_vec = len(vector_ranked_ids) + 1
    inf_rank_bm25 = len(bm25_ranked_ids) + 1

    for chunk_id in union_ids:
        if chunk_id not in chunk_data_by_id:
            continue

        rank_vec = vector_rank.get(chunk_id, inf_rank_vec)
        rank_bm25 = bm25_rank.get(chunk_id, inf_rank_bm25)

        # Blended score: lower is better.
        # Example: score = (rank_vec * w_vec) + (rank_bm25 * w_bm25)
        # Or, to prioritize vector results if ranks are similar:
        score = (rank_vec * w_vec) + (rank_bm25 * w_bm25)

        chunk_data_by_id[chunk_id]["hybrid_score"] = score
        scored_chunks.append(chunk_data_by_id[chunk_id])

    # Sort by hybrid score in ascending order (lower score is better)
    sorted_retrieved_chunks = sorted(
        scored_chunks, key=lambda x: x["hybrid_score"]
    )

    return sorted_retrieved_chunks[:k]


def run_qa(
    question: str,
    index_dir: Path,
    k: int,
    llm_provider: str,
    llm_model: str,
    out_path: Path,
    chunks_path: Path | None = None,
    audit_path: Path | None = None,
) -> None:
    """Run the full QA pipeline for a single question."""
    # Read embed_model and embed_provider from index_meta.json
    index_meta_path = index_dir / "index_meta.json"
    if not index_meta_path.exists():
        raise FileNotFoundError(
            f"Index metadata file not found at {index_meta_path}. "
            "Please ensure the index was built correctly."
        )
    with index_meta_path.open("r", encoding="utf-8") as f:
        index_meta = json.load(f)

    embed_provider = index_meta.get("embed_provider")
    embed_model = index_meta.get("embed_model")
    
    # Resolve chunks_path: argument > index_meta > fallback
    resolved_chunks_path = chunks_path
    if resolved_chunks_path is None:
        chunks_path_from_meta = index_meta.get("chunks_path")
        if chunks_path_from_meta:
            resolved_chunks_path = Path(chunks_path_from_meta)
        else:
            resolved_chunks_path = Path("data/cache/chunks.parquet") # Fallback

    if not resolved_chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {resolved_chunks_path}. "
            "Please run `la rag chunk` first or specify --chunks-path."
        )
    chunks_df = pd.read_parquet(resolved_chunks_path)


    if not embed_provider or not embed_model:
        raise ValueError(
            "Embedding provider or model not found in index_meta.json. "
            "Ensure 'embed_provider' and 'embed_model' are present."
        )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY. Set it in .env.")

    def _generate_qa_response(current_k: int) -> tuple[QAResult, list[dict[str, Any]]]:
        retrieved_chunks = retrieve(
            question, index_dir, current_k, chunks_df, embed_model, embed_provider
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
        elif llm_provider == "openai":
            model = MockOpenAIModel() # Use the module-level mock
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

    if audit_path is None:
        audit_path = index_dir.parent / "logs" / "qa_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

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
            audit_path, mode="a", header=not audit_path.exists(), index=False
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(qa_result.model_dump_json() + "\n")

    print(f"QA complete for question: '{question}'. Result saved to {out_path}")
    print(
        f"Summary: {len(qa_result.claims)} claims, {len(qa_result.sources_used)} unique sources."
    )
