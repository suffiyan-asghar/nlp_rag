"""
retriever.py — Football RAG retrieval module
Extracted from football-rag-complete.ipynb.

Best configuration (ablation winner):
  Dense → CrossEncoder rerank → MMR dedup
  Chunking  : recursive
  Index     : sports-rag-v2   (namespace per strategy)
  Embedding : BAAI/bge-small-en-v1.5  (dim=384)

Required environment variables:
  PINECONE_API_KEY
  GROQ_API_KEY
"""

import os
import re
import json
import time
from collections import defaultdict

import numpy as np
from openai import OpenAI, RateLimitError as _RLE
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Constants (mirror notebook exactly) ───────────────────────────────────────
INDEX_NAME              = "sports-rag-v2"
BEST_NAMESPACE          = "recursive"
EMBEDDING_MODEL         = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL     = "BAAI/bge-reranker-base"
JUDGE_MODEL             = "llama-3.1-8b-instant"
QA_GEN_MODEL            = "llama-3.1-8b-instant"

TOP_K                   = 10
RRF_K                   = 60
DENSE_TOP_K             = 30
RERANK_CANDIDATES_TOP_K = 30
QA_CONTEXT_TOP_K        = 5
MMR_LAMBDA              = 0.6
MAX_CONTEXT_CHARS       = 2500

_RECENCY_KEYWORDS = {"current","now","today","latest","recent","2025","2026","2027","2028"}

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embed_model   = None
_cross_encoder = None
_pc_index      = None
_groq_client   = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _get_pc_index():
    global _pc_index
    if _pc_index is None:
        api_key = os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set.")
        pc = Pinecone(api_key=api_key)
        _pc_index = pc.Index(INDEX_NAME)
    return _pc_index


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _groq_client


# ── BGE query instruction prefix (cell 15) ────────────────────────────────────

def _query_instruction(q: str) -> str:
    return f"Represent this sentence for searching relevant passages: {q}"


# ── Dense retrieval ───────────────────────────────────────────────────────────

def dense_retrieve(query: str, namespace: str = BEST_NAMESPACE,
                   top_k: int = DENSE_TOP_K) -> list:
    em    = _get_embed_model()
    index = _get_pc_index()
    q_vec = em.encode(_query_instruction(query), normalize_embeddings=True).tolist()
    resp  = index.query(vector=q_vec, top_k=top_k,
                        namespace=namespace, include_metadata=True)
    results = []
    for m in resp.get("matches", []):
        meta = m.get("metadata", {}) or {}
        results.append({
            "id"          : m.get("id", ""),
            "score"       : float(m.get("score", 0.0)),
            "text"        : meta.get("text", ""),
            "source"      : meta.get("source", ""),
            "url"         : meta.get("url", ""),
            "fetched_date": meta.get("fetched_date", ""),
        })
    return results


# ── Cross-encoder reranking ───────────────────────────────────────────────────

def _cross_encoder_rerank(query: str, candidates: list,
                           top_k: int = RERANK_CANDIDATES_TOP_K) -> list:
    if not candidates:
        return []
    ce     = _get_cross_encoder()
    pairs  = [[query, c.get("text", "")] for c in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(
        [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)],
        key=lambda x: x["rerank_score"], reverse=True,
    )
    return ranked[:top_k]


# ── MMR deduplication (cell 19) ───────────────────────────────────────────────

def _mmr_deduplicate(query: str, candidates: list,
                     top_k: int = QA_CONTEXT_TOP_K,
                     lambda_mmr: float = MMR_LAMBDA) -> list:
    """
    Maximal Marginal Relevance selection.
    lambda_mmr=0.6 balances relevance with source diversity,
    preventing repeated chunks from the same article.
    """
    if len(candidates) <= top_k:
        return candidates
    em    = _get_embed_model()
    texts = [c.get("text", "") for c in candidates]
    vecs  = em.encode(texts, normalize_embeddings=True)
    q_vec = em.encode(_query_instruction(query), normalize_embeddings=True)

    rel_scores   = vecs @ q_vec
    selected_idx = []
    remaining    = list(range(len(candidates)))

    for _ in range(top_k):
        if not remaining:
            break
        if not selected_idx:
            best = max(remaining, key=lambda i: rel_scores[i])
        else:
            sel_vecs = vecs[selected_idx]
            best, best_score = None, -1e9
            for i in remaining:
                max_sim = float(np.max(sel_vecs @ vecs[i]))
                mmr_sc  = lambda_mmr * rel_scores[i] - (1 - lambda_mmr) * max_sim
                if mmr_sc > best_score:
                    best, best_score = i, mmr_sc
        selected_idx.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_idx]


# ── Full pipeline (mirrors retrieve_and_rerank, cell 19) ─────────────────────

def retrieve_and_rerank(query: str, namespace: str = BEST_NAMESPACE,
                         n_candidates: int = RERANK_CANDIDATES_TOP_K,
                         final_k: int = QA_CONTEXT_TOP_K) -> list:
    """
    Dense → CrossEncoder rerank → MMR dedup.
    (BM25 leg omitted on HF Spaces — no chunk JSONs available.
     Dense + rerank + MMR still matches the ablation semantic_only variant.)
    """
    dense_hits = dense_retrieve(query, namespace=namespace, top_k=n_candidates)
    reranked   = _cross_encoder_rerank(query, dense_hits, top_k=n_candidates)
    return _mmr_deduplicate(query, reranked, top_k=final_k)


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(hits: list, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, total = [], 0
    for r in hits:
        txt = r.get("text", "")
        if not txt or total + len(txt) > max_chars:
            break
        parts.append(txt)
        total += len(txt)
    return "\n\n".join(parts)


def may_be_stale(question: str) -> bool:
    return any(kw in question.lower() for kw in _RECENCY_KEYWORDS)


# ── Answer generation (cell 28) ───────────────────────────────────────────────

def _make_qa_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using ONLY the context provided below.\n"
        "Write a complete, fluent sentence.\n"
        "If the context does not contain enough information, "
        "respond with exactly: I don't know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def is_abstention(answer: str) -> bool:
    pats = ["i don't know", "i do not know", "insufficient context",
            "not enough context", "cannot determine", "can't determine", "unknown"]
    return any(p in (answer or "").strip().lower() for p in pats)


def generate_answer(question: str, context: str, max_tokens: int = 200) -> str:
    client = _get_groq_client()
    prompt = _make_qa_prompt(question, context)
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=QA_GEN_MODEL, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except _RLE:
            if attempt == 5:
                raise
            time.sleep(2.0 * (2 ** attempt))


# ── LLM-as-Judge (cell 30) ────────────────────────────────────────────────────

def _llm_json(sys_p: str, usr_p: str) -> dict:
    client = _get_groq_client()
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user",   "content": usr_p}],
            )
            return json.loads(resp.choices[0].message.content)
        except _RLE:
            if attempt == 5:
                raise
            time.sleep(2.0 * (2 ** attempt))
        except Exception:
            raise


def extract_claims(answer: str) -> list:
    data = _llm_json(
        "You extract factual atomic claims from answers.",
        "Extract atomic factual claims from the answer. "
        "Return JSON with key 'claims' as a list of short strings. "
        "Return an empty list if the answer contains no factual content.\n\n"
        f"Answer:\n{answer}"
    )
    claims = data.get("claims", [])
    return [str(c).strip() for c in (claims if isinstance(claims, list) else []) if str(c).strip()]


def verify_claim(question: str, claim: str, context: str) -> dict:
    data = _llm_json(
        "You verify whether a claim is supported by provided context only.",
        "Verify if the claim is supported by the context only. "
        "Return JSON with keys: label ('supported' or 'unsupported'), "
        "rationale (1-2 lines), evidence (short phrase or empty).\n\n"
        f"Question: {question}\n\nClaim: {claim}\n\nContext:\n{context}"
    )
    label = str(data.get("label", "unsupported")).strip().lower()
    if label not in {"supported", "unsupported"}:
        label = "unsupported"
    return {
        "claim"    : claim,
        "label"    : label,
        "rationale": str(data.get("rationale", "")).strip(),
        "evidence" : str(data.get("evidence",  "")).strip(),
    }


def generate_alt_questions(answer: str) -> list:
    data = _llm_json(
        "You generate alternative user queries that the given answer could respond to.",
        "Generate exactly 3 different questions this answer could respond to. "
        "Return JSON with key 'questions' as a list of exactly 3 strings.\n\n"
        f"Answer:\n{answer}"
    )
    qs = data.get("questions", [])
    return [str(q).strip() for q in (qs if isinstance(qs, list) else []) if str(q).strip()][:3]


def _cosine_sim(a: str, b: str) -> float:
    em   = _get_embed_model()
    vecs = em.encode([a, b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


def score_faithfulness(question: str, answer: str, context: str) -> tuple:
    """Returns (score 0–1, list of verification dicts)."""
    claims = extract_claims(answer)
    if not claims:
        return 0.0, []
    verifications = []
    for claim in claims:
        verifications.append(verify_claim(question, claim, context))
        time.sleep(0.15)
    supported = sum(1 for v in verifications if v["label"] == "supported")
    return supported / len(verifications), verifications


def score_relevancy(original_query: str, answer: str) -> tuple:
    """Returns (avg score 0–1, list of (alt_question, similarity) tuples)."""
    alt_qs = generate_alt_questions(answer)
    if not alt_qs:
        return 0.0, []
    sims = [_cosine_sim(original_query, q) for q in alt_qs]
    return float(np.mean(sims)), list(zip(alt_qs, sims))


# ── Main public entry point ───────────────────────────────────────────────────

def query(question: str, namespace: str = BEST_NAMESPACE) -> dict:
    """Full RAG pipeline. Returns answer, chunks, scores, latency."""
    t0 = time.perf_counter()

    t_r = time.perf_counter()
    hits = retrieve_and_rerank(question, namespace=namespace)
    retrieval_time = time.perf_counter() - t_r

    context = build_context(hits)

    t_g = time.perf_counter()
    answer = generate_answer(question, context)
    generation_time = time.perf_counter() - t_g

    faithfulness_score, faith_details = score_faithfulness(question, answer, context)
    relevancy_score,    rel_details   = score_relevancy(question, answer)

    return {
        "answer"              : answer,
        "context_chunks"      : hits,
        "faithfulness"        : round(faithfulness_score, 4),
        "relevancy"           : round(relevancy_score, 4),
        "faithfulness_details": faith_details,
        "relevancy_details"   : rel_details,
        "stale_warning"       : may_be_stale(question),
        "retrieval_time_sec"  : round(retrieval_time, 3),
        "generation_time_sec" : round(generation_time, 3),
        "total_time_sec"      : round(time.perf_counter() - t0, 3),
    }
