---
title: Football RAG QA System
emoji: ⚽
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
---

# ⚽ Football RAG · Question-Answering System

A Retrieval-Augmented Generation (RAG) system for football knowledge, built on Wikipedia articles covering clubs, players, tournaments, and rules.

## Architecture

| Component | Choice |
|---|---|
| Corpus | Wikipedia football articles (clubs, players, tournaments, rules) |
| Chunking | Recursive (`CHUNK_SIZE=1000`, `OVERLAP=100`) |
| Vector DB | Pinecone (`sports-rag-v2`, namespace = `recursive`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (dim=384) with BGE query prefix |
| Retrieval | Dense (Pinecone) → CrossEncoder rerank → MMR deduplication |
| Re-ranker | `BAAI/bge-reranker-base` |
| MMR | λ=0.6, ensures diverse source coverage across top-5 chunks |
| LLM | Groq `llama-3.1-8b-instant` |
| Evaluation | LLM-as-Judge (Faithfulness + Relevancy) |

## Ablation Study Results

| Chunking | Retrieval | Faithfulness | Relevancy | Avg Response Time |
|---|---|---|---|---|
| fixed | semantic\_only | 0.978 | 0.883 | 5.01s |
| **recursive** | **hybrid+rerank** | **0.978** | **0.860** | **5.55s** |
| recursive | semantic\_only | 0.978 | 0.883 | 5.32s |
| fixed | hybrid+rerank | 0.978 | 0.859 | 5.60s |
| semantic | semantic\_only | 0.956 | 0.876 | 5.10s |
| semantic | hybrid+rerank | 0.933 | 0.868 | 5.24s |

## Required Secrets

Set in **HF Space Settings → Variables and secrets**:

- `PINECONE_API_KEY`
- `GROQ_API_KEY`

## Local Setup

```bash
pip install -r requirements.txt
export PINECONE_API_KEY=your_key
export GROQ_API_KEY=your_key
python app.py
```