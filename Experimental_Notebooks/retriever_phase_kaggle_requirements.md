# Retriever Phase - Kaggle Inputs, Hyperparameters, and Outputs

## Notebook
- `Notebooks/retriever_phase_experiments.ipynb`

## Required Inputs
- Chunk files:
  - `Keyword Chunks/chunks_fixed.json`
  - `Keyword Chunks/chunks_recursive.json`
  - `Keyword Chunks/chunks_semantic.json`
- Pinecone API key (required for dense + hybrid retrieval):
  - Environment variable: `PINECONE_API_KEY`
- Pinecone index and namespaces:
  - Index name: `sports-rag`
  - Namespace for fixed hybrid: `fixed`
  - Namespace for recursive hybrid: `recursive`
  - Namespace for semantic hybrid: `semantic`
- Manual evaluation set:
  - Exactly 25 questions in `evaluation_questions` list
  - Each question must include `relevant_chunk_ids`

## Optional Inputs
- If needed, set Hugging Face cache directory in Kaggle for faster reruns.

## Hyperparameters Used

### Retrieval
- `TOP_K = 10`
- `DENSE_TOP_K = 50`
- `RRF_K = 60`
- BM25 tokenizer: lowercased alphanumeric regex (`[A-Za-z0-9]+`)

### Dense Query Encoder
- Model: `BAAI/bge-small-en-v1.5`
- Normalization: `normalize_embeddings=True`

### QA (Final Winner Comparison)
- Model: `google/flan-t5-base`
- Generation config:
  - `max_new_tokens = 128`
  - `do_sample = False`

## What This Produces

### Main Results in Notebook
- Metrics table for retrieval experiments (`results_df`) with:
  - `Recall@5`
  - `Recall@10`
  - `MRR@10`
- Winner selection:
  - 1 best sparse (`bm25_*`)
  - 1 best hybrid (`hybrid_*`)
- QA comparison table (`qa_df`) only for those two winners

### Saved Files
- `outputs/retrieval_metrics.csv`
- `outputs/qa_winner_comparison.csv`

## Kaggle Setup Notes
- Upload project folder or ensure Kaggle notebook has access to the `Keyword Chunks` directory.
- In Kaggle, add `PINECONE_API_KEY` in Secrets and export it before running dense/hybrid cells.
- Run notebook top-to-bottom.
- Dense/hybrid experiments are added automatically when `PINECONE_API_KEY` is available; otherwise BM25-only runs are executed.
