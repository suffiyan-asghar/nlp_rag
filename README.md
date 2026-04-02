# NLP Assignment 3: Football RAG System

This repository contains experiments and final notebooks for building and evaluating a football-focused Retrieval-Augmented Generation (RAG) pipeline for NLP Assignment 3.

## Project Overview

The project explores the full RAG workflow:
- Preparing and chunking a sports/football corpus.
- Testing retrieval strategies (e.g., BM25 vs hybrid retrieval).
- Running ablation experiments to compare design choices.
- Evaluating QA and LLM-judge performance with exported metrics.

## Repository Structure

- `Experimental_Notebooks/`: Iterative notebooks for preprocessing, retrieval experiments, and ablation workflows.
- `Final_Notebooks/`: Final end-to-end notebooks (with and without stored outputs).
- `Keyword Chunks/`: Precomputed corpus and chunking artifacts in JSON format.
- `Output files/`: Evaluation tables and experiment results in CSV format.
- `8-day-work-plan.md`: Project planning timeline.

## Main Outputs

The `Output files/` folder includes key result artifacts such as:
- Retrieval metrics
- Ablation summaries/details
- QA artifacts and winner comparisons
- LLM-judge examples and metrics

## Notes

Use the notebooks in `Final_Notebooks/` for the most complete and reproducible project flow, and `Experimental_Notebooks/` for intermediate exploration and comparisons.
