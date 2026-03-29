# Industry RAG for Fintel

This project implements a high-performance RAG pipeline for complex financial datasets. Moving beyond naive RAG simple vector search, this architecture follows the industry RAG implementation - A multi-stage execution flow designed to minimize hallucinations and maximize context precision.

## Project Implementation

The goal is to bridge local prototyping with production systems using a hybrid hardware strategy (local macbook for orchestration, remote H100s for inference).

## Key Objectives:

- Context Precision: Implementing **Hybrid Search** and **Cross-Encoder Reranking** to ensure LLM only sees the most relevant context.
- Reasoning over Tables: Utilize `t2-ragbench` dataset to solve retrieving and information synthesis from financial tables.
- Quantifiable Quality: Using **RAGAS evaluations** (Faithfulness, Answer relevance, Context Precision)

## Industry RAG Flow

Unlike naive search and prompt script, this pipeline executes a linear series of nodes including:

1. **Query Transformation:** The system uses **Multi-Query Expansion** and **Hypothetical Document Embeddings** to turn vague user question into optimized search strategy.
2. **Hybrid Retrieval:** A dual search combining **Dense Vector Search** (semantic meaning) and **Sparse BM25 Search** (keyword/SKU/Entity accuracy), merged via **Reciprocal Rank Fusion** (RRF) in Qdrant.
3. **Cross-Encoder Reranking:** The top 20 candidates are filtered by specialized reranker model (BGE-Reranker-v2) to find top-k most useful chunks
4. **Synthesis (Streaming):** Final context is sent to high-parameter model (Qwen2.5) on the H100 cluster, streaming response back via FastAPI.

## Tech Stack

- **Orchestration:** LangGraph, FastAPI, PydanticAI
- **VectorDB:** Qdrant via Local Docker
- **Embeddings & Rerank:** BAAI BGE-Large (Local)
- **Synthesis:** vLLM hosting Qwen
- **Dataset:** G4KMU/t2-ragbench (FinQA)
