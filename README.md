# 🔍 RAG Pipeline

> Retrieval-Augmented Generation from scratch — building and iterating on a production-ready RAG system using LangChain, ChromaDB and advanced retrieval techniques.
>
> ![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![LangChain](https://img.shields.io/badge/LangChain-0.3-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)
>
> ---
>
> ## 📌 Overview
>
> This project builds a RAG pipeline step by step, starting from a naive baseline and progressively adding advanced techniques to improve retrieval quality and answer accuracy.
>
> The goal is to have a real, evaluable system — not just a demo — with experiments tracked and results documented.
>
> ---
>
> ## 🗂️ Project Structure
>
> ```
> rag-pipeline/
> ├── data/                  # Raw and processed documents
> ├── notebooks/
> │   ├── 01_naive_rag.ipynb          # Baseline RAG
> │   ├── 02_advanced_retrieval.ipynb # HyDE, reranking, RAG-Fusion
> │   ├── 03_evaluation.ipynb         # RAGAS evaluation
> │   └── 04_app_demo.ipynb           # Streamlit demo
> ├── src/
> │   ├── ingestion/         # Document loading & chunking
> │   ├── retrieval/         # Vector store + retrieval strategies
> │   ├── generation/        # LLM chain & prompts
> │   └── evaluation/        # RAGAS metrics
> ├── app/                   # Streamlit demo app
> ├── requirements.txt
> └── README.md
> ```
>
> ---
>
> ## 🧱 Tech Stack
>
> | Component | Tool |
> |---|---|
> | Framework | LangChain 0.3 |
> | Vector DB | ChromaDB (local) / Pinecone (cloud) |
> | Embeddings | OpenAI `text-embedding-3-small` / BGE-M3 (free) |
> | LLM | OpenAI GPT-4o / Ollama (local) |
> | Evaluation | RAGAS |
> | Demo | Streamlit |
>
> ---
>
> ## 🚀 Roadmap
>
> - [ ] **Phase 1 — Naive RAG**: load docs → chunk → embed → retrieve → generate
> - [ ] - [ ] **Phase 2 — Advanced Retrieval**
> - [ ]   - [ ] HyDE (Hypothetical Document Embeddings)
> - [ ]     - [ ] RAG-Fusion (multi-query + reciprocal rank fusion)
> - [ ]   - [ ] CrossEncoder Reranking
> - [ ]   - [ ] **Phase 3 — Evaluation** with RAGAS (faithfulness, answer relevancy, context recall)
> - [ ]   - [ ] **Phase 4 — Streamlit App** with chat interface
>
> - [ ]   ---
>
> - [ ]   ## ⚙️ Setup
>
> - [ ]   ```bash
> - [ ]   git clone https://github.com/NikitoGlez/rag-pipeline.git
> - [ ]   cd rag-pipeline
> - [ ]   pip install -r requirements.txt
> - [ ]   cp .env.example .env  # add your API keys
> - [ ]   ```
>
> - [ ]   ---
>
> - [ ]   ## 📊 Evaluation Results
>
> - [ ]   *To be filled as experiments progress.*
>
> - [ ]   | Strategy | Faithfulness | Answer Relevancy | Context Recall |
> - [ ]   |---|---|---|---|
> - [ ]   | Naive RAG | - | - | - |
> - [ ]   | + HyDE | - | - | - |
> - [ ]   | + Reranking | - | - | - |
> - [ ]   | + RAG-Fusion | - | - | - |
>
> - [ ]   ---
>
> - [ ]   ## 📚 References
>
> - [ ]   - [RAG paper — Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
> - [ ]   - [RAGAS evaluation framework](https://github.com/explodinggradients/ragas)
> - [ ]   - [Advanced RAG techniques — LlamaIndex blog](https://www.llamaindex.ai/blog)
>
> - [ ]   ---
>
> - [ ]   *Part of my Data Science portfolio — built step by step, publicly.*
