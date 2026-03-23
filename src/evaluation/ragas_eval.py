"""
Evaluación con RAGAS — optimizado para bajo consumo de tokens.

¿Por qué evaluar?
  Sin métricas, no sabes si HyDE mejoró o empeoró el sistema.
  RAGAS cuantifica la calidad de forma objetiva usando el propio LLM como juez.

Métricas usadas (2 en lugar de 4 para ahorrar tokens):
  - Faithfulness:       ¿La respuesta está respaldada por el contexto? (evita alucinaciones)
  - Answer Relevancy:   ¿La respuesta responde la pregunta?

Optimizaciones de tokens:
  - Modelo llama-3.1-8b-instant (8B vs 70B): ~9x más barato por llamada
  - 3 preguntas en lugar de 5
  - Contextos truncados a 400 chars por chunk
  - max_tokens=256 en el LLM juez
  - Embeddings locales (HuggingFace) — sin coste de API
"""

import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# Contextos truncados a este número de caracteres para reducir tokens enviados al LLM juez
CONTEXT_MAX_CHARS = 400

# 3 preguntas en lugar de 5 — cubre los temas principales con menos llamadas
EVAL_QUESTIONS = [
    {
        "question": "What is the core innovation of the Transformer architecture?",
        "ground_truth": "The Transformer replaces recurrent layers entirely with self-attention mechanisms, allowing for more parallelization and achieving better results on translation tasks.",
    },
    {
        "question": "What problem does RAG solve compared to standard language models?",
        "ground_truth": "RAG addresses the inability of parametric models to easily expand or revise their memory and their tendency to hallucinate facts, by grounding generation in retrieved documents from a non-parametric memory.",
    },
    {
        "question": "What is HyDE and how does it improve retrieval?",
        "ground_truth": "HyDE generates a hypothetical document using a language model that answers the query, then uses that document's embedding for retrieval instead of the query embedding, improving dense retrieval performance.",
    },
]


def _truncate_contexts(contexts: List[str], max_chars: int = CONTEXT_MAX_CHARS) -> List[str]:
    """Recorta cada chunk para no saturar el LLM juez."""
    return [c[:max_chars] for c in contexts]


def build_eval_dataset(
    questions: List[Dict],
    rag_chain,
    vectorstore,
) -> Dataset:
    """Ejecuta las preguntas y recoge respuestas + contextos para RAGAS."""
    rows = []
    for item in questions:
        q = item["question"]
        print(f"  Evaluando: {q[:60]}...")

        # Recuperar contextos (k=3 para menos tokens)
        docs = vectorstore.similarity_search(q, k=3)
        contexts = _truncate_contexts([doc.page_content for doc in docs])

        # Generar respuesta
        answer = rag_chain.invoke(q)

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    return Dataset.from_list(rows)


def run_evaluation(dataset: Dataset) -> pd.DataFrame:
    """
    Ejecuta RAGAS con configuración ahorrativa de tokens:
    - llama-3.1-8b-instant como juez (vs 70B)
    - Solo faithfulness + answer_relevancy (vs 4 métricas)
    - max_tokens=256
    """
    # 8b-instant: ~9x menos tokens que 70b-versatile
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=256,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )

    return result.to_pandas()


def print_summary(df: pd.DataFrame, strategy_name: str) -> Dict[str, float]:
    """Imprime resumen de métricas para una estrategia."""
    metrics = ["faithfulness", "answer_relevancy"]
    summary = {m: round(df[m].mean(), 3) if not df[m].isna().all() else float("nan") for m in metrics if m in df.columns}
    print(f"\n{'='*50}")
    print(f"Estrategia: {strategy_name}")
    print(f"{'='*50}")
    for metric, value in summary.items():
        if pd.isna(value):
            print(f"  {metric:<22}  N/A")
        else:
            bar = "█" * int(value * 20)
            print(f"  {metric:<22} {value:.3f}  {bar}")
    return summary
