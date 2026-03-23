"""
Estrategias avanzadas de retrieval.

Problema del Naive RAG:
  Si preguntas "¿qué limitaciones tiene la atención multi-cabeza?",
  el sistema busca literalmente eso. Pero el paper puede hablar de
  "computational complexity of self-attention" sin usar esas palabras exactas.
  → Problema: vocabulary mismatch

Soluciones que implementamos aquí:

1. HyDE (Hypothetical Document Embeddings)
   → El LLM genera un párrafo hipotético que "respondería" la pregunta.
   → Buscamos ese párrafo en lugar de la pregunta original.
   → Funciona mejor porque el hipotético usa el vocabulario del dominio.

2. RAG-Fusion (Multi-Query + Reciprocal Rank Fusion)
   → Generamos 4 versiones distintas de la pregunta.
   → Buscamos con cada una por separado.
   → Fusionamos los resultados con RRF (los que aparecen en más listas suben).
   → Reduce el sesgo de una sola query.

3. CrossEncoder Reranking
   → Primero recuperamos más chunks (k=10).
   → Luego los reordenamos con un modelo que evalúa (pregunta, chunk) juntos.
   → Los CrossEncoders son más lentos pero mucho más precisos que coseno.
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

load_dotenv()


def _get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ─── 1. HyDE ─────────────────────────────────────────────────────────────────

HYDE_PROMPT = """Eres un investigador experto en IA. Escribe un párrafo técnico
de ~150 palabras que respondería directamente esta pregunta. No incluyas la
pregunta en tu respuesta, solo el párrafo.

Pregunta: {question}

Párrafo:"""


def retrieve_hyde(vectorstore: Chroma, question: str, k: int = 4) -> List[Document]:
    """HyDE: genera documento hipotético y busca con él."""
    llm = _get_llm()
    hypothetical = llm.invoke(HYDE_PROMPT.format(question=question)).content
    print(f"[HyDE] Documento hipotético generado ({len(hypothetical)} chars)")

    results = vectorstore.similarity_search(hypothetical, k=k)
    return results


# ─── 2. RAG-Fusion ───────────────────────────────────────────────────────────

MULTIQ_PROMPT = """Genera 4 versiones distintas de esta pregunta para buscar
en una base de datos de papers de IA. Varía el vocabulario y el ángulo.
Devuelve SOLO las 4 preguntas, una por línea, sin numeración.

Pregunta original: {question}

Preguntas alternativas:"""


def _reciprocal_rank_fusion(
    results_list: List[List[Document]], k: int = 60
) -> List[Document]:
    """
    Reciprocal Rank Fusion: combina múltiples rankings.
    Fórmula: RRF(d) = Σ 1/(k + rank(d))
    Documentos que aparecen en múltiples listas suben en el ranking final.
    """
    scores: dict[str, Tuple[float, Document]] = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.page_content[:100]  # identificador del chunk
            if key not in scores:
                scores[key] = (0.0, doc)
            current_score, _ = scores[key]
            scores[key] = (current_score + 1.0 / (k + rank + 1), doc)

    sorted_docs = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


def retrieve_rag_fusion(
    vectorstore: Chroma, question: str, k: int = 4
) -> List[Document]:
    """RAG-Fusion: multi-query + RRF."""
    llm = _get_llm()
    response = llm.invoke(MULTIQ_PROMPT.format(question=question)).content
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    queries = [question] + queries[:3]  # incluye la original
    print(f"[RAG-Fusion] {len(queries)} queries generadas")

    all_results = []
    for q in queries:
        results = vectorstore.similarity_search(q, k=k)
        all_results.append(results)

    fused = _reciprocal_rank_fusion(all_results)
    return fused[:k]


# ─── 3. CrossEncoder Reranking ────────────────────────────────────────────────

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def retrieve_with_reranking(
    vectorstore: Chroma,
    question: str,
    k_initial: int = 10,
    k_final: int = 4,
) -> List[Document]:
    """
    CrossEncoder Reranking:
    1. Recupera k_initial chunks con similitud coseno (rápido, menos preciso)
    2. Reordena con CrossEncoder (lento, muy preciso)
    3. Devuelve los k_final mejores
    """
    # Paso 1: recuperación inicial amplia
    candidates = vectorstore.similarity_search(question, k=k_initial)
    print(f"[Reranking] {len(candidates)} candidatos → reranking con CrossEncoder")

    # Paso 2: reranking
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = model.predict(pairs)

    # Paso 3: ordenar por score y devolver los mejores
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k_final]]
