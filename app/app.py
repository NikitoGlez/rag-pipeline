"""
Streamlit App — RAG Pipeline Demo

Interfaz de chat con selector de estrategia de retrieval.
Ejecutar con: streamlit run app/app.py
"""

import sys
import os
from pathlib import Path

# Añade el root al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.ingestion.embedder import get_embeddings
from src.retrieval.vectorstore import load_vectorstore
from src.generation.chain import format_docs, get_llm
from src.retrieval.strategies import retrieve_hyde, retrieve_rag_fusion, retrieve_with_reranking

# ─── Config ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Pipeline — AI Papers Q&A",
    page_icon="🔍",
    layout="wide",
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
Eres un experto en investigación en IA. Responde la pregunta ÚNICAMENTE basándote
en el contexto proporcionado. Si la información no está en el contexto, di
"No encuentro esa información en los documentos disponibles."

Contexto:
{context}

Pregunta: {question}
Respuesta:""")

STRATEGY_INFO = {
    "Naive RAG": "Búsqueda directa por similitud coseno. Simple y rápido.",
    "HyDE": "Genera un documento hipotético con el LLM y busca con él. Mejor para preguntas abstractas.",
    "RAG-Fusion": "Genera 4 variantes de la pregunta y fusiona resultados con Reciprocal Rank Fusion.",
    "CrossEncoder": "Recupera 10 candidatos y los reordena con un CrossEncoder. Más preciso.",
}

# ─── Cache (carga una sola vez) ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_resources():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    return vectorstore

# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("🔍 RAG Pipeline — AI Papers Q&A")
st.caption("Pregunta sobre los papers más influyentes de IA · Powered by BGE-M3 + Llama 3.3 70B (Groq) + ChromaDB")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    strategy = st.selectbox(
        "Estrategia de Retrieval",
        options=list(STRATEGY_INFO.keys()),
    )
    st.info(STRATEGY_INFO[strategy])

    st.divider()
    llm_provider = "groq"
    st.markdown("**Modelo LLM:** Llama 3.3 70B (Groq)")

    st.divider()
    k = st.slider("Chunks a recuperar (k)", min_value=2, max_value=8, value=4)

    st.divider()
    st.markdown("**Papers indexados:**")
    papers = [
        "Attention is All You Need",
        "RAG (Lewis et al. 2020)",
        "BERT",
        "GPT-3",
        "LLaMA 2",
        "Chain-of-Thought",
        "HyDE",
        "RAGAS",
        "Mixtral",
        "RAG Survey 2023",
    ]
    for p in papers:
        st.markdown(f"- {p}")

# Carga recursos
try:
    vectorstore = load_resources()
except Exception as e:
    st.error(f"Error cargando recursos: {e}\n\nAsegúrate de haber ejecutado el notebook 01 primero.")
    st.stop()

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input("Pregunta sobre los papers de IA..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Recuperando con {strategy}..."):
            # Retrieval según estrategia
            if strategy == "Naive RAG":
                docs = vectorstore.similarity_search(question, k=k)
            elif strategy == "HyDE":
                docs = retrieve_hyde(vectorstore, question, k=k)
            elif strategy == "RAG-Fusion":
                docs = retrieve_rag_fusion(vectorstore, question, k=k)
            elif strategy == "CrossEncoder":
                docs = retrieve_with_reranking(vectorstore, question, k_initial=k*2+2, k_final=k)

            # Generación con Groq
            llm, llm_name = get_llm(llm_provider)
            context = format_docs(docs)
            chain = RAG_PROMPT | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})

        st.caption(f"Generado con {llm_name}")
        st.markdown(answer)

        # Muestra las fuentes
        with st.expander(f"📄 Fuentes recuperadas ({len(docs)} chunks)"):
            for i, doc in enumerate(docs):
                st.markdown(f"**[{i+1}] {doc.metadata.get('source', '?')} — p.{doc.metadata.get('page', '?')}**")
                st.text(doc.page_content[:300] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
