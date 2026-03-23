"""
Generation: el LLM genera la respuesta usando los chunks recuperados.

¿Qué pasa aquí?
  1. Recuperamos los chunks más relevantes (retrieval)
  2. Los metemos en el prompt como "contexto"
  3. Le pedimos al LLM que responda SOLO basándose en ese contexto

LLM: Groq — Llama 3.3 70B (~300 tokens/seg, 14.400 req/día)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

load_dotenv()

# Prompt diseñado para minimizar alucinaciones
RAG_PROMPT = ChatPromptTemplate.from_template("""
Eres un experto en investigación en IA. Responde la pregunta ÚNICAMENTE basándote
en el contexto proporcionado. Si la información no está en el contexto, di
"No encuentro esa información en los documentos disponibles."

Contexto:
{context}

Pregunta: {question}

Respuesta:""")


def get_llm(provider: str = "groq"):
    """
    Devuelve el LLM de Groq.

    provider: ignorado, siempre usa Groq (mantenido por compatibilidad)
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    ), "Llama 3.3 70B (Groq)"


def format_docs(docs) -> str:
    """Formatea los chunks recuperados como texto para el prompt."""
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')} | p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vectorstore: Chroma, k: int = 4, provider: str = "auto"):
    """
    Construye el chain RAG completo.

    k: número de chunks a recuperar por pregunta
    provider: ignorado, siempre usa Groq (parámetro mantenido por compatibilidad)
    """
    llm, llm_name = get_llm(provider)
    print(f"LLM: {llm_name}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, question: str) -> str:
    """Hace una pregunta al chain y devuelve la respuesta."""
    return chain.invoke(question)
