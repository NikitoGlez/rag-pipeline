"""
Vector Store: ChromaDB

¿Qué es un vector store?
  Es una base de datos especializada en guardar vectores (embeddings) y
  hacer búsquedas por similitud muy rápido.

  Cuando preguntas algo:
  1. Tu pregunta se convierte en embedding
  2. ChromaDB busca los N vectores más cercanos (similitud coseno)
  3. Devuelve los chunks originales asociados a esos vectores

ChromaDB guarda todo en disco (carpeta chroma_db/) — sin servidor, sin config.
"""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = str(Path(__file__).parent.parent.parent / "chroma_db")
COLLECTION_NAME = "ai_papers"


def build_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Crea el vector store desde cero con los chunks.
    Guarda en disco para no recalcular cada vez.
    """
    print(f"Construyendo vector store con {len(chunks)} chunks...")
    print("(Esto tarda un par de minutos la primera vez)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store guardado en: {CHROMA_DIR}")
    return vectorstore


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Carga el vector store desde disco (sin recalcular embeddings)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Carga si existe, construye si no. Útil para notebooks."""
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        print("Vector store encontrado en disco. Cargando...")
        return load_vectorstore(embeddings)
    raise RuntimeError(
        "Vector store no encontrado. Ejecuta primero el notebook 01_naive_rag.ipynb "
        "para construirlo."
    )
