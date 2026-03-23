"""
Ingestion: carga PDFs y los divide en chunks.

¿Por qué chunking?
  Un LLM tiene un límite de contexto. No puedes meterle 30 páginas de un PDF.
  La idea es dividir el documento en trozos pequeños (chunks), indexarlos,
  y luego recuperar solo los más relevantes para cada pregunta.

Estrategia usada: RecursiveCharacterTextSplitter
  Intenta dividir por párrafo → frase → palabra, en ese orden.
  Es la estrategia más robusta para texto general.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def load_pdfs(data_dir: Path = DATA_DIR) -> List[Document]:
    """Carga todos los PDFs del directorio y devuelve una lista de Documents."""
    docs = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No hay PDFs en {data_dir}. Ejecuta primero: python data/download_papers.py"
        )

    for pdf_path in sorted(pdf_files):
        print(f"  Cargando: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Añade el nombre del paper como metadata
        for page in pages:
            page.metadata["source"] = pdf_path.stem
        docs.extend(pages)

    print(f"\nTotal páginas cargadas: {len(docs)}")
    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Divide los documentos en chunks.

    chunk_size: caracteres por chunk (~200 tokens ≈ 800 chars)
    chunk_overlap: solapamiento entre chunks para no perder contexto en los bordes
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks generados: {len(chunks)}")
    return chunks
