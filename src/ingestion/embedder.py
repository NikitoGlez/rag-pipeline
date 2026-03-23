"""
Embeddings: convierte texto en vectores numéricos.

¿Qué es un embedding?
  Un embedding es una representación numérica del significado de un texto.
  Frases con significado similar tienen vectores "cercanos" en el espacio.

  Ejemplo:
    "¿Cómo funciona la atención en transformers?" → [0.12, -0.45, 0.78, ...]
    "Mecanismo de self-attention en redes neuronales" → [0.11, -0.44, 0.79, ...]
    "Receta de tortilla española" → [0.89, 0.23, -0.11, ...]
    (las dos primeras son cercanas; la tercera está lejos)

Modelo usado: BAAI/bge-m3
  - Gratuito, local (sin API)
  - Estado del arte en retrieval
  - Soporta 100+ idiomas
  - Ventana de 8192 tokens
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = "BAAI/bge-m3") -> HuggingFaceEmbeddings:
    """
    Devuelve el modelo de embeddings listo para usar.
    La primera vez descarga el modelo (~570 MB). Luego se cachea localmente.
    """
    print(f"Cargando modelo de embeddings: {model_name}")
    print("(Primera vez: descarga ~570 MB — después va al instante)")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # cambia a "cuda" si tienes GPU
        encode_kwargs={"normalize_embeddings": True},  # necesario para BGE
    )
    return embeddings
