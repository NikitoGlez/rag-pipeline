"""
Download the most influential AI papers from ArXiv.
Run: python data/download_papers.py
"""

import os
import urllib.request
import time

# Papers: (filename, arxiv_pdf_url, description)
PAPERS = [
    (
        "attention_is_all_you_need.pdf",
        "https://arxiv.org/pdf/1706.03762",
        "Attention Is All You Need (Transformers) — Vaswani et al. 2017",
    ),
    (
        "rag_original.pdf",
        "https://arxiv.org/pdf/2005.11401",
        "RAG — Retrieval-Augmented Generation — Lewis et al. 2020",
    ),
    (
        "bert.pdf",
        "https://arxiv.org/pdf/1810.04805",
        "BERT — Devlin et al. 2018",
    ),
    (
        "gpt3.pdf",
        "https://arxiv.org/pdf/2005.14165",
        "GPT-3 — Language Models are Few-Shot Learners — Brown et al. 2020",
    ),
    (
        "llama2.pdf",
        "https://arxiv.org/pdf/2307.09288",
        "LLaMA 2 — Touvron et al. 2023",
    ),
    (
        "chain_of_thought.pdf",
        "https://arxiv.org/pdf/2201.11903",
        "Chain-of-Thought Prompting — Wei et al. 2022",
    ),
    (
        "hyde.pdf",
        "https://arxiv.org/pdf/2212.10496",
        "HyDE — Hypothetical Document Embeddings — Gao et al. 2022",
    ),
    (
        "ragas.pdf",
        "https://arxiv.org/pdf/2309.15217",
        "RAGAS — Automated Evaluation of RAG — Es et al. 2023",
    ),
    (
        "mixtral.pdf",
        "https://arxiv.org/pdf/2401.04088",
        "Mixtral of Experts — Jiang et al. 2024",
    ),
    (
        "rag_survey.pdf",
        "https://arxiv.org/pdf/2312.10997",
        "RAG for Large Language Models — Survey — Gao et al. 2023",
    ),
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_paper(filename: str, url: str, description: str) -> None:
    dest = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(dest):
        print(f"  [skip] {filename} — ya existe")
        return
    print(f"  [↓] {description}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        with open(dest, "wb") as f:
            f.write(response.read())
    print(f"       → guardado en data/raw/{filename}")


if __name__ == "__main__":
    print(f"Descargando {len(PAPERS)} papers de ArXiv...\n")
    errors = []
    for filename, url, description in PAPERS:
        try:
            download_paper(filename, url, description)
            time.sleep(1)  # respeta el rate limit de arxiv
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            errors.append(filename)

    print(f"\nListo. Papers en: data/raw/")
    if errors:
        print(f"Fallaron: {errors} — inténtalo manualmente.")
