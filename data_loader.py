from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from typing import List

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # Dimension for all-MiniLM-L6-v2

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=400)


embedding_model = SentenceTransformer(EMBED_MODEL)
def loadAndChunkPDF(path: str):
    """Load PDF and split into chunks"""
    docs = PDFReader().load_data(file=path)
    text = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in text:
        chunks.extend(splitter.split_text(t))

    return chunks


def embedText(text: List[str]) -> List[List[float]]:
    """
    Embed text using Hugging Face sentence-transformers
    Returns list of embeddings as lists of floats
    """
    embeddings = embedding_model.encode(
        text,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    # Convert numpy arrays to lists
    return [emb.tolist() for emb in embeddings]