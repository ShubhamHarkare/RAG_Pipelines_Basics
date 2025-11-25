from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from typing import List

load_dotenv()
client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072


splitter = SentenceSplitter(chunk_size=1000,chunk_overlap=400)

#TODO: Loading data files
def loadAndChunkPDF(path:str):
    docs = PDFReader().load_data(file=path)
    text = [d.text for d in docs if getattr(d,"text",None)]
    chunks = []
    for t in text:
        chunks.extend(splitter.split_text(t))


    return chunks

def embedText(text:List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model = EMBED_MODEL,
        dimensions=EMBED_DIM,
        input=text
    )
    return [item.embedding for item in response.data]



