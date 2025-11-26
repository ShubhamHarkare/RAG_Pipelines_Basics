import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import uvicorn
from huggingface_hub import InferenceClient
from data_loader import loadAndChunkPDF, embedText
from vector_db import QdrantStorage
from custome_types import RAGQueryResult, RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult

load_dotenv()

# Initialize Hugging Face Inference Client
hf_client = InferenceClient(token=os.getenv("HUGGINGFACE_API_TOKEN"))

inngest_client = inngest.Inngest(
    app_id="rag_application",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

app = FastAPI()


@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf')
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def __load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data['pdf_path']
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = loadAndChunkPDF(pdf_path)

        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def __upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embedText(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{'source': source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: __load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run('embed_and_upsert', lambda: __upsert(chunks_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id='Querying vector DB',
    trigger=inngest.TriggerEvent(event='rag/query')
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> RAGSearchResult:
    def __search(question: str, top_k: int = 5):
        query_vector = embedText([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=found['contexts'], sources=found['sources'])

    question = ctx.event.data['question']
    top_k = int(ctx.event.data.get('top_k', 5))
    found = await ctx.step.run("embed_and_search", lambda: __search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f' - {c}' for c in found.contexts)
    
    # Build the prompt for Hugging Face model
    system_prompt = "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."
    user_prompt = f"""Use the following context to answer the question.

Context:
{context_block}

Question: {question}

Answer concisely using only the context above:"""

    # Use Hugging Face Inference API instead of OpenAI
    # Using Mistral-7B-Instruct model (fast and good quality)
    try:
        response = hf_client.text_generation(
            prompt=f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        answer = response.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        answer = "Sorry, I couldn't generate an answer at this time."

    return {
        "answer": answer,
        "sources": found.sources,
        "num_context": len(found.contexts)
    }


@app.get('/')
def home():
    return {"message": "RAG API is running", "status": "healthy"}


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)