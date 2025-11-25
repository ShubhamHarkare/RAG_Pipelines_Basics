import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental import ai
import uvicorn
from data_loader import loadAndChunkPDF,embedText
from vector_db import QdrantStorage
from custome_types import RAGQueryResult,RAGChunkAndSrc,RAGSearchResult,RAGUpsertResult
#DESC: Vector database solutions


load_dotenv()
inngest_client = inngest.Inngest(
    app_id ="rag_application",
    logger = logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer()
)

app = FastAPI()
@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf')
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def __load(ctx:inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data['pdf_path']
        source_id = ctx.event.data.get("source_id",pdf_path)
        chunks = loadAndChunkPDF(pdf_path)

        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)


    def __upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embedText(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{'source':source_id,"text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids,vecs,payloads)
        return RAGUpsertResult(ingested=len(chunks))



    chunks_and_src = await ctx.step.run("load-and-chunk",lambda: __load(ctx),output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run('embed_and_upsert',lambda: __upsert(chunks_and_src),output_type=RAGUpsertResult)

    return ingested.model_dump()


#TODO: Function to query our vector database
@inngest_client.create_function(
        fn_id='Querying vector DB',
        trigger=inngest.TriggerEvent(event='rag/query')

)
async def rag_query_pdf_ai(ctx: inngest.Context) -> RAGSearchResult:
    def __search(question:str,top_k:int = 5):
        query_vector = embedText([question])[0]
        store = QdrantStorage()
        found = store.search(store,top_k)
        return RAGSearchResult(contexts=found['contexts'],sources = found['sources'])
    
    question = ctx.event.data

    question = ctx.event.data['question']
    top_k = int(ctx.event.data.get('top_k',5))
    found = await ctx.step.run("embed_and_search", lambda: __search(question,top_k),output_type=RAGSearchResult)


    context_block = "\n\n".join(f' - {c}' for c in found.contexts)
    user_content = [
        'Use the following context to answer the question. \n\n',
        f'Context : \n{context_block}\n\n',
        f'Question: {question}\n',
        "Answer conciesly using the context above"
    ]

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model = "gpt-4o-mini"
    )
        
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperatur": 0.3,
            "messages": [
                {'role':'system','content': "Answer:"},
                {"role": "user", "content":user_content}
            ]
        })
    answer = res["choices"][0]['message']["content"].strip()
    return {
        "answer":answer,
        "sources": found.sources,
        "num_context": len(found.contexts)
    }

@app.get('/')
def home():
    return "Hello"


inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf,rag_query_pdf_ai])



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)