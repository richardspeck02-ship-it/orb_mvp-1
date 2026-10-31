import asyncio
import traceback
import json
import sys
from typing import List, Optional, Dict

import aiohttp
from fastapi import FastAPI, Request, Query, HTTPException, status, APIRouter, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from loguru import logger as log

from vector_db import client, get_relevant_docs, get_points_for_doc
from utils import config
from trimmer import auto_trim

class DocumentPoint(BaseModel):
    doc_id: str
    point_id: str
    is_image: bool
    version: int
    text_content: str
    mapped_sections: Optional[List[str]]
    doc_name: str
    vector: Optional[List[float]] = None
    score: Optional[float] = None
    image_path: Optional[str] = None
    fusion_score: Optional[float] = None
    relevance_score: Optional[float] = None
    original_section: str
    queryable: bool

class QuerySuccessResponse(BaseModel):
    total_points: int
    data: List[DocumentPoint]

class QueryErrorResponse(BaseModel):
    error: str

class RerankPoint(BaseModel):
    index: int
    relevance_score: float

class RerankSuccessResponse(BaseModel):
    total_points: int
    data: List[RerankPoint]

router = APIRouter()

app = FastAPI(
    title="ORB Generation Retrieval API",
    summary="This API allows interactions with the Vector database.",
    description=f"""
This API supports batched and stream responses.
For clients supporting compression, responses greater than {config['retrieval'].get('compressionThreshold', 128)}KB are compressed with gzip.
Stream responses are always compressed if the client supports it.
""",
    version="1.0.0",
)

KILO_BYTES = 1024
# Automatically gzip any payload over threshold KB if the client supports it
app.add_middleware(GZipMiddleware, minimum_size=KILO_BYTES * config['retrieval'].get('compressionThreshold', 128))

@router.get("/healthz", status_code=200, include_in_schema=False)
async def check_qdrant():
    try:
        await client.get_collections()
    except Exception:
        log.error(
            f"Ping failed! Qdrant is not available at "
            f"{config['qdrant']['url']}:{config['qdrant'].get('port', 6379)}!"
        )
        sys.exit(1)

async def ndjson_streamer(raw_docs):
    yield json.dumps({"type": "meta", "total": len(raw_docs)}) + "\n"
    for record in raw_docs:
        try:
            dp = DocumentPoint(**record)
            yield json.dumps(jsonable_encoder(dp)) + "\n"
        except Exception:
            log.error(f"Streaming error - {traceback.format_exc()}")
            err = {"type": "error", "error": "Stream interrupted: unexpected error"}
            yield json.dumps(err) + "\n"
            break

@router.get(
    "/query",
    summary="Search for chunks relevant to a question and get them in a batch.",
    description="""
This endpoint returns the entire result in one batch.
""",
    response_model=QuerySuccessResponse,
    responses={
        400: {"model": QueryErrorResponse},
        500: {"model": QueryErrorResponse},
    },
)
async def query(
    req: Request,
    question: str = Query(..., title="Question", description="The question to search documents for."),
    top_k: int = Query(config["retrieval"]["topK"], ge=1, title="Top K", description="Maximum number of points to return."),
    return_vector: bool = Query(False, title="Return Vectors", description="Whether to include embeddings in the response."),
    limit_sections: str = Query("", title="Search Constraints", description="Comma-separated sections to limit search."),
    collection: str = Query(config["qdrant"]["collection"], title="Search Collection", description="The collection to query against."),
    points_type: str = Query(default="all", title="Points Type", description="Types of points to return.", examples=["image", "text", "all"]),
    rerank: bool = Query(default=False, title="Rerank Points", description="Rerank the retrieved points as per relevance to the question. When reranking is enabled, relevance score is returned as well.")
):
    raw_docs = await validate_and_fetch(collection, limit_sections, question, req, return_vector, top_k, rerank, points_type)
    return QuerySuccessResponse(
        total_points=len(raw_docs),
        data=[DocumentPoint(**record) for record in raw_docs],
    )

@router.get(
    "/query/stream",
    summary="Search for chunks relevant to a question and get them stream record by record.",
    description="""
This endpoint stream search results using **NDJSON** (newline-delimited JSON).
Each line is a standalone JSON object-clients should read and parse line-by-line:

1. **Metadata record** (always first):
   ```json
   {"type":"meta","total":<total_points>}
   '''
   - Total points contains the number of records that follow.
   - The total points can also be found in the header X-Total-Points.

2. **Data record** (one per document):
   ```json
   {
  "doc_id": 123,
  "point_id": "abc-456",
  "is_image": false,
  "version": 2,
  "text_content": "...",
  "mapped_sections": ["sec1", "sec2"],
  "doc_name": "DOC1",
  "vector": [0.12, -0.03, ...], // set only if return_vector=true
  "score": 0.87,
  "image_path": null, // set only if the point is an image
  "fusion_score": 0.1,
  "relevance_score": 0.1 // set only if reranking is enabled
}

3. **Error record** (if any failures happen when streaming):
   ```json
    {"type":"error","error":"Stream interrupted: unexpected error"}
    ```
    - Clients must stop reading on receiving this.
""",
response_model=DocumentPoint,
responses={
400: {"model": QueryErrorResponse},
500: {"model": QueryErrorResponse},
},
)
async def query_stream(
    req: Request,
    question: str = Query(..., title="Question", description="The question to search documents for."),
    top_k: int = Query(config["retrieval"]["topK"], ge=1, title="Top K",
                       description="Maximum number of points to return."),
    return_vector: bool = Query(False, title="Return Vectors",
                                description="Whether to include embeddings in the response."),
    limit_sections: str = Query("", title="Search Constraints",
                                description="Comma-separated sections to limit search."),
    collection: str = Query(config["qdrant"]["collection"], title="Search Collection",
                            description="The collection to query against."),
    rerank: bool = Query(default=False, title="Rerank Points", description="Rerank the retrieved points as per relevance to the question. When reranking is enabled, relevance score is returned as well."),
    points_type: str = Query(default="all", title="Points Type", description="Types of points to return.", examples=["image", "text", "all"]),
):
    raw_docs = await validate_and_fetch(collection, limit_sections, question, req, return_vector, top_k, rerank, points_type)

    return StreamingResponse(
        ndjson_streamer(raw_docs),
        media_type="application/x-ndjson",
        headers={"X-Total-Points": str(len(raw_docs))},
        status_code=status.HTTP_200_OK,
    )

@router.get(
    "/query/id",
    summary="Get all points for a given document ID.",
    response_model=QuerySuccessResponse,
    responses={
        400: {"model": QueryErrorResponse},
        500: {"model": QueryErrorResponse},
    },
)
async def query_by_id(
    doc_id: str = Query(default=..., description="The document ID for which to fetch points."),
    sections: str = Query(default="", description="Limit points to only the given sections."),
    collection: str = Query(config["qdrant"]["collection"], title="Search Collection",
                            description="The collection to query against."),
):
    if sections.strip():
        sections = sections.split(',')
    else:
        sections = None
    docs = await get_points_for_doc(collection, doc_id, sections)
    return QuerySuccessResponse(
        total_points=len(docs),
        data=[DocumentPoint(**record) for record in docs],
    )

@router.post(
    "/rerank",
    summary=f"Rerank a list of documents in order of relevance to a given question. Uses {config['retrieval']['rerankModel']} reranker model.",
    response_model=RerankSuccessResponse,
    responses={
        400: {"model": QueryErrorResponse},
        500: {"model": QueryErrorResponse},
    },
)
async def rerank_(
    documents: List[str] = Body(default=..., description="List of documents to rerank."),
    question: str = Body(default=..., description="The question to rank the documents for."),
    top_k: int = Body(default=..., description="Maximum number of points to return.", ge=1)
):
    if not question.strip():
        raise HTTPException(status_code=400, detail='Question is empty!')
    if not documents or not [i for i in documents if i]:
        raise HTTPException(status_code=400, detail='No docs to rerank!')
    docs = await asyncio.gather(*[auto_trim(question, d) for d in documents])
    res = await call_reranker(question, docs, top_k)
    res = res['results']
    return RerankSuccessResponse(
        total_points=len(res),
        data=[RerankPoint(**record) for record in res],
    )

async def validate_and_fetch(collection, limit_sections, question, req, return_vectors, top_k, rerank, points_type):
    errors: List[str] = []
    if not points_type.strip() in ('image', 'text', 'all'):
        errors.append('Invalid points type! Must be either image, text or all.')
    if not question.strip():
        errors.append("Please pass a question!")
    if not collection.strip():
        log.warning("No collection was passed. Using default collection.")
        collection = config["qdrant"]["collection"]
    if not await client.collection_exists(collection):
        errors.append(f"Collection '{collection}' does not exist.")
    if errors:
        detail = "\n".join(["Following errors were found in the request:"] + errors)
        raise HTTPException(status_code=400, detail=detail)
    if rerank:
        top_k = top_k * config['retrieval']['scaleFactor']
    sections = [s.strip() for s in limit_sections.split(",") if s.strip()]
    try:
        raw_docs = await get_relevant_docs(
            question,
            sections,
            collection,
            top_k,
            return_vectors,
            points_type
        )
        if rerank:
            top_k = int(top_k / config['retrieval']['scaleFactor'])
            raw_docs = await rerank_points(question, raw_docs, top_k)
    except Exception:
        log.error(f"Request params: {req.query_params}")
        log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Sorry! An unexpected error occurred! Please try again later."
        )
    return [i for i in raw_docs if 'text_content' in i and 'doc_id' in i]

async def rerank_points(question, points: List[Dict], top_k):
    try:
        docs = [d['text_content'] for d in points]
        docs = await asyncio.gather(*[auto_trim(question, d) for d in docs])
        resp = await call_reranker(question, docs, top_k)
        if not resp:
            return points[:top_k]
        idx = [i['index'] for i in resp['results']]
        res = []
        for i, index in enumerate(idx):
            points[index]['relevance_score'] = resp['results'][i]['relevance_score']
            res.append(points[index])
        return res
    except Exception:
        log.error(f'Failed to rerank! Error: {traceback.format_exc()}')
        return points[:top_k]

async def call_reranker(question, docs, top_k):
    params = dict(query=question, documents=docs, return_documents=False,
                  raw_scores=False, top_n=top_k,
                  model=config['retrieval']['rerankModel'])
    async with aiohttp.ClientSession(headers={'Accept-Encoding': 'gzip'}) as sess:
        async with sess.post(config['retrieval']['rerankEndpoint'],
                             json=params, verify_ssl=config['retrieval'].get('verifySSL', True)) as resp:
            if resp.status != 200:
                print(f'Error! Cannot rerank! Response: {await resp.text()}')
                return None
            return await resp.json()

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config["uvicorn"]["host"],
        port=config["uvicorn"]["port"],
    )
