import asyncio
import math

import qdrant_client
from qdrant_client.http import models
from utils import config, embeddings, late_interaction_embedding_model
from utils import sparse_embedding_model

client = qdrant_client.AsyncQdrantClient(url=config['qdrant']['url'], port=config['qdrant']['port'],
                                         prefer_grpc=config['qdrant']['preferGRPC'],
                                         # api_key=os.getenv(config['qdrant']['apiKey']).strip(),
                                         grpc_port=config['qdrant']['port'],
                                         https=True, timeout=config['qdrant'].get('timeout', 30)
                                         )

DENSE_NAME = 'Dense Model (' + config['embeddingModels']['defaultModel'] + ')'
SPARSE_NAME = 'Sparse Model (' + config['embeddingModels']['sparseModel'] + ')'
LATE_NAME = 'Late-Interaction Model (' + config['embeddingModels']['lateInteractionModel'] + ')'

async def get_relevant_docs(query, sections=None,
                            collection=config['qdrant']['collection'], top_k=config['retrieval']['topK'],
                            return_vectors=False, points_type='all'):
    dense_vector = embeddings.aembed_query(query)
    sparse_vector = asyncio.to_thread(sparse_embedding_model.query_embed, query)
    late_vector = asyncio.to_thread(late_interaction_embedding_model.query_embed, query)
    vectors = await asyncio.gather(dense_vector, sparse_vector, late_vector)
    dense_vector = vectors[0]
    sparse_vector = vectors[1]
    late_vector = vectors[2]
    sparse_vector = next(sparse_vector).as_object()
    n_chunks = await client.count(collection, exact=False)
    if n_chunks.count < 1:
        mult = 0
    else:
        mult = math.log(n_chunks.count)
    scale = max(3, int(config['retrieval']['scaleFactor'] * mult))
    prefetch_k = top_k * scale

    late_vector = next(late_vector).tolist()

    filters = []
    if points_type != 'all':
        filters.append(models.FieldCondition(
            key="is_image",
            match=models.MatchValue(value=points_type == 'image'),
        ))
    if sections:
        filters.append(models.FieldCondition(
            key="mapped_sections",
            match=models.MatchAny(any=sections),
        ))
    query_filter = models.Filter(must_not=models.FieldCondition(
        key="queryable",
        match=models.MatchValue(value=False),
    ))
    if filters:
        query_filter.must = models.Filter(must=filters)
    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using=DENSE_NAME,
            limit=prefetch_k,
            filter=query_filter,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vector),
            using=SPARSE_NAME,
            limit=prefetch_k,
            filter=query_filter,
        ),
    ]
    return await perform_fetch(prefetch, query_filter, collection, late_vector, top_k,
                               return_vectors)

def rrf(rank): return 1 / (60 + rank)


async def perform_fetch(prefetch, query_filter, collection, late_vectors, top_k, return_vectors):
    results = await client.query_points(
        collection,
        prefetch=prefetch,
        query=late_vectors,
        with_payload=True,
        using=LATE_NAME,
        limit=top_k * config['retrieval']['scaleFactor'],
        query_filter=query_filter,
        with_vectors=True
    )
    fused = {}
    for rank, point in enumerate(results.points, start=1):
        key = point.id
        fused.setdefault(key, 0.)
        fused[key] += rrf(rank)

    top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
    id2point = {p.id: p for p in results.points}
    data = []
    for pid in top_ids:
        p = id2point[pid]
        payload = p.payload
        if return_vectors:
            payload['vector'] = p.vector.get(DENSE_NAME, [])
        payload['fusion_score'] = fused[pid]
        payload['score'] = p.score
        data.append(payload)

    return data


async def get_points_for_doc(collection, doc_id, sections):
    all_got = False
    next_pg = None
    filters = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    if sections:
        filters.append(models.FieldCondition(
            key="mapped_sections",
            match=models.MatchAny(any=sections),
        ))
    data = []
    while not all_got:
        page, next_pg = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=filters
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=next_pg
        )
        all_got = not next_pg
        for point in page:
            data.append(point.payload)
    return data
