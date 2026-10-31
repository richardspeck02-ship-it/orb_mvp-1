import asyncio
import hashlib
import multiprocessing
import traceback
import uuid

from typing import Union
import qdrant_client

from qdrant_client.http import models
from qdrant_client.http.models import PayloadSchemaType
from utils import config, embeddings, late_interaction_embedding_model
from utils import sparse_embedding_model, log, EXECUTOR

client = qdrant_client.AsyncQdrantClient(url=config['qdrant']['url'], port=config['qdrant']['port'],
                                        prefer_grpc=config['qdrant']['preferGRPC'],
                                        # api_key=os.getenv(config['qdrant']['apiKey']).strip(),
                                        grpc_port=config['qdrant']['port'],
                                        https=True
                                        )

DENSE_NAME = 'Dense Model (' + config['embeddingModels']['defaultModel'] + ')'
SPARSE_NAME = 'Sparse Model (' + config['embeddingModels']['sparseModel'] + ')'
LATE_NAME = 'Late-Interaction Model (' + config['embeddingModels']['lateInteractionModel'] + ')'
VIT_NAME = 'Vision Model (' + config['embeddingModels']['visionModel'] + ')'


def deterministic_uuid(content: Union[str, bytes]) -> str:
    """
    Generates a deterministic UUID for a document chunk which serves as the point ID.
    :param content:
    :return:
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid


async def create_collection(name, overwrite=False, exist_ok=True):
    """
    Creates a collection if it does not exist or overwrites when overwrite=True.
    The created collection contains three vectors - a dense, a sparse and a late-interaction vector.
    To allow points to be searchable by their annotations/payload, also creates an index on those payload fields.
    :param name:
    :param overwrite:
    :param exist_ok:
    :return:
    """
    if await client.collection_exists(name):
        if not overwrite and not exist_ok:
            raise RuntimeError(f'Collection {name} already exists.')
        return True
    if overwrite:
        await client.delete_collection(name)
    modifier = models.Modifier.NONE if not config['embeddingModels']['useTFIDF'] else models.Modifier.IDF
    emb = await embeddings.aembed_documents(['The quick brown fox jumped over the lazy dog.'])
    # late_emb = next(late_interaction_embedding_model.embed('The quick brown fox jumped over the lazy dog.')).tolist()
    await client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_NAME: models.VectorParams(
                size=len(emb[0]),
                distance=models.Distance.COSINE,
                on_disk=True,
                datatype=models.Datatype.FLOAT16,
            ),
            LATE_NAME: models.VectorParams(
                size=config['embeddingModels']['lateInteractionDimensions'],
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
            VIT_NAME: models.VectorParams(size=config['embeddingModels']['visionModelDimensions'],
                                          distance=models.Distance.COSINE,
                                          datatype=models.Datatype.FLOAT16,
                                          multivector_config=models.MultiVectorConfig(
                                              comparator=models.MultiVectorComparator.MAX_SIM,
                                          )
                                          ),
        },
        sparse_vectors_config={
            SPARSE_NAME: models.SparseVectorParams(
                modifier=modifier,
            )
        },
        hnsw_config=models.HnswConfigDiff(on_disk=True, m=128,
                                          ef_construct=1024)
        ,
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=multiprocessing.cpu_count())
    )
    payloads = {'mapped_sections': PayloadSchemaType.KEYWORD,
                'queryable': PayloadSchemaType.BOOL, 'is_image': PayloadSchemaType.BOOL}
    await asyncio.gather(*[client.create_payload_index(
        collection_name=name,
        field_name=p,
        field_schema=payloads[p]
    )
        for p in payloads])

    log.info(f'Collection - {name} created. Payload indexes - {payloads}')
    return None


async def embed_document(document, payload: dict, prog=None):
    """
    Generates the three vectors for a document chunk and writes a point to the vector database.
    :param document:
    :param payload:
    :param prog:
    :return:
    """
    try:
        if not document.strip():
            log.warning(f'No content passed for a chunk of document: {payload["doc_id"]}.')
            if prog:
                prog.update(1)
            return
        pid = deterministic_uuid(document)
        payload['text_content'] = document
        payload['point_id'] = pid

        dense_vector, late_vector, sparse_vector = await vectorize_text(document)
        collection = config['qdrant']['collection']
        await client.upsert(collection,
                            points=[
                                models.PointStruct(
                                    id=pid,
                                    vector={
                                        DENSE_NAME:
                                            dense_vector,
                                        SPARSE_NAME: sparse_vector
                                        ,
                                        LATE_NAME: late_vector,
                                    },
                                    payload=payload,
                                )
                            ],
                            )

        if prog:
            prog.update(1)
        # await asyncio.sleep(1)
        return True
    except Exception:
        log.error(f'Failed to embed a chunk of document with ID: {payload["doc_id"]}. Error: {traceback.format_exc()}')


async def embed_image(image_desc, payload: dict, prog=None):
    """
    Generates the three vectors for a document chunk and writes a point to the vector database.
    :param image_desc:
    :param payload:
    :param prog:
    :return:
    """
    if not image_desc.strip():
        log.warning(f'Bad image received! No desc: {bool(image_desc.strip())}')
        if prog:
            prog.update(1)
        return
    pid = deterministic_uuid(image_desc)
    payload['text_content'] = image_desc
    payload['point_id'] = pid
    tasks = [asyncio.create_task(vectorize_text(image_desc)),
             ]
    feats = await asyncio.gather(*tasks)
    # image_features = feats[-1]
    dense_vector, late_vector, sparse_vector = feats[0][0], feats[0][1], feats[0][2]
    collection = config['qdrant']['collection']
    await client.upsert(collection,
                        points=[
                            models.PointStruct(
                                id=pid,
                                vector={
                                    DENSE_NAME: dense_vector
                                    ,
                                    SPARSE_NAME: sparse_vector
                                    ,
                                    LATE_NAME: late_vector,
                                    # VIT_NAME: image_features
                                },
                                payload=payload,
                            )
                        ],
                        )

    if prog:
        prog.update(1)
    # await asyncio.sleep(1)
    return True


# async def vectorize_image(image_path):
#     image = await asyncio.get_event_loop().run_in_executor(EXECUTOR, Image.open, os.path.abspath(image_path))
#     image = await asyncio.get_event_loop().run_in_executor(EXECUTOR, vision_preprocessor, image)
#     image = image.unsqueeze(0)
#
#     with torch.no_grad(), torch.autocast("cpu"):
#         image_features = await asyncio.get_event_loop().run_in_executor(EXECUTOR, vision_model.encode_image, image)
#         image_features = image_features.unsqueeze(0).flatten().tolist()
#     return image_features


async def vectorize_text(text):
    sparse_vector = await asyncio.get_event_loop().run_in_executor(EXECUTOR, sparse_embedding_model.embed, text)
    sparse_vector = list(sparse_vector)[0].as_object()
    late_vector = await asyncio.get_event_loop().run_in_executor(EXECUTOR, late_interaction_embedding_model.embed, text)
    late_vector = next(late_vector).tolist()
    dense_vector = (await embeddings.aembed_documents([text]))[0]
    return dense_vector, late_vector, sparse_vector


async def delete_document(doc_id):
    log.debug(f'Deleting all points with doc_id = {doc_id}.')
    collection = config['qdrant']['collection']
    affected_sections = []
    affected_content = []
    all_got = False
    next_pg = None
    while not all_got:
        page, next_pg = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=next_pg
        )
        all_got = not next_pg
        for point in page:
            affected_sections.extend(point.payload['mapped_sections'])
            affected_content.append(point.payload['text_content'])

    await client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    ),
                ],
            )
        ),
    )
    return set(affected_sections), '\n'.join(affected_content)


async def update_document_sections(point_id: str, sections: str) -> dict:
    sections = sections.strip().split(',')
    queryable = len(sections) > 0
    await client.set_payload(
        collection_name=config['qdrant']['collection'],
        payload={
            "mapped_sections": sections,
            "queryable": queryable,
        },
        points=[point_id],
    )
    return {'queryable': queryable, 'mapped_sections': sections}
