import asyncio
import base64
import datetime
import os.path
import traceback
from time import time_ns

import aiofiles
import tqdm

from vector_db import delete_document, embed_document, embed_image
from publisher import notify_orb_writer, get_latest_timestamp, is_currently_embedding, update_embed_status, \
    republish_or_dlq
from utils import log, EmbedResult, FAIL, ENDC, EMBED_CONSUMER_NAME, WARNING, \
    garbage_collect, display_errors, transcribe_image, Document
from telemetry import record
from annotate import annotate_chunk
from chunk_section import make_chunks


def current_time_millis():
    return round(time_ns() / 1e-6)


async def record_telemetry(doc_id: str, doc_name: str, status: str, version: str, timestamp: float,
                           event_type, error=None, message=None, num_chunks=0, num_images=0 ):
    telemetry_payload = dict(doc_id=doc_id, doc_name=doc_name, status=status,
                             error=error, message=message, version=version, timestamp=timestamp,
                             worker=EMBED_CONSUMER_NAME, num_chunks=num_chunks,
                             num_images=num_images, event_type=event_type)
    await record(doc_id, telemetry_payload, status)


async def describe_and_embed_image(image_, image_meta, no_sections=False, prog=None):
    image_path = image_['image_path']
    image_meta['is_image'] = True
    image_meta['image_path'] = image_path
    image_meta['text_content'] = ''
    if not os.path.exists(image_path):
        log.warning(f'{WARNING}No image found at: {image_path}. Cannot describe.{ENDC}')
        if prog:
            prog.update(1)
        return None
    async with aiofiles.open(image_path, "rb") as image_file:
        image_bytes = base64.b64encode(await image_file.read()).decode('utf-8')
    response = await transcribe_image(image_bytes)
    if response.strip():
        image_meta['text_content'] = response
    image = Document(metadata=image_meta, page_content=response)
    chunk = await annotate_and_embed(image, prog, no_sections, is_image=True)
    if prog:
        prog.update(1)
    return chunk


async def act_on_document(payload):
    garbage_collect()
    event_type = payload['event_type']
    timestamp = payload['timestamp']
    doc_id = payload['doc_id']
    doc_path = payload['doc_path']
    image_data = payload['images']
    doc_name = doc_path.rsplit('/', 1)[-1]
    sections_tagged = None
    deleted_delta = None
    chunks = 0

    if event_type not in ("create", "update", "delete"):
        log.error(f"{FAIL}Bad event type: {event_type}. Skipping processing.{ENDC}")
        return EmbedResult.SKIP_AND_ACK
    if isinstance(timestamp, str):
        timestamp = float(timestamp)

    if not os.path.exists(os.path.abspath(doc_path)) and event_type != 'delete':
        log.warning(f'{WARNING}No file found at {doc_path}! Skipping.{ENDC}')
        await record_telemetry(doc_id, doc_name, 'skipped', timestamp,
                               current_time_millis(), event_type,
                               error=False,
                               message=f"No doc found at {doc_path}")
        return EmbedResult.SKIP_AND_ACK
    latest_ts = await get_latest_timestamp(doc_id)
    if latest_ts != -1 and timestamp < latest_ts:
        log.info(
            f"Stale update for {doc_id}: latest={latest_ts}, received={timestamp}. Skipping."
        )
        await record_telemetry(doc_id, doc_name, 'skipped', timestamp,
                               current_time_millis(), event_type,
                               error=False,
                               message=f"Stale update for {doc_id}: latest={latest_ts}, received={timestamp}.")
        return EmbedResult.SKIP_AND_ACK
    is_embedding = await is_currently_embedding(doc_id)
    if is_embedding:
        log.info(f'Doc {doc_id} is being worked on. Will try later.')
        await record_telemetry(doc_id, doc_name, 'queued', timestamp,
                               current_time_millis(), event_type,
                               message=f'Another worker is working on {doc_name}. Pushing back to queue.')
        return EmbedResult.REQUEUE_LATER
    await update_embed_status(doc_id, latest_ts, in_progress=True)
    await record_telemetry(doc_id, doc_name, 'started', timestamp,
                           current_time_millis(), event_type, )

    success = False
    try:
        sections_tagged, deleted_delta = await delete_document(doc_id)
        if event_type in ("create", "update"):
            sections_tagged, chunks = await embed(doc_id, doc_path, timestamp, no_sections=payload['origin'] == 'generate',
                                                  images=image_data)
        success = True

    except Exception:
        log.exception(f"{FAIL}Error handling document {doc_id}: {traceback.format_exc()}{ENDC}")

    if success:
        await record_telemetry(doc_id, doc_name, 'finished', timestamp,
                               current_time_millis(), event_type)
        tasks = [asyncio.create_task(update_embed_status(doc_id, latest_ts, in_progress=False)),
                 asyncio.create_task(notify_orb_writer({
                     "event_type": event_type,
                     "doc_id": doc_id,
                     "mapped_sections": list(set(sections_tagged)),
                     "delete_delta": deleted_delta,
                     "timestamp": datetime.datetime.now().timestamp()
                 }))
        ]
        log.debug(f'{WARNING}Sections touched: {list(set(sections_tagged))}{ENDC}')
        await asyncio.gather(*tasks)
        garbage_collect()
        return EmbedResult.SUCCESS
    await record_telemetry(doc_id, doc_name, "finished", timestamp,
                           current_time_millis(), event_type,
                           True, str(traceback.format_exc()),
                           chunks, len(image_data))

    tasks = [asyncio.create_task(republish_or_dlq(payload, traceback.format_exc())), asyncio.create_task(update_embed_status(doc_id, timestamp, in_progress=False))]

    await asyncio.gather(*tasks)
    garbage_collect()
    return EmbedResult.FAILURE


async def embed(doc_id, doc_path, version, no_sections, images):
    doc_name = doc_path.rsplit('/', 1)[-1]
    metadata = dict(doc_id=doc_id, doc_name=doc_name, version=version, is_image=False)
    collected_chunks = []
    image_meta = metadata.copy()
    image_meta['original_section'] = 'n/a'
    if images:
        prog = tqdm.tqdm(total=len(images), desc=f'Processing {len(images)} images of {doc_name}: ')
        tasks = [describe_and_embed_image(image_, image_meta.copy(), no_sections, prog) for image_ in images if image_meta]
        result = await asyncio.gather(*tasks, return_exceptions=True)
        display_errors(result)
        result = [i for i in result if i and not isinstance(i, Exception)]

        if not result:
            log.warning(f'{WARNING}Describing all images failed! Chunks will not be enriched with image context and no images will be embedded.{ENDC}')
        else:
            collected_chunks.extend(result)
    collected_chunks.extend(await process_document(doc_path, metadata, no_sections))
    distinct_sections = []
    miss = 0
    for doc in collected_chunks:
        if doc and not isinstance(doc, Exception):
            distinct_sections.extend(doc.metadata['mapped_sections'])
        else:
            miss += 1
    if miss:
        log.info(f'Dropped {miss}/{len(collected_chunks)} for document: {doc_name}')
    return set(distinct_sections), len(collected_chunks)


async def process_document(doc_path, metadata, no_sections):
    log.info(f'Processing document: {doc_path}')
    chunks = await make_chunks(doc_path, metadata)
    doc_name = doc_path.rsplit('/', 1)[-1]
    prog = tqdm.tqdm(total=len(chunks), desc=f'Annotating and Embedding {len(chunks)} chunks of {doc_name}: ')
    ret = []
    tasks = []
    for c in chunks:
        tasks.append(annotate_and_embed(c, prog, no_sections))
    if tasks:
        chunks = await asyncio.gather(*tasks)
        ret.extend(chunks)

    return ret


async def annotate_and_embed(chunk: Document, prog, no_sections, is_image=False):
    if no_sections:
        if is_image:
            return await embed_image(chunk.page_content, chunk.metadata, prog)
        else:
            return await embed_document(chunk.page_content, chunk.metadata, prog)
    annotated_chunk = await annotate_chunk(chunk)
    if annotated_chunk:
        if is_image:
            await embed_image(annotated_chunk.page_content, annotated_chunk.metadata, prog)
        else:
            await embed_document(annotated_chunk.page_content, annotated_chunk.metadata, prog)
    prog.update(1)
    return annotated_chunk
