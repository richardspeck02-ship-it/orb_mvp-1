import asyncio
import json
import random
import traceback

import valkey.exceptions
from valkey.exceptions import TimeoutError as ValkeyTimeoutError, ValkeyError
from embed import act_on_document
from utils import import config, log, redis_client, EmbedResult, FAIL, ENDC, WARNING, OKGREEN, MAX_CONCURRENCY, EMBED_CONSUMER_NAME

EMBED_QUEUE = config['redis']['embedQueueName']
EMBED_CONSUMER_GROUP = config['redis']['embedConsumerGroupName']

BLOCK_MS = config['redis'].get('pollTimeMs', 5000)

stop_event = asyncio.Event()
EXPECTED_KEYS = {'event_type', 'doc_path', 'delta', 'doc_id', 'timestamp', 'origin', 'images'}
RETRY_KEYS = {'traceback', 'last_retry_on', 'doc_name', 'module', 'retry_count'}


async def requeue_with_backoff(payload):
    backoff_seconds = random.uniform(1, 10)
    log.info(f"Backoff for {backoff_seconds}s before re-queueing doc_id={payload['doc_id']}")
    await asyncio.sleep(backoff_seconds)

    try:
        await redis_client.xadd(
            EMBED_QUEUE,
            fields={'payload': json.dumps(payload)},
            maxlen=10000,
            approximate=True
        )
        log.info(f"Re-queued doc_id={payload['doc_id']} after backoff")
    except Exception as e:
        log.exception(f"{FAIL}Failed to requeue after backoff: {e}{ENDC}")


async def handle_message(message_id, data):
    # await redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id)
    # return
    async with MAX_CONCURRENCY:
        try:
            data = json.loads(data['payload'])
            received_keys = set(data.keys())
            received_keys = received_keys - RETRY_KEYS
            if EXPECTED_KEYS - received_keys:
                log.error(f'{FAIL}Keys mismatch for message ID: {message_id}. Content: {data}. Skipping.{ENDC}')
                result = EmbedResult.SKIP_AND_ACK
            else:
                log.info(f'Processing event type: {data["event_type"]} from {data["origin"]}.')
                result = await act_on_document(data)

            if result == EmbedResult.SUCCESS:
                await redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id)

            elif result == EmbedResult.SKIP_AND_ACK:
                await redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id)

            elif result == EmbedResult.REQUEUE_LATER:
                await requeue_with_backoff(data)
                await redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id)
                return

            elif result == EmbedResult.FAILURE:
                log.warning(f"{WARNING}Embedding failed for {message_id}, will let retry naturally or push to DLQ.{ENDC}")
                await redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id)
                return
        except asyncio.CancelledError:
            await requeue_with_backoff(data)
            await redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id)
        except Exception:
            log.exception(f"{FAIL}Error processing {message_id}: {traceback.format_exc()}. {ENDC}")


async def reclaim_old_messages():
    """
    Scan for stale IDs (idle >= threshold), skip any already owned by another consumer,
    then XCLAIM the rest into this consumer and process them.
    """

    try:
        # 1) fetch up to N stale entries
        raw = await redis_client.xpending_range(
            EMBED_QUEUE,
            EMBED_CONSUMER_GROUP,
            min="-",
            max="+",
            count=config['redis']['reclaim']['maxMessages'],
            idle=config['redis']['reclaim']['idleThresholdMs'],
        )

        stale_ids = [i['message_id'] for i in raw]
        if not stale_ids:
            return

        log.debug(f"Found {len(stale_ids)} stale IDs to consider: {stale_ids}")

        # 2) one-shot pending lookup over the full ID-range
        # (make sure to sort IDs by timestamp portion so min/max is correct)
        sorted_ids = sorted(stale_ids, key=lambda mid: int(mid.split('-')[0]))
        low, high = sorted_ids[0], sorted_ids[-1]
        pending = await redis_client.xpending_range(
            EMBED_QUEUE,
            EMBED_CONSUMER_GROUP,
            min=low,
            max=high,
            count=len(stale_ids),
        )
        # build a map: msg_id -> owning consumer name
        owner_map = {msg_id: consumer for msg_id, consumer, *_ in pending}

        # 3) filter in-memory
        reclaimable = []
        for msg_id in stale_ids:
            owner = owner_map.get(msg_id)
            if owner and not owner.startswith('embed_'):
                log.debug(f"Skipping {msg_id} (owned by {owner})")
                continue
            reclaimable.append(msg_id)

        if not reclaimable:
            return

        log.info(f"Reclaiming {len(reclaimable)} messages: {reclaimable}")

        # 3) claim them all at once
        claimed = await redis_client.xclaim(
            EMBED_QUEUE,
            EMBED_CONSUMER_GROUP,
            EMBED_CONSUMER_NAME,
            min_idle_time=config['redis']['reclaim']['idleThresholdMs'],
            message_ids=reclaimable,
            force=True,
        )
        if not claimed:
            return

        log.debug(f"{OKGREEN}Reclaimed messages{ENDC}: {len(claimed)}")

        # 4) dedupe & dispatch
        entries = await deduplicate_messages(claimed)
        # tasks = [asyncio.create_task(handle_message(msg_id, msg_blob))
        #          for msg_id, msg_blob in entries]
        tasks = [redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id) for message_id, _ in entries]
        await asyncio.gather(*tasks)

    except Exception:
        log.error(f'{FAIL}PEL draining crashed! Error: {traceback.format_exc()}')


async def deduplicate_messages(entries):
    to_ack = []
    latest = {}
    rets = {}
    for msg_id, msg_blob in entries:
        if 'payload' not in msg_blob:
            to_ack.append(msg_id)
            continue
        payload = json.loads(msg_blob['payload'])
        if payload['doc_id'] in latest:
            if float(payload['timestamp']) > latest[payload['doc_id']]:
                rets[payload['doc_id']] = (msg_id, msg_blob)
            else:
                to_ack.append(msg_id)
        else:
            latest[payload['doc_id']] = float(payload['timestamp'])
            rets[payload['doc_id']] = (msg_id, msg_blob)

    if to_ack:
        tasks = [asyncio.create_task(redis_client.xack(EMBED_QUEUE, EMBED_CONSUMER_GROUP, message_id))
                 for message_id in to_ack]
        log.debug(f'Found {len(tasks)} messages to skip in stream. Acknowledging them.')
        await asyncio.gather(*tasks)
    return list(rets.values())


async def consume_loop():
    max_jitters = config['redis'].get('jitterTolerance', 100)
    jitters = 0
    while not stop_event.is_set():
        try:
            worked = await process_new_messages()
            if not worked:
                await reclaim_old_messages()
            jitters = 0
        except (ValkeyError, valkey.exceptions.ConnectionError, valkey.exceptions.TimeoutError) as e:
            jitters += 1
            await asyncio.sleep(3)
            if jitters > max_jitters:
                log.error(f'{FAIL}Could not establish redis connection after {max_jitters} tries. Exiting.{ENDC}')
                raise RuntimeError(f'{FAIL}Could not establish redis connection after {max_jitters} tries. Exiting.{ENDC}') from e
        except RuntimeError:
            exit(1)
        except asyncio.CancelledError:
            log.info("Consumer cancelled.")
            break
        except Exception:
            log.exception(f"{FAIL}Unexpected error in consume loop - {traceback.format_exc()}{ENDC}")
            await asyncio.sleep(3)


async def process_new_messages():
    try:
        result = await redis_client.xreadgroup(
            groupname=EMBED_CONSUMER_GROUP,
            consumername=EMBED_CONSUMER_NAME,
            streams={EMBED_QUEUE: ">"},
            count=config['redis']['batchSize'],
            block=BLOCK_MS
        )
        if result:
            _, entries = result[0]
            entries = await deduplicate_messages(entries)
            tasks = [
                asyncio.create_task(handle_message(msg_id, msg_blob))
                for msg_id, msg_blob in entries
            ]
            log.info(f'{OKGREEN}Processing latest records - {len(tasks)}{ENDC}')
            await asyncio.gather(*tasks)
            return True
    except ValkeyTimeoutError:
        pass
    return False
