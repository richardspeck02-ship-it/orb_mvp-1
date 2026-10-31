import asyncio
import json
import random
import sys
import traceback
from datetime import datetime

import aiohttp
import valkey.exceptions
from valkey.exceptions import TimeoutError as ValkeyTimeoutError, ValkeyError
from writer import writer, PdfGenerationError, NoExistingORBError
from utils import config, log, redis_client, FAIL, ENDC, WARNING, OKGREEN, MAX_CONCURRENCY, GENERATE_CONSUMER_NAME
from publisher import republish_or_dlq, is_currently_generating, update_generate_status

GENERATE_QUEUE = config['redis']['generatorQueueName']
GENERATE_CONSUMER_GROUP = config['redis']['generatorConsumerGroupName']

BLOCK_MS = config['redis'].get('pollTimeMs', 5000)

stop_event = asyncio.Event()

EXPECTED_KEYS = {'event_type', 'doc_id', 'mapped_sections', 'delete_delta', 'timestamp'}
RETRY_KEYS = {'traceback', 'last_retry_on', 'module', 'retry_count'}


async def requeue_with_backoff(payload):
    backoff_seconds = random.uniform(1, 10)
    log.info(f"Backoff for {backoff_seconds}s before re-queueing generation")
    await asyncio.sleep(backoff_seconds)

    try:
        await redis_client.xadd(
            GENERATE_QUEUE,
            fields={'payload': json.dumps(payload)},
            maxlen=10000,
            approximate=True
        )
        log.info(f"Re-queued generation for doc_id={payload.keys()} after backoff")
    except Exception as e:
        log.exception(f"{FAIL}Failed to requeue after backoff: {e}{ENDC}")


async def handle_message(message_id, data):
    data = json.loads(data['payload'])
    received_keys = set(data.keys())
    if data['event_type'] == 'create':
        log.info('Create events are no longer handled. Skipping.')
        return await acknowledge_message(message_id)
    try:
        async with aiohttp.ClientSession(conn_timeout=1) as sess:
            async with sess.get(config['generation']['healthcheckEndpoint']) as resp:
                resp.raise_for_status()
    except Exception:
        log.warning(f'{WARNING}Retrieval service is not responding. Requeue generation.{ENDC}')
        await acknowledge_message(message_id)
        return await requeue_with_backoff(data)
    async with MAX_CONCURRENCY:
        try:
            log.info(f'Processing event type: {data["event_type"]}.')
            received_keys = received_keys - RETRY_KEYS
            if EXPECTED_KEYS - received_keys:
                log.warning(f'{WARNING}Keys mismatch for message ID: {message_id}. Expected keys: {EXPECTED_KEYS} Received keys: {received_keys}. Skipping.{ENDC}')
                return await acknowledge_message(message_id)

            try:
                if await is_currently_generating():
                    log.warning(f'{WARNING}ORB generation is currently in progress. Queuing this one.{ENDC}')
                    return await requeue_with_backoff(data)
                await update_generate_status(datetime.now().timestamp(), True)
                log.debug(f'{WARNING}Received sections to write: {data["mapped_sections"]}{ENDC}')
                state = dict(section_to_write=data['mapped_sections'],
                             event=data['event_type'], doc_id=data['doc_id'],
                             delta_content=data['delete_delta'] or "",
                             timestamp=str(data['timestamp'])
                             )
                result = await writer.ainvoke(state, config={"recursion_limit": 10})
                if not result.get('out_path'):
                    log.warning(f'{WARNING}No output path returned by agent. Was ORB generated? {ENDC}')
                else:
                    log.info(f'{OKGREEN}ORB written successfully to - {result["out_path"]}{ENDC}')
            except PdfGenerationError:
                log.error(f'{FAIL}ORB was generated but PDF saving failed! Will retry naturally or move to DLQ.{ENDC}')
                await republish_or_dlq(data, str(traceback.format_exc()))
            except NoExistingORBError:
                log.error(f'{FAIL}ORB cannot be edited as no ORB exists! Write the ORB first! {ENDC}')

        except asyncio.CancelledError:
            await requeue_with_backoff(data)
        except Exception:
            log.exception(f"{FAIL}Error generating ORB for {message_id}: {traceback.format_exc()}. {ENDC}")
            await republish_or_dlq(data, str(traceback.format_exc()))
        finally:
            await update_generate_status(datetime.now().timestamp(), False)
            await acknowledge_message(message_id)
            return None


async def acknowledge_message(message_id):
    return await redis_client.xack(GENERATE_QUEUE, GENERATE_CONSUMER_GROUP, message_id)


async def reclaim_old_messages():
    """
    Scan for stale IDs (idle >= threshold), skip any already owned by another consumer,
    then XCLAIM the rest into this consumer and process them.
    """

    try:
        # 1) fetch up to N stale entries
        raw = await redis_client.xpending_range(
            GENERATE_QUEUE,
            GENERATE_CONSUMER_GROUP,
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
            GENERATE_QUEUE,
            GENERATE_CONSUMER_GROUP,
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
            if owner and not owner.startswith('generate_'):
                log.debug(f"Skipping {msg_id} (owned by {owner})")
                continue
            reclaimable.append(msg_id)

        if not reclaimable:
            return

        log.info(f"Reclaiming {len(reclaimable)} messages: {reclaimable}")

        # 3) claim them all at once
        claimed = await redis_client.xclaim(
            GENERATE_QUEUE,
            GENERATE_CONSUMER_GROUP,
            GENERATE_CONSUMER_NAME,
            min_idle_time=config['redis']['reclaim']['idleThresholdMs'],
            message_ids=reclaimable,
            force=True,
        )
        if not claimed:
            return

        log.debug(f"{OKGREEN}Reclaimed messages{ENDC}: {len(claimed)}")

        # 4) dedupe & dispatch
        entries = await deduplicate_messages(claimed)
        tasks = [asyncio.create_task(handle_message(msg_id, msg_blob))
                 for msg_id, msg_blob in entries]
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
        if 'timestamp' not in payload:
            to_ack.append(msg_id)
            continue
        if payload['doc_id'] in latest:
            if float(payload['timestamp']) > latest[payload['doc_id']]:
                rets[payload['doc_id']] = (msg_id, msg_blob)
            else:
                to_ack.append(msg_id)
        else:
            latest[payload['doc_id']] = float(payload['timestamp'])
            rets[payload['doc_id']] = (msg_id, msg_blob)

    if to_ack:
        tasks = [asyncio.create_task(redis_client.xack(GENERATE_QUEUE, GENERATE_CONSUMER_GROUP, message_id))
                 for message_id in to_ack]
        log.debug(f'Found {len(tasks)} messages to skip in stream. Acknowledging them.')
        await asyncio.gather(*tasks)
    return list(rets.values())


async def consume_loop():
    max_jitters = config['redis'].get('jitterTolerance', 10)
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
                log.error(f'{FAIL}Could not establish redis connection after {max_jitters} tries. Last Error: {traceback.format_exc()} Exiting.{ENDC}')
                sys.exit(1)
        except RuntimeError:
            sys.exit(1)
        except asyncio.CancelledError:
            log.info("Consumer cancelled.")
            break
        except Exception:
            log.exception(f"{FAIL}Unexpected error in consume loop - {traceback.format_exc()}{ENDC}")
            await asyncio.sleep(3)


async def process_new_messages():
    try:
        result = await redis_client.xreadgroup(
            groupname=GENERATE_CONSUMER_GROUP,
            consumername=GENERATE_CONSUMER_NAME,
            streams={GENERATE_QUEUE: ">"},
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
