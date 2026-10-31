import asyncio
import json
import time

from utils import config, log, redis_client, FAIL, ENDC, WARNING

GENERATOR_QUEUE = config['redis']['generatorQueueName']
GENERATOR_CONSUMER_GROUP = config['redis']['generatorConsumerGroupName']
DLQ_QUEUE = config['redis']['dlQueueName']
MAX_EMBED_RETRIES = config['redis'].get('embedRetries', 3)
DOC_LOCK_KEY = config['redis'].get('envPrefix', 'local') + ':orb-generation:generation-layer:generation-lock'
DOC_CONTENT_KEY = config['redis'].get('envPrefix', 'local') + ':orb-generation:generation-layer:contents'
VERSION_HISTORY_KEY = config['redis'].get('envPrefix', 'local') + ':orb-generation:generation-layer:revisions'
DOC_ID = 'ORB-TRANSCEND-IM'


async def store_latest_content(content_dict, out_path):
    version = await get_latest('version')
    if not version:
        version = 1
    else:
        version += 1
    latest_data = dict(version=version, content=content_dict, path=out_path)
    history = await get_history_contents()

    history[version] = {'content': content_dict, 'path': out_path, 'version': version}
    await asyncio.gather(redis_client.hset(DOC_CONTENT_KEY, DOC_ID, json.dumps(latest_data)), redis_client.hset(VERSION_HISTORY_KEY, DOC_ID,
                                                                                                                json.dumps(history)))


async def get_latest(field=None):
    data = await redis_client.hget(DOC_CONTENT_KEY, DOC_ID)
    if not data:
        return {}
    if field:
        return json.loads(data).get(field)
    return json.loads(data)


async def get_history_contents():
    data = await redis_client.hget(VERSION_HISTORY_KEY, DOC_ID)
    if not data:
        return {}
    return json.loads(data)


async def update_generate_status(timestamp, in_progress):
    payload = {
        "timestamp": str(timestamp),
        "in_progress": in_progress
    }
    if in_progress:
        payload["lock_acquired_at"] = str(time.time())
    await redis_client.hset(DOC_LOCK_KEY, DOC_ID, json.dumps(payload))


async def is_currently_generating():
    data = await redis_client.hget(DOC_LOCK_KEY, DOC_ID)
    if not data:
        return False
    parsed = json.loads(data)
    in_progress = parsed.get('in_progress', False)
    lock_time = float(parsed.get('lock_acquired_at', 0))

    if in_progress:
        now = time.time()
        if lock_time == 0 or (now - lock_time) > int(config['generation'].get('lockClearTimeoutSeconds', config['lockClearTimeoutSeconds'])):
            log.warning(f"{WARNING}Detected stale lock for doc_id={DOC_ID}. Releasing stale lock.{ENDC}")
            # clear the in-progress flag
            await update_generate_status(parsed['timestamp'], in_progress=False)
            return False
        else:
            return True
    else:
        return False


async def republish_or_dlq(payload, traceback):
    retry_count = int(payload.get('retry_count', 0))
    payload['traceback'] = str(traceback)
    payload['last_retry_on'] = time.time()
    payload['module'] = 'generate-layer'
    if retry_count >= MAX_EMBED_RETRIES:
        # Move to DLQ
        try:
            await redis_client.xadd(
                DLQ_QUEUE,
                fields={'payload': json.dumps(payload)},
                maxlen = 10000, # cap
                approximate = True
            )
            log.error(f"{FAIL}Moved generation of doc_id={payload['doc_id']} to DLQ after {retry_count} retries.{ENDC}")
        except Exception as e:
            log.exception(f"{FAIL}Failed moving generation of doc_id={payload['doc_id']} to DLQ: {e}{ENDC}")
        return

    # Republish back to main embedding queue with retry_count incremented
    payload['retry_count'] = retry_count + 1

    try:
        await redis_client.xadd(
            GENERATOR_QUEUE,
            fields={'payload': json.dumps(payload)},
            maxlen=10000, # cap
            approximate=True
        )
        log.warning(f"{WARNING}Republished generation of doc_id={payload['doc_id']} for retry #{payload['retry_count']}{ENDC}")
    except Exception as e:
        log.exception(f"{FAIL}Failed to generation of republish doc_id={payload['doc_id']} for retry: {e}{ENDC}")
