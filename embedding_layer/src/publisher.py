import json
import time

from utils import config, log, redis_client, FAIL, ENDC, WARNING, OKCYAN

EMBED_QUEUE = config['redis']['embedQueueName']

GENERATOR_QUEUE = config['redis']['generatorQueueName']
GENERATOR_CONSUMER_GROUP = config['redis']['generatorConsumerGroupName']
DLQ_QUEUE = config['redis']['dlQueueName']
MAX_EMBED_RETRIES = config['redis'].get('embedRetries', 3)
DOC_VERSION_KEY = config['redis'].get('envPrefix', 'local') + ':orb-generation:embedding-layer:version-control'


async def get_latest_timestamp(doc_id):
    data = await redis_client.hget(DOC_VERSION_KEY, doc_id)
    if not data:
        return -1
    return float(json.loads(data)['timestamp'])


async def update_embed_status(doc_id, timestamp, in_progress):
    payload = {
        "timestamp": str(timestamp),
        "in_progress": in_progress
    }
    if in_progress:
        payload["lock_acquired_at"] = str(time.time())
    await redis_client.hset(DOC_VERSION_KEY, doc_id, json.dumps(payload))


async def is_currently_embedding(doc_id):
    data = await redis_client.hget(DOC_VERSION_KEY, doc_id)
    if not data:
        return False
    parsed = json.loads(data)
    in_progress = parsed.get('in_progress', False)
    lock_time = float(parsed.get('lock_acquired_at', 0))

    if in_progress:
        now = time.time()
        if lock_time == 0 or (now - lock_time) > int(config['lockClearTimeoutSeconds']):
            log.warning(f"{WARNING}Detected stale lock for doc_id={doc_id}. Releasing stale lock.{ENDC}")
            # clear the in-progress flag
            await update_embed_status(doc_id, parsed['timestamp'], in_progress=False)
            return False
        else:
            return True
    else:
        return False


async def notify_orb_writer(payload):
    try:
        message_id = await redis_client.xadd(
            GENERATOR_QUEUE,
            fields=dict(payload=json.dumps(payload)),
            maxlen=10000,  # cap
            approximate=True
        )
        log.info(f"{OKCYAN}Notification sent for doc_id={payload['doc_id']}, message_id={message_id}{ENDC}")
        return message_id
    except Exception as e:
        log.exception(f"{FAIL}Failed to notify ORB writer for doc_id={payload['doc_id']}: {e}{ENDC}")
        raise


async def republish_or_dlq(payload, traceback):
    retry_count = int(payload.get('retry_count', 0))
    payload['traceback'] = str(traceback)
    payload['last_retry_on'] = time.time()
    payload['doc_name'] = payload['doc_path'].rsplit('/', 1)[-1]
    payload['module'] = 'embed-layer'
    if retry_count >= MAX_EMBED_RETRIES:
        # Move to DLQ
        try:
            await redis_client.xadd(
                DLQ_QUEUE,
                fields={'payload': json.dumps(payload)},
                maxlen = 10000,  # cap
                approximate = True
            )
            log.error(f"{FAIL}Moved doc_id={payload.get('doc_id')} to DLQ after {retry_count} retries.{ENDC}")
        except Exception as e:
            log.exception(f"{FAIL}Failed moving doc_id={payload.get('doc_id')} to DLQ: {e}{ENDC}")
        return

    # Republish back to main embedding queue with retry_count incremented
    payload['retry_count'] = retry_count + 1

    try:
        await redis_client.xadd(
            EMBED_QUEUE,
            fields={'payload' : json.dumps(payload)},
            maxlen=10000,  # cap
            approximate=True
        )
        log.warning(f"{WARNING}Republished doc_id={payload.get('doc_id')} for retry #{payload['retry_count']}{ENDC}")
    except Exception as e:
        log.exception(f"{FAIL}Failed to republish doc_id={payload.get('doc_id')} for retry: {e}{ENDC}")
