import asyncio
import datetime
import json

from utils import log, redis_client, config, WARNING, ENDC

KEY_BASE = config['redis'].get('envPrefix', 'local') + ':embedding-layer:audit-logs'


async def record(doc_id, data, status):
    # if status in {'skipped', 'queued'}:
    #     await record_completed_status(doc_id, data, status)

    key = f'{KEY_BASE}:embed:telemetry:{status}'
    if status == 'started':
        existing_data = await redis_client.hget(key, doc_id)
        if existing_data:
            log.warning(
                f'{WARNING}Another embed instance of this doc {doc_id} is already in progress! Please check locks properly. Overwriting.{ENDC}')
        return await redis_client.hset(key, doc_id, json.dumps(data))
    elif status == 'finished':
        await redis_client.hdel(f'{KEY_BASE}:embed:telemetry:started', doc_id)
        return await record_completed_status(doc_id, data, status)
    elif status == 'queued':
        existing_data = await redis_client.hget(key, doc_id)
        if existing_data:
            existing_data = json.loads(existing_data)
            queued = False
            for item in existing_data:
                if data['timestamp'] == item['timestamp'] and data['version'] == item['version']:
                    queued = True
                    break
            if not queued:
                return await record_completed_status(doc_id, data, status)
            return None
        else:
            return await redis_client.hset(key, doc_id, json.dumps([data]))
    elif status == 'skipped':
        return await record_completed_status(doc_id, data, status)
    return None


async def record_completed_status(doc_id, data, status):
    key = f'{KEY_BASE}:embed:telemetry:{status}'
    existing_data = await redis_client.hget(key, doc_id)
    if not existing_data:
        existing_data = [data]
    else:
        existing_data = json.loads(existing_data)
        existing_data.append(data)
    await redis_client.hset(key, doc_id, json.dumps(existing_data))
    if status != 'queued':
        key = f'{KEY_BASE}:embed:telemetry:queued'
        queued_data = await redis_client.hget(key, doc_id)
        if queued_data:
            queued_data = json.loads(queued_data)
            updated = []
            for item in queued_data:
                if data['timestamp'] == item['timestamp'] and data['version'] == item['version']:
                    continue
                updated.append(item)
            if not updated:
                return await redis_client.hdel(key, doc_id)
            return await redis_client.hset(key, doc_id, json.dumps(updated))
        return None
    return None


async def get_all_processing_docs(as_time_series=False):
    key = f'{KEY_BASE}:embed:telemetry:started'
    keys = await redis_client.hkeys(key)
    if not keys:
        return []
    all_data = []
    for key in keys:
        doc = await redis_client.hget(key, key)
        if not doc:
            continue
        if as_time_series:
            doc['_time'] = datetime.datetime.now().timestamp() * 1000
        all_data.append(doc)
    return all_data


async def get_all_queued_docs(as_time_series=False):
    statuses = ['skipped', 'success', 'failure', 'queued']
    data = await asyncio.gather(*[get_docs_with_status(status, as_time_series)
                                  for status in statuses])

    final_data = []
    [final_data.append(j) for i in data if i for j in i]
    return final_data


async def get_docs_with_status(status, as_time_series):
    key = f'{KEY_BASE}:embed:telemetry:{status}'
    keys = await redis_client.hkeys(key)
    if not keys:
        return []
    all_data = []
    for key in keys:
        docs = await redis_client.hget(key, key)
        if not docs:
            continue
        docs = json.loads(docs)
        if as_time_series:
            for doc in docs:
                doc['_time'] = datetime.datetime.now().timestamp() * 1000
        all_data.extend(docs)
    return all_data
