"""
A service to reclaim unacknowledged messages in the stream and assign them to available workers in a
round-robin fashion. A greedy round-robin is utilized where amount of messages passed to a worker is
inversely proportional to the size of its pending queue.
Useful for scenarios which have frequent deltas causing an overflow of messages to the stream and overloading
the workers, leading to messages remaining unacknowledged.
IMPORTANT - If you have just one worker running in a high-stress environment, reclaimer will just end up burdening
that worker
"""
import asyncio
import heapq
import logging
from collections import defaultdict
from valkey.asyncio import Valkey as Redis
from valkey.exceptions import ValkeyError as RedisError

from utils import config, create_streams

log = logging.getLogger("reclaimer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

redis = Redis(host=config['redis']['host'], port=config['redis']['port'], decode_responses=True)

async def _live_consumers(key: str, group: str) -> list[tuple[int, str]]:
    """
    Returns a list [(pending, name), ...] for every consumer in the group,
    ordered by current pending count (ascending).
    """
    try:
        info = await redis.xinfo_consumers(key, group)
    except RedisError:
        log.exception("cannot fetch consumer info")
        return []

    return sorted([(c["pending"], c["name"]) for c in info], key=lambda x: x[0])

async def _stale_ids(key: str, group: str) -> list[str]:
    """
    Returns up to BATCH_COUNT message IDs whose idle time >= IDLE_THRESHOLD.
    """
    raw = await redis.xpending_range(
        key, group, min="-", max="+", count=config['redis']['reclaim']['maxMessages'], idle=config['redis']['reclaim']['idleThresholdMs']
    )
    # raw entries are (id, consumer, idle, deliveries)
    return [eid for eid, *_ in raw]


async def smart_reassign(key, group):
    consumers = await _live_consumers(key, group)
    if not consumers:
        log.warning(f"no live consumers in group {group} - skip reclaim")
        return
    if len(consumers) <= 1 :
        log.warning("Just one consumer is running and it is busy. Skipping reclaim this round.")

    stale = await _stale_ids(key, group)
    if not stale:
        return
    log.info("reclaiming %d stale IDs across %d consumers", len(stale), len(consumers))

    heap = [(p, n) for p, n in consumers]
    heapq.heapify(heap)

    assignments: dict[str, list[str]] = defaultdict(list)

    # greedy round-robin assignment
    for mid in stale:
        pending, name = heapq.heappop(heap)
        assignments[name].append(mid)
        heapq.heappush(heap, (pending + 1, name))

    claimed_total = 0
    for name, mids in assignments.items():
        try:
            claimed = await redis.xclaim(
                key,
                group,
                name,
                min_idle_time=config['redis']['reclaim']['idleThresholdMs'],
                message_ids=mids,
                force=True,
            )
            claimed_total += len(claimed)
            log.info(" -> %s got %d msgs (now pending=%d)",
                     name, len(claimed), len(claimed) + next(p for p, n in heap if n == name))
        except Exception:
            log.exception("xclaim failed for consumer %s", name)

    log.info("reclaimed %d IDs total", claimed_total)


async def reclaimer_loop():
    streams = ((config['redis']['embedQueueName'], config['redis']['embedConsumerGroupName']),
               (config['redis']['generatorQueueName'], config['redis']['generatorConsumerGroupName']))
    await create_streams()
    while True:
        try:
            await asyncio.gather(*[asyncio.create_task(smart_reassign(s[0], s[1]))
                                   for s in streams])
        except Exception:
            log.exception("unexpected error in reclaimer loop")
        await asyncio.sleep(config['redis']['reclaim']['intervalSeconds'])


if __name__ == "__main__":
    asyncio.run(reclaimer_loop())