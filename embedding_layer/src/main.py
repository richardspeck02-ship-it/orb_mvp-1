import asyncio
import signal
import aiohttp.web

from vector_db import create_collection
from utils import log, redis_client, config, create_streams
from consumer import stop_event, consume_loop

async def handle_ping(request):
    try:
        await redis_client.ping()
    except Exception:
        raise RuntimeError('Queue unavailable! Redis is down!')
    return aiohttp.web.Response(text="pong", status=200)

async def start_health_server():
    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.get('/ping', handle_ping)])
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, '0.0.0.0', 80)
    await site.start()
    log.info("Health server started on port 80")
    return runner

async def main():
    loop = asyncio.get_running_loop()
    await create_streams()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: stop_event.set())
    health_server = await start_health_server()
    await create_collection(config['qdrant']['collection'],
                            overwrite=config['qdrant']['overwrite'],
                            exist_ok=config['qdrant']['existOk'])
    consumer_task = asyncio.create_task(consume_loop())
    await stop_event.wait()
    log.info("Shutdown initiated...")
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        log.info("Consumer task cancelled cleanly.")

    await health_server.cleanup()

    await redis_client.aclose()
    log.info("Shutdown complete.")

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
