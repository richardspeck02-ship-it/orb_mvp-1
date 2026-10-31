import asyncio
import json
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Process
from typing import List

from publisher import get_history_contents, VERSION_HISTORY_KEY
from fastapi import APIRouter, Body, FastAPI, HTTPException, status
from loguru import logger as log
from pydantic import BaseModel

from publisher import DOC_CONTENT_KEY, DOC_ID, is_currently_generating, update_generate_status
from utils import (
    config,
    create_streams,
    redis_client,
    WARNING,
    ENDC,
)
from writer import writer
from consumer import consume_loop

# ----------------------------------------------------------------------
# Pydantic models for response schemas
# ----------------------------------------------------------------------
class GenerationStatus(BaseModel):
    status: str
    sections: List[str]


class ErrorResponse(BaseModel):
    error: str


# ----------------------------------------------------------------------
# FastAPI setup
# ----------------------------------------------------------------------
router = APIRouter()


@router.post(
    "/reset",
    summary="Reset the ORB content and version. Does not generate a new ORB.",
)
async def reset_orb_version():
    # If a generation is in progress, return 409
    if await is_currently_generating():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A generation is in progress. Please try later.",
        )

    # Reset the Redis hash
    await redis_client.hset(
        DOC_CONTENT_KEY, DOC_ID, json.dumps({})
    )
    await redis_client.hset(VERSION_HISTORY_KEY, DOC_ID, json.dumps({}))
    log.info("ORB version reset successful.")
    return "ok"


@router.get("/healthz", status_code=200, include_in_schema=False)
async def check_qdrant():
    try:
        await redis_client.ping()
    except Exception:
        log.error(
            f"Ping failed! Redis is not available at "
            f"{config['redis']['host']}:{config['redis'].get('port', 6379)}!"
        )
        sys.exit(1)
    return {"status": "ok"}


@router.get("/history", status_code=200, include_in_schema=True,
            summary="Get all revision history of generated ORBs",
)
async def get_history():
    return await get_history_contents()


@router.post(
    "/write",
    summary="Write a new version of the ORB. Replaces existing versions.",
    response_model=GenerationStatus,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def write(
    sections: List[str] = Body(
        [],
        title="Sections",
        description="The ORB sections or sub-sections to write. "
        "These must exist in the config.",
    ),
):
    # If a generation is already running, conflict
    if await is_currently_generating():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A generation is in progress. Please try later.",
        )

    await update_generate_status(datetime.now().timestamp(), True)

    # Trim whitespace and filter out empty names
    sections = [i for i in sections if i.strip()]
    if sections:
        log.debug(f"Limiting new ORB to {sections} sections.")
    else:
        # Default to all sections and subSections if none provided
        sections = []
        for sec in config["generateSections"]:
            sections.append(sec["name"])
            for sub in sec.get("subSections", []):
                sections.append(sub["name"])

    state = {
        "section_to_write": sections,
        "event": "api-create",
        "doc_id": "NA",
        "delta_content": {},
        "timestamp": str(datetime.now().timestamp()),
    }
    result = await writer.ainvoke(state, config={"recursion_limit": 10})
    await update_generate_status(datetime.now().timestamp(), False)
    if not result.get("out_path"):
        log.warning(
            f"{WARNING}No output path returned by agent. Was ORB generated?{ENDC}"
        )
        return GenerationStatus(status="Writing failed.", sections=sections)

    return GenerationStatus(
        status=f"Document saved to {result['out_path']}", sections=sections
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI startup: application is initializing.")
    await update_generate_status(datetime.now().timestamp(), False)
    yield

    log.info("FastAPI shutdown: closing Redis connection.")
    try:
        await update_generate_status(datetime.now().timestamp(), False)
        await redis_client.aclose()
    except Exception as e:
        log.error(f"Error closing Redis client: {e}")


async def main():
    await create_streams()

    consume_task = asyncio.create_task(consume_loop())

    shutdown_event = asyncio.Event()

    def _signal_handler():
        log.info("Received termination signal. Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    config_uvicorn = {
        "app": app,
        "host": config["uvicorn"]["host"],
        "port": config["uvicorn"]["port"],
        "log_level": "info",
    }
    server = None
    shutdown_wait_task = None
    try:
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(**config_uvicorn))
        server_task = asyncio.create_task(server.serve())
        log.info("Uvicorn server started.")

        shutdown_wait_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [shutdown_wait_task, consume_task, server_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if consume_task in done:
            exc = consume_task.exception()
            if exc:
                log.error(f"consume_loop() terminated with exception: {exc}")
                raise exc
            else:
                log.info("consume_loop() finished normally.")

        if server_task in done:
            exc = server_task.exception()
            if exc:
                log.error(f"Uvicorn terminated with exception: {exc}")
                raise exc
            else:
                log.info("Uvicorn server stopped normally.")

        log.info("Shutdown event triggered. Beginning graceful shutdown.")

    finally:
        if server and not server.should_exit:
            server.should_exit = True
            log.info("Signaled Uvicorn to shut down.")

        if not consume_task.done():
            consume_task.cancel()
            try:
                await consume_task
            except asyncio.CancelledError:
                log.info("consume_loop() was cancelled successfully.")

        if shutdown_wait_task and not shutdown_wait_task.done():
            shutdown_wait_task.cancel()
            try:
                await shutdown_wait_task
            except asyncio.CancelledError:
                pass

        try:
            await update_generate_status(datetime.now().timestamp(), False)
            await redis_client.aclose()
        except Exception:
            pass

        log.info("Application shutdown complete.")


app = FastAPI(
    title="ORB Generation Writer API",
    summary="This API allows interactions with the ORB writer agent suite and instructs it to write or update the ORB.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

if __name__ == "__main__":
    asyncio.run(main())
