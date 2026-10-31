import asyncio
import json
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Optional, Annotated, List, Callable, Hashable, OrderedDict, AsyncIterator

import aiohttp
import tqdm
from aiohttp import ClientResponseError, ClientTimeout
from pydantic import Field, BaseModel

from utils import config, WARNING, ENDC, log

JSON = Dict[str, "JSON"] | List["JSON"] | str | int | float | bool | None
_SENTINEL = object()

StrictStr = Annotated[str, Field(strict=True)]
StrictInt = Annotated[int, Field(strict=True)]
StrictBool = Annotated[bool, Field(strict=True)]
StrictFloat = Annotated[float, Field(strict=True)]

class DocumentPoint(BaseModel):
    """
    Mirror of the server's schema. `extra="forbid"` ensures we
    reject silently added fields instead of letting them creep in.
    """
    doc_id: StrictStr
    point_id: StrictStr
    is_image: StrictBool
    version: StrictInt | StrictStr
    text_content: StrictStr
    mapped_sections: Optional[List[StrictStr]]
    original_section: StrictStr
    doc_name: StrictStr
    score: Optional[StrictFloat] = None
    fusion_score: Optional[StrictFloat] = None
    relevance_score: Optional[StrictFloat]
    image_path: Optional[StrictStr] = None
    vector: Optional[List[StrictFloat]] = None
    queryable: StrictBool

    # ---- model configuration --------------------------------
    model_config = {
        "extra": "forbid",     # crash on unknown keys
        "frozen": False,       # hashable / immutable
        "validate_default": True,
        "str_strip_whitespace": True,
    }

@dataclass(frozen=True)
class Hit:
    point: DocumentPoint
    origin_query: str
    origin_rank: int

KeyFn = Callable[[Hit], Hashable]

class StreamProtocolError(RuntimeError):
    """Server sent malformed or out-of-contract data."""


@asynccontextmanager
async def _get(
    url: str,
    params: dict[str, str | int | bool],
    session: Optional[aiohttp.ClientSession] = None,
) -> AsyncGenerator[aiohttp.ClientResponse, None]:
    """Context-manager wrapper that owns the session only if the caller
    didn't supply one. Ensures cleanup even on Task cancellation."""
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(
            timeout=ClientTimeout(total=0),
            auto_decompress=True,
            raise_for_status=False,
            trust_env=True,
        )
    try:
        async with session.get(
            url=url,
            params=params,
            headers={"Accept-Encoding": "gzip", "Accept": "application/x-ndjson"},
        ) as resp:
            if resp.status != 200:
                # Prefer raising explicit errors instead of letting caller inspect status every time
                raise ClientResponseError(
                    status=resp.status,
                    message=f"Upstream returned {resp.status}: {await resp.text()}",
                    request_info=resp.request_info,
                    history=tuple(resp.history),
                )
            yield resp
    finally:
        if owns_session:
            await session.close()


async def query_stream(
    question: str,
    top_k: int = 10,
    return_vector: bool = False,
    limit_sections: str = "",
    collection: str = config['qdrant']['collection'],
    points_type='all',
    session: Optional[aiohttp.ClientSession] = None,
) -> AsyncGenerator[Hit, None]:
    """
    Asynchronously query the `/query_stream` endpoint and yield `DocumentPoint`
    objects as soon as each arrives.
    """
    params = {
        "question": question,
        "top_k": top_k,
        "return_vector": json.dumps(return_vector),
        "limit_sections": limit_sections,
        "collection": collection,
        'points_type': points_type
    }

    async with _get(config['generation']['queryEndpoint'], params, session) as resp:
        # Optionally inspect X-Total-Points if you want progress bars later
        total_points = int(resp.headers.get("X-Total-Points", "-1"))
        if total_points == -1:
            total_points = int(resp.headers.get("x-total-points", "-1"))
        prog = tqdm.tqdm(total=total_points, desc=f'Fetching {total_points} chunks for {question[:10]}...')
        # Use a small buffer so you're never stuck waiting for huge chunks
        # NB: readline() handles chunk boundaries for us.
        collected = 0
        yield_rank = 1
        while True:
            raw = await resp.content.readline()
            if not raw:
                break # EOF

            try:
                record: JSON = json.loads(raw)

                prog.update(1)
            except json.JSONDecodeError as e:
                raise StreamProtocolError(f"Non-JSON payload: {raw!r}") from e

            # Contract handling -------------------------------------
            if record.get("type") == "meta":
                continue # skip meta header completely
            if record.get("type") == "error":
                raise RuntimeError(record.get("error", "Unknown upstream error"))

            try:
                collected += 1
                yield Hit(point=DocumentPoint.model_validate(record),
                          origin_query=question,
                          origin_rank=yield_rank)
                yield_rank += 1
                if collected == total_points:
                    break
                # yield DocumentPoint.model_validate(record) # type: ignore[arg-type]
            except TypeError as e:
                raise StreamProtocolError(f"Bad schema in record: {record}") from e


class LRUSet:
    """
    A memory-bounded 'seen' cache with O(1) look-ups and updates.
    Evicts the oldest items when `maxlen` is exceeded.
    """
    __slots__ = ("_data", "_maxlen")

    def __init__(self, maxlen: int = 100_000) -> None:
        self._data: OrderedDict[Hashable, None] = OrderedDict()
        self._maxlen = maxlen

    def __contains__(self, key: Hashable) -> bool:            # noqa: Dunder
        return key in self._data

    def add(self, key: Hashable) -> None:
        self._data[key] = None
        self._data.move_to_end(key, last=True)
        if len(self._data) > self._maxlen:
            self._data.popitem(last=False) # evict LRU


class DedupFanIn:
    """
    Merge an arbitrary number of async iterators of DocumentPoint,
    deduplicating on the fly. Use 'async for' on the instance.
    """
    def __init__(
        self,
        *,
        key_fn: KeyFn = lambda hit: hit.point.point_id,
        max_seen: int = 100_000,
        queue_size: int = 1_000,
    ) -> None:
        self._key_fn = key_fn
        self._seen = LRUSet(max_seen)
        self._queue: asyncio.Queue[Hit] = asyncio.Queue(queue_size)
        self._tg: Optional[asyncio.TaskGroup] = None
        self._stop = asyncio.Event()
        self._active = 0

    async def __aenter__(self) -> "DedupFanIn":
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._stop.set()
        if self._tg:
            await self._tg.__aexit__(None, None, None)

    def register(self, agen: AsyncIterator[Hit]) -> None:
        """
        Register a new stream at runtime.
        """
        if not self._tg:
            raise RuntimeError("DedupFanIn must be used within 'async with'")
        self._active += 1
        self._tg.create_task(self._consume_stream(agen))

    async def _consume_stream(self, agen: AsyncIterator[Hit]) -> None:
        try:
            async for item in agen:
                k = self._key_fn(item)
                if k in self._seen:
                    continue
                self._seen.add(k)
                await self._queue.put(item)
        except Exception as e:
            log.warning(f'{WARNING}agen crash:\n{traceback.format_exc()}{ENDC}')
            await self._queue.put(e) # bubble up
        finally:
            self._active -= 1
            if self._active == 0: # last worker -> wake consumer
                await self._queue.put(_SENTINEL)

    async def _generator(self) -> AsyncGenerator[Hit, None]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL: # graceful finish
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def __aiter__(self) -> AsyncGenerator[Hit, None]:
        return self._generator()


def rrf(rank, k=60):
    return 1.0 / (k + rank + 1)

def fuse_rrf(hits: list[Hit]) -> list[DocumentPoint]:
    fused: dict[str, float] = defaultdict(float)        # point_id -> score
    for h in hits:
        fused[h.point.point_id] += rrf(h.origin_rank)

    id2doc: dict[str, DocumentPoint] = {}
    for h in hits:
        pid = h.point.point_id
        if pid not in id2doc or fused[pid] > fused[id2doc[pid].point_id]:
            id2doc[pid] = h.point

    for pid, doc in id2doc.items():
        doc.fusion_score = fused[pid]

    return sorted(id2doc.values(), key=lambda d: d.fusion_score, reverse=True)

async def get_chunks(questions: list[str], sections: List = None, top_k: int = config['retrieval']['topK'],
                     points=False, points_type='text') -> List[str]|List[DocumentPoint]:
    gathered = []
    if not sections:
        sections = ""
    else:
        sections = ",".join(sections)
    async with DedupFanIn(max_seen=200_000) as fanin:
        for q in questions:
            fanin.register(
                query_stream(
                    question=q,
                    top_k=top_k,
                    limit_sections=sections,
                    points_type=points_type
                )
            )

        # Consume deduplicated output
        async for dp in fanin:
            gathered.append(dp)
    log.debug(f'Gathered {len(gathered)} points for {len(questions)} queries.')
    fused_docs = fuse_rrf(gathered)
    if len(fused_docs) <= top_k:
        if not points:
            return [f.text_content for f in fused_docs]
        return fused_docs
    top_m = top_k * 3
    return await rerank_document_points(fused_docs, points, questions, top_k, top_m)


async def rerank_document_points(fused_docs, as_points, questions, top_k=config['retrieval']['topK'], top_m=config['retrieval']['topK'] * 3):
    candidate_docs = fused_docs[:top_m]
    candidate_texts = [d.text_content for d in candidate_docs]
    max_ce_scores = {d.point_id: float("-inf") for d in candidate_docs}
    await asyncio.gather(*[rerank_for_query(q, candidate_texts, candidate_docs, max_ce_scores)
                           for q in questions])
    for dp in candidate_docs:
        dp.relevance_score = max_ce_scores.get(dp.point_id, float("-inf"))
    ranked = sorted(candidate_docs, key=lambda d: d.relevance_score, reverse=True)
    if not as_points:
        return [r.text_content for r in ranked[:top_k]]
    return ranked[:top_k]


async def rerank_for_query(question, candidate_texts, candidate_docs, max_ce_scores):
    rerank_results = await call_reranker(question, candidate_texts)
    if rerank_results is None:
        return

    for entry in rerank_results:
        idx = entry["index"] # index into candidate_texts
        score = entry["relevance_score"]
        dp = candidate_docs[idx]
        if score > max_ce_scores[dp.point_id]:
            max_ce_scores[dp.point_id] = score


async def call_reranker(question, docs):
    body = dict(question=question, documents=docs, top_k=len(docs))
    async with aiohttp.ClientSession(headers={'Accept-Encoding': 'gzip'}) as sess:
        async with sess.post(config['generation']['rerankEndpoint'],
                             json=body, verify_ssl=config['generation'].get('verifySSL', True)) as resp:
            if resp.status != 200:
                print(f'Error! Cannot rerank! Response: {await resp.text()}')
                return None
            return (await resp.json())['data']


async def get_points_for_doc(doc_id, sections=None) -> List[DocumentPoint]:
    if not sections:
        sections = ''
    else:
        sections = ','.join(sections)
    async with aiohttp.ClientSession(headers={'Accept-Encoding': 'gzip', 'Accept': 'application/json'}) as sess:
        async with sess.get(config['generation']['pointsEndpoint'], params={'doc_id': doc_id, 'sections': sections}) as resp:
            resp.raise_for_status()
            points = (await resp.json())['data']
    return points
