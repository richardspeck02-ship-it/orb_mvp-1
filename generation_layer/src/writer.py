import asyncio
import copy
import gc
import os
import pathlib
import random
import re
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Dict, List, Sequence, Iterable, Set, Optional, Tuple

import aiohttp.client_exceptions
from rapidfuzz import process, fuzz
import openai
import pypandoc
import unicodedata
from langgraph.constants import END
from langgraph.graph import StateGraph
import aiofiles

from prompts import query_prompt, section_write_prompt, section_update_prompt, section_review_prompt, summary_prompt,\
    image_insert_prompt
from utils import config, call_llm, call_llm_with_structured, OKCYAN, WARNING, FAIL, ENDC, log, OKGREEN, \
    get_sorted_sections, \
    CostCounter
from query import get_chunks, get_points_for_doc, rerank_document_points
from publisher import store_latest_content, get_latest

WRITER_MODEL = config['llmModel'].get('writerModel', config['llmModel']['defaultModel'])
META_MODEL = config['llmModel'].get('metaModel', config['llmModel']['defaultModel'])
IMAGE_PATTERN = re.compile(r"<IMAGE\d+>")


class PdfGenerationError(Exception):
    """Raised when a PDF generation fails."""


class NoExistingORBError(Exception):
    """Raised when an update or delete operation is requested but no ORB was found to edit."""


class State(TypedDict):
    generated_sections: Dict
    section_to_write: List
    delta_content: str | Dict
    event: str
    doc_id: str
    version: str
    out_path: str
    route_to: str
    timestamp: str
    generation_summary: str
    cost: CostCounter


async def generate_search_terms(section, chunks, state, extra=None):
    prompt = query_prompt.format(chunks=chunks, section=section, extra=extra)
    queries = await call_llm_with_structured(prompt, META_MODEL, 0.05, cost_counter=state['cost'],
                                             retries=config['generation'].get('generationRetries'),
                                             timeout=config['generation'].get('generationTimeout'))
    log.debug(f"Queries for section: {section['name']} -> {queries}")
    return queries['queries']


async def get_chunks_with_retry(search_terms, section=None, points=False, points_type='text'):
    if points_type == 'image':
        top_k = config['generation']['maxImages']
    else:
        top_k = config['retrieval']['topK']

    total = config['generation']['queryRetries']
    tries = 0
    while tries < total:
        try:
            tries += 1
            return await get_chunks(search_terms, section, points=points, points_type=points_type, top_k=top_k)
        except aiohttp.client_exceptions.ClientResponseError:
            backoff_seconds = random.uniform(3, config['generation']['maxRetrySeconds'])
            log.warning(f"{WARNING}Retrieval API failed. Retrying after {backoff_seconds}seconds.{ENDC}")
            await asyncio.sleep(backoff_seconds)
    return []


async def summarize_content_so_far(state: State):
    prompt = summary_prompt.format(template=config['generateSections'],
                                   generated=state['generated_sections'])
    state['generation_summary'] = await call_llm_safe(prompt, WRITER_MODEL, state, 0.1)
    return state


async def retrieve_points(search_terms, section, get_as_dp=False, points_type='text'):
    points = await get_chunks_with_retry(search_terms, [section], get_as_dp, points_type=points_type)
    if not points:
        log.warning(f"{WARNING}Did not get any points for search terms: {search_terms}. Doing naive query.{ENDC}")
        points = await get_chunks_with_retry(search_terms, points=get_as_dp, points_type=points_type)
    log.info(f"Retrieved {len(points)} chunks.")
    return points


async def write_section(section_chunks, section_template, state: State, extra_instructions=None, parent=None, so_far="",
                        prompt=None):
    if not prompt:
        prompt = section_write_prompt.format(chunks=section_chunks, template=section_template, extra=extra_instructions,
                                             parent=parent, generated=so_far)
    content = await call_llm_safe(prompt, WRITER_MODEL, state, 0.05)
    log.debug(f"Generated {len(content)} characters for section: {section_template['name']}")
    return content


async def call_llm_safe(prompt, model, state: State, temperature, fallback=' This section could not be generated due to an error.'):
    try:
        content = await call_llm(prompt, model, temperature, cost_counter=state['cost'],
                                  retries=config['generation'].get('generationRetries'),
                                  timeout=config['generation'].get('generationTimeout'))
    except openai.BadRequestError as err:
        log.error(f"{FAIL}Prompt failed! Error: {err.message}. \nPrompt: {prompt}")
        content = fallback
    return content


async def update_section(section_chunks, section_template, event, current, state: State, extra=None, parent=None):
    prompt = section_update_prompt.format(chunks=section_chunks, template=section_template, event=event.upper(),
                                          extra=extra, section=current, parent=parent)
    content = await call_llm_safe(prompt, META_MODEL, state, 0.05)
    log.debug(f"Updated {len(content)} characters for section: {section_template['name']}")
    return content


async def review_section(section_template, content, chunks, state: State, parent=None, extra=None):
    prompt = section_review_prompt.format(template=section_template, content=content,
                                          chunks=chunks, parent=parent, extra=extra)
    try:
        data = await call_llm_with_structured(prompt, META_MODEL, 0.05, cost_counter=state['cost'],
                                              retries=config['generation'].get('generationRetries'),
                                              timeout=config['generation'].get('generationTimeout'))
    except openai.BadRequestError as e:
        log.error(f"{FAIL}Review failed! Error: {e} Prompt: {prompt}")
        return None
    if data['needs_change']:
        log.info(f"Flagged section {section_template['name']} for update.")
    return data


async def generate_section_update(section_template, state: State):
    log.info(f"{OKCYAN}Updating: {section_template['name']}{ENDC}")
    parent = None
    parent_content = None
    current_content = state['generated_sections'].get(section_template['name'])
    if section_template['name'] not in state['generated_sections']:
        for section in state['generated_sections']:
            if section_template['name'] in state['generated_sections'][section]['subsections']:
                parent = section
                parent_content = state['generated_sections'][section]['content']
                for sect in state['generated_sections'][section]['subsections']:
                    if sect['name'] == section_template['name']:
                        current_content = state['generated_sections'][section]['subsections'][sect['name']]['content']
    delta = await get_chunks_with_retry(section_template['name']) if state['event'] == 'update' else state['delta_content']
    content = await update_section(delta, section_template, state['event'], current_content, state, parent=parent)
    times = config['generation'].get('reviews', 0)
    tried = 0
    parent = {'name': parent}
    review_extra = f"""The main document had a {state['event']} event so the section is being updated as per the delta given in the chunks. The original content of this section was {current_content}.
    *For delete event, existing content must be removed if it suits the deleted parts and for update event, existing content must be updated if new information is in delta or existing info gets updated.*
    *The section can also remain untouched if the delta is relevant to it.*"""

    while tried < times:
        review = await review_section(section_template, content, delta, state, extra=review_extra)
        tried += 1
        if not review:
            content = content + f"\n\n> Human evaluation recommended.\n"
            state = await store_generated_section(content, parent, section_template, state)
            return state
        if review['needs_change']:
            extra = f"""You already wrote the update as: {content}. Now, you must try to address this feedback if possible : {review['feedback']} and update the section."""
            content = await update_section(delta, section_template, state['event'], content, state, extra, parent_content)
            review_extra = f"""The main document had a {state['event']} event so the section is being updated as per the delta given in the chunks. The original content of this section was {content}.
            *For delete event, existing content must be removed if it suits the deleted parts and for update event, existing content must be updated if new information is in delta or existing info gets updated.*
            *The section can also remain untouched if the delta is relevant to it.*"""
        else:
            state = await store_generated_section(content, parent, section_template, state)
            return state
    if tried >= times:
        if review['needs_change']:
            content = content + f"\n\n> Human evaluation recommended.\n"
            state = await store_generated_section(content, parent, section_template, state)
            return state
    state = await store_generated_section(content, parent, section_template, state)
    return state


async def clean_caption(raw: str, state: State, max_words: int = 20) -> str:
    """
    Turn a potentially multi-line, markdown-rich string into a short plain caption.
    """
    # 1) Remove any markdown links: [text](url) -> text
    prompt = f"""Make a caption of this image from its description within {max_words} words. - <IMAGE DESCRIPTION - {raw}>"""
    collapsed = await call_llm_safe(prompt, META_MODEL, state, 0.1, '')
    if collapsed.strip():
        return collapsed
    no_links = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw)
    collapsed = re.sub(r'\s+', ' ', no_links).strip()
    words = collapsed.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return collapsed


async def generate_section(section_template, state: State, parent=None, parent_content=None):
    log.info(f"{OKGREEN}Writing: {section_template['name']}{ENDC}")
    d_points = await get_chunks_with_retry([f"{section_template['name']}\n{section_template['description']}"],
                                           [section_template['name']], True)
    i_points = await get_image_points(section_template)
    chunks = [d.text_content for d in d_points]
    search = await generate_search_terms(section_template, chunks, state, parent)
    if search:
        if isinstance(search, dict):
            temp = search.copy()
            for key in temp:
                search = temp[key]

        section_points = await retrieve_points(search, section_template['name'], get_as_dp=True)
        section_points.extend(d_points)
        section_points = await rerank_document_points(section_points, as_points=True, questions=search)
    else:
        section_points = chunks
    content = await write_section(section_points, section_template, state, None, parent_content,
                                  state.get('generation_summary', ''))

    # state = await summarize_content_so_far(state)
    times = config['generation'].get('reviews', 0)
    tried = 0
    while tried < times:
        tried += 1
        log.info(f"{WARNING}Review {tried} of {times} for - {section_template['name']}{ENDC}")
        review = await review_section(section_template, content, section_points, state, parent_content)
        if not review:
            if i_points:
                content = await insert_images(content, i_points, parent_content, section_points, section_template, state)
            content = content + f"\n\n> Human evaluation recommended.\n"
            state = await store_generated_section(content, parent, section_template, state)
            return state

        if review['needs_change']:
            extra = f"""You already wrote: {content}. Now, the queries you write must try to address this feedback - {review['feedback']}"""
            search = await generate_search_terms(section_template, section_points, state, extra)
            section_points = await retrieve_points(search, section_template['name'])
            extra = f"""You already wrote: {content}. Now, you must try to update your content to address this feedback - {review['feedback']} - using the newly retrieved chunks."""
            content = await write_section(section_points, section_template, state, extra, parent, state.get('generation_summary', ''))
            # state = await summarize_content_so_far(state)
        else:
            if i_points:
                content = await insert_images(content, i_points, parent_content, section_points, section_template, state)
            state = await store_generated_section(content, parent, section_template, state)
            return state
    if tried >= times:
        if review['needs_change']:
            if i_points:
                content = await insert_images(content, i_points, parent_content, section_points, section_template, state)
            content = content + f"\n\n> Human evaluation recommended.\n"
            state = await store_generated_section(content, parent, section_template, state)
            return state
    if i_points:
        content = await insert_images(content, i_points, parent_content, section_points, section_template, state)
    state = await store_generated_section(content, parent, section_template, state)
    log.debug(f"{WARNING}Generated: {content[:25]}... For {section_template['name']}{ENDC}")
    return state


async def insert_images(
    content: str,
    i_points: dict[str, dict],
    parent_content,
    section_points,
    section_template: dict,
    state: State
) -> str:
    # 1) Generate the new section content
    prompt = image_insert_prompt.format(
        draft=content,
        images=i_points,
        section=section_template['name'],
    )
    content = await write_section(
        section_points,
        section_template,
        state,
        None,
        parent_content,
        state.get("generation_summary", ""),
        prompt=prompt,
    )

    # 2) Find all the keys that need captions
    keys = {m.group(1) for m in IMAGE_PATTERN.finditer(content)}
    max_caption_len = config['generation'].get("maxCaptionSize", 20)

    async def _make_caption(key: str) -> str:
        info = i_points.get(key, {})
        desc = info.get("description", "")
        try:
            return await clean_caption(desc, state, max_caption_len)
        except Exception:
            log.exception(f"Clean_caption failed for image key {key!r}")
            return ""

    caption_tasks: dict[str, asyncio.Task[str]] = {
        key: asyncio.create_task(_make_caption(key))
        for key in keys
        if key in i_points
    }

    # 4) Await all captions
    if caption_tasks:
        await asyncio.gather(*caption_tasks.values())

    # 5) Do a synchronous regex replace using the ready captions
    def _repl(match: re.Match) -> str:
        key = match.group(1)
        info = i_points.get(key)
        if not info:
            return ""
        caption = caption_tasks.get(key).result() if key in caption_tasks else ""
        path = info.get("path", "")
        return f"![{caption}]({path})"

    result = IMAGE_PATTERN.sub(_repl, content)
    log.info(
        f"{OKCYAN}Images inserted to section: {section_template['name']}. Final chars: {len(result)}{ENDC}"
    )
    return result


async def get_image_points(section_template):
    i_points = await get_chunks_with_retry([f"{section_template['name']} {section_template['description']}"],
                                           section=[section_template['name']], points=True,
                                           points_type='image')
    if not i_points:
        i_points = await get_chunks_with_retry([f"{section_template['name']} {section_template['description']}"],
                                               points=True,
                                               points_type='image')
    if i_points:
        i_points = {f'IMAGE{i}': {'description': d.text_content, 'path': d.image_path}
                    for i, d in enumerate(i_points)}
    else:
        log.debug(f"No images found for: {section_template['name']}")
    return i_points


async def store_generated_section(content, parent, section_template, state):
    if not parent:
        state['generated_sections'][section_template['name']] = {'content': content,
                                                                 'subsections': {}}
    else:
        if parent['name'] not in state['generated_sections']:
            state['generated_sections'][section_template['name']] = {'content': None,
                                                                     'subsections': {
                                                                         section_template['name']: {
                                                                             'content': content,
                                                                             'subsections': {}
                                                                         }
                                                                     }}
        else:
            state['generated_sections'][parent['name']]['subsections'][section_template['name']] = {
                'content': content,
                'subsections': {}
            }
    return state


async def write_sections(state: State):
    log.info(f"{OKCYAN}--- WRITE SECTIONS ---{ENDC}")
    sections = state['section_to_write']
    sections = sorted(sections, key=lambda i: i.get('position', 0))
    for sect in sections:
        state = await generate_section(sect, state)
        log.info(f"Cost for section: {sect['name']}: {state['cost'].get_final_cost()}")
        parent_content = state['generated_sections'][sect['name']]['content']
        if sect.get('subSections'):
            log.info(f"{OKCYAN}Writing sub-sections of {sect['name']}.{ENDC}")
            subs = sect['subSections']
            subs = sorted(subs, key=lambda i: i.get('position', 0))
            for sub in subs:
                log.info(f"{OKGREEN}Writing sub-section {sub['name']} of {sect['name']}{ENDC}")
                state = await generate_section(sub, state, sect, parent_content)
                log.info(f"Cost for sub-section: {sub['name']}: {state['cost'].get_final_cost()}")
        log.info(f"Total cost of generating section {sect['name']}: {state['cost'].get_final_cost()}")
    return state

IMAGE_RX = re.compile(r'!\[([^\]]*)\]\(([^/\s)]+)\)') # md ![alt](relpath)


async def output(state: State):
    log.info(f"{OKCYAN}--- PUBLISH ORB ---{ENDC}")
    state = await update_doc_version(state)
    md_path, out_path = await store_temp_md(state)
    state = await md_to_document(pathlib.Path(md_path), pathlib.Path(out_path), state)
    if not state['out_path']:
        raise RuntimeError('Failed to write docx!')
    await store_latest_content(state['generated_sections'], state['out_path'])
    log.info(f"Cost for writing ORB: {state['cost'].get_final_cost()}")
    return state


async def store_temp_md(state):
    contents = []
    generated = state['generated_sections']
    config_sections = get_sorted_sections()
    for sect in config_sections:
        if sect['name'] in generated:
            contents.append(f"# {sect['name']}")
            contents.append(generated[sect['name']]['content'])
            if sect.get('subSections'):
                generated_subsections = generated[sect['name']]['subsections']
                for sub in sect['subSections']:
                    if sub['name'] in generated_subsections:
                        contents.append(f"## {sub['name']}")
                        contents.append(generated_subsections[sub['name']]['content'])
    if 'Revision History' in generated:
        contents.extend([f"# Revision History", generated['Revision History']['content'], '\n'])
    contents = '\n\n'.join(contents)
    md_path = os.path.abspath('/tmp/' + f"ORB_TRANSCEND_IM_{state['version']}.md")
    async with aiofiles.open(md_path, 'w') as f:
        await f.write(contents)
    log.debug(f"{OKCYAN}Wrote {len(contents)} characters in md.{ENDC}")
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.abspath(
        os.path.join(config['generation']['outputPath'], today, f"ORB_TRANSCEND_IM_{state['timestamp']}_v{state['version']}.docx")
    )
    return md_path, out_path


async def update_sections(state: State):
    log.info(f"{OKCYAN}--- UPDATE SECTIONS ---{ENDC}")
    sections = state['section_to_write']
    await asyncio.gather(*[generate_section_update(sect, state)
                           for sect in sections])
    return state


async def md_to_document(
    src_md: Path,
    out_file: Path,
    state: State,
    extra_pandoc_args: Sequence[str] | None = None,
    resource_paths: Iterable[Path] | None = None,
    margin_cm: float = 0.1,
    timeout: float | None = None,
) -> State:
    """
    Asynchronously convert Markdown -> PDF via Pandoc + LuaLaTeX.

    Assumes **all image paths in the Markdown are absolute** (or URLs).
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"{OKGREEN}Writing ORB PDF to: {out_file}{ENDC}")
    src_md = src_md.expanduser().resolve(strict=True)
    out_file = out_file.expanduser().resolve()

    workdir = Path(tempfile.mkdtemp(prefix="md2pdf_"))

    try:
        tmp_md = workdir / src_md.name
        tmp_md.write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")

        # 3. Assemble Pandoc command
        pandoc_extra: list[str] = [
            "-s",
            # f"--pdf-engine={config['generation']['pdfEngine']}",
            # "--variable", f"geometry:margin={margin_cm}cm",
            # "--pdf-engine-opt=-Xsearch-path=/usr/share/fonts",
            # "--variable", "mainfont=Noto Sans Symbols2",
            # "--variable", "monofont=Noto Sans Mono",
        ]

        if resource_paths:
            # Note: join multiple paths with os.pathsep (':' on *nix, ';' on Windows)
            joined = os.pathsep.join(str(p) for p in resource_paths)
            pandoc_extra.extend(["--resource-path", joined])

        if extra_pandoc_args:
            pandoc_extra.extend(extra_pandoc_args)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    pypandoc.convert_file,
                    str(tmp_md), to="docx", format="md",
                    extra_args=pandoc_extra,
                    outputfile=str(out_file)
                ),
                timeout,
            )
            state["out_path"] = str(out_file)
            gc.collect()
            log.info(f"PDF generated and written to: {out_file}")
        except asyncio.TimeoutError:
            raise PdfGenerationError(f"Pandoc conversion exceeded timeout={timeout}s")
        except RuntimeError as e:
            # pypandoc throws RuntimeError if Pandoc returns non-zero exit code
            raise PdfGenerationError(f"Pandoc failed: {traceback.format_exception(e)}")
    except Exception as e:
        log.error(f"{FAIL}PDF creation failed! Error: {traceback.format_exception(e)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        os.unlink(src_md)
        return state


async def init(state: State):
    log.info(f"{OKCYAN}--- INIT WRITER ---{ENDC}")
    state['generated_sections'] = (await get_latest('content')) or {}
    state['cost'] = CostCounter()
    sections = state['section_to_write']
    config_sections = get_sorted_sections()
    section_templates = plan_sections(config_sections, sections, state)
    if not section_templates:
        log.info(f"{WARNING}No sections to write!{ENDC}")
        state['route_to'] = END
        return state
    state['section_to_write'] = section_templates
    state['version'] = (await get_latest('version')) or 1
    state['delta_content'] = state.get('delta_content', None)
    if state['event'] == 'api-create':
        state['route_to'] = 'create'
    elif state['event'] != 'create' and not state['generated_sections']:
        raise NoExistingORBError('No prior ORB found but non-create event was requested.')
    elif state['event'] == 'create' and state['generated_sections']:
        log.info('Create event but ORB exists...will do update only. Getting chunks for sections to generate from current doc id.')
        state['delta_content'] = await get_current_document_delta(sections, state)
        state['route_to'] = 'update'
    elif state['event'] == 'create' and not state['generated_sections']:
        state['route_to'] = 'create'
    elif state['event'] in ['update', 'delete']:
        state['delta_content'] = await get_current_document_delta(sections, state)
        state['route_to'] = 'update'
    return state


async def get_current_document_delta(sections, state):
    delta = await get_points_for_doc(state['doc_id'], sections)
    this_delta = {}
    for dp in delta:
        for sect in dp['mapped_sections']:
            if sect not in this_delta:
                this_delta[sect] = [dp['text_content']]
            else:
                this_delta[sect].append(dp['text_content'])
    return this_delta

# -----------------
# Slug helpers
# -----------------
_slug_rx = re.compile(r"[^a-z0-9]+", re.I)

def slugify(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"slugify() expects str, got {type(text).__name__}")
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return _slug_rx.sub("-", text).strip("-")


# --------------------------------------------------
# Planner
# --------------------------------------------------
CREATE_EVENT = ("create", "api-create")

def plan_sections(
    config_sections: List[dict],
    sections: List[str],
    state: State,
    *,
    fuzzy_similarity: Optional[int] = 90,  # None == disable fuzzy
) -> List[dict]:
    """
    Return list of templates that match `sections`.
    - Exact slug matches first.
    - If `fuzzy_similarity` is not None, try fuzzy for leftovers.
    """

    requested: Set[str] = {slugify(s) for s in sections}
    log.debug(f"Requested (slug): {requested}")

    # --- Build quick lookups ---
    parent_by_slug: Dict[str, dict] = {}
    sub_by_slug: Dict[str, Tuple[str, dict]] = {}
    for parent in config_sections:
        pslug = slugify(parent["name"])
        parent_by_slug[pslug] = parent
        for sub in parent.get("subSections", []):
            sslug = slugify(sub["name"])
            sub_by_slug[sslug] = (pslug, sub)

    # --- Track results ---
    final: List[dict] = []
    claimed: Set[str] = set()

    # Helper to emit or merge parent/sub
    def _ensure_parent(pslug: str) -> dict:
        for tmpl in final:
            if slugify(tmpl["name"]) == pslug:
                return tmpl
        # Not yet in final -> deep-copy original or dummy
        if pslug in parent_by_slug:
            tmpl = copy.deepcopy(parent_by_slug[pslug])
        else:  # dummy parent
            tmpl = {
                "name": parent_by_slug.get(pslug, {}).get("name", pslug),
                "description": "NO CHANGES REQUIRED",
                "subSections": [],
            }
        final.append(tmpl)
        return tmpl

    # --------------------------------------------------
    # 1) Exact matches
    # --------------------------------------------------
    for pslug, parent in parent_by_slug.items():
        if pslug in requested:
            claimed.add(pslug)
            if state.get("event") in CREATE_EVENT:  # create-mode
                _ensure_parent(pslug)  # always keep parent
            else:  # update-mode
                final.append(copy.deepcopy(parent))

        # handle subSections
        matched_subs = [
            sub for sub in parent.get("subSections", [])
            if slugify(sub["name"]) in requested
        ]
        if matched_subs:
            for sub in matched_subs:
                sslug = slugify(sub["name"])
                claimed.add(sslug)

            if state.get("event") in CREATE_EVENT:
                tmpl = _ensure_parent(pslug)
                tmpl["subSections"] = matched_subs
            else:
                final.extend(copy.deepcopy(sub) for sub in matched_subs)

    # --------------------------------------------------
    # 2) Fuzzy fallback (optional)
    # --------------------------------------------------
    remaining = requested - claimed
    if remaining and fuzzy_similarity is not None:
        # Candidates are every slug we know about
        candidates = list(parent_by_slug.keys()) + list(sub_by_slug.keys())
        for rslug in list(remaining):
            best, score, *_ = process.extractOne(
                rslug, candidates, scorer=fuzz.QRatio
            )
            if score >= fuzzy_similarity:
                log.warning(
                    f"Fuzzy-matched '{rslug}' -> '{best}' (score {score})"
                )
                remaining.discard(rslug)
                if best in parent_by_slug:
                    continue
                else:
                    parent_slug, sub_cfg = sub_by_slug[best]
                    if state.get("event") in CREATE_EVENT:
                        _ensure_parent(parent_slug)["subSections"].append(
                            copy.deepcopy(sub_cfg)
                        )
                    else:
                        final.append(copy.deepcopy(sub_cfg))

    # --------------------------------------------------
    # 3) Dummy parents for orphaned subs (create only)
    # --------------------------------------------------
    if state.get("event") in CREATE_EVENT:
        for sslug in requested - claimed:
            if sslug in sub_by_slug:
                parent_slug, sub_cfg = sub_by_slug[sslug]
                _ensure_parent(parent_slug)["subSections"].append(
                    copy.deepcopy(sub_cfg)
                )
                claimed.add(sslug)

    # --------------------------------------------------
    # 4) Warn about anything still unclaimed
    # --------------------------------------------------
    still_missing = requested - claimed
    if still_missing:
        log.warning(f"Unmatched section names: {still_missing}")

    # Maintain original order of config_sections
    ordered: List[dict] = []
    seen = set()
    for parent in config_sections:
        pslug = slugify(parent["name"])
        for tmpl in final:
            if slugify(tmpl["name"]) == pslug and id(tmpl) not in seen:
                ordered.append(tmpl)
                seen.add(id(tmpl))

    # Append any dummy parents or standalone subs that were created later
    ordered.extend(t for t in final if id(t) not in seen)

    log.info(f"Sections resolved: {len(ordered)}")
    for sec in ordered:
        log.debug(f"- {sec['name']}")
        if sec.get("subSections"):
            for sub in sec.get("subSections", []):
                log.debug(f"  - {sub['name']}")
    return ordered


async def update_doc_version(state: State):
    log.info(f"{OKGREEN}Updating version...{ENDC}")
    latest_version = await get_latest("version")
    if not latest_version:
        latest_version = 1
        content = [{"version": latest_version, "author": "Content Writer Agent", "timestamp": datetime.now().isoformat()}]
        text = [f"| Version | Author | Timestamp |\n|---|---|---|",
                f"| {latest_version} | Content Writer Agent | {datetime.now().isoformat()} |"]
        state["generated_sections"]["Revision History"] = {"content": "\n".join(text),
                                                           "previous_history": content}
    else:
        latest_version += 1
        previous_history = state["generated_sections"]["Revision History"]["previous_history"]
        previous_history.append({"version": latest_version, "author": "Content Writer Agent", "timestamp": datetime.now().isoformat()})
        previous_history = sorted(previous_history, key=lambda k: k["version"])
        text = ["| Version | Author | Timestamp |\n|---|---|---|"]
        for hist in previous_history:
            text.append(f"| {hist['version']} | {hist['author']} | {hist['timestamp']} |")
        state["generated_sections"]["Revision History"] = {"content": "\n".join(text),
                                                           "previous_history": previous_history}
    return state


async def route(state: State):
    return state["route_to"]


write_agent = StateGraph(State)
write_agent.add_node("initialize", init)
write_agent.add_node("create", write_sections)
write_agent.add_node("update", update_sections)
write_agent.add_node("publish", output)

write_agent.add_conditional_edges("initialize", route)
write_agent.add_edge("create", "publish")
write_agent.add_edge("update", "publish")

write_agent.add_edge("publish", END)
write_agent.set_entry_point("initialize")

writer = write_agent.compile()