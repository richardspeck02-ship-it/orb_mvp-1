import base64
import re
from collections import defaultdict, Counter
from typing import List
import os

import aiofiles

os.environ['TOKENIZERS_PARALLELISM'] = "false"
from langchain_unstructured import UnstructuredLoader

from utils import config, transcribe_image, log, Document

# -------------------------------------------------------------
HF_RE = re.compile(r'(johnson\s*&\s*johnson|time-l[\w-]+)', re.I)
PAGE_RE = re.compile(r'^page\s+\d+\s+of\s+\d+', re.I)
BAD = {"n/a", "na", "tbd"}

def _is_trivial(s: str) -> bool:
    s = s.strip().lower()
    return s in BAD or len(s) < 30 or len(s.split(' ')) < config['embedding'].get('minChunkTokens', 75)


async def make_chunks(pdf_path: str, metadata: dict) -> List[Document]:
    # --- stream the PDF once ----------------------------------
    loader = UnstructuredLoader(
        file_path=pdf_path, strategy="hi_res",
        infer_table_structure=True, hi_res_model_name="yolox",
        new_indent_parser=True,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=f"./imgs/{os.path.basename(pdf_path).replace('.pdf', '').replace(' ', '-').replace('/', '_')}",
        split_pdf_page=True,
        split_pdf_allow_failed=True,
        split_pdf_concurrency_level=15,
        extract_image_block_types=["Image"],
        # extract_image_block_to_payload=True,
    )

    nodes, children, roots = {}, defaultdict(list), []
    text_freq, pages_seen = Counter(), set()
    last_eid, order, skip_toc = None, 0, False
    orphans = []

    async for d in loader.alazy_load():
        pg = d.metadata.get("page_number", -1)
        cat = d.metadata["category"]
        raw = d.page_content.strip()

        # 1. drop the title cover page
        if pg == 1:
            continue

        # 2. TOC skip block
        if cat == "Title" and raw.strip().lower().startswith("table of contents"):
            skip_toc = True
            continue
        if skip_toc:
            if cat == "Title":
                skip_toc = False
            else:
                continue

        # 3. strip obvious page headers / footers immediately
        if HF_RE.search(raw) or PAGE_RE.match(raw):
            continue

        # 4. collect header/footer candidates for short-line frequency filter
        if cat in ("Text", "NarrativeText") and len(raw) < 40:
            text_freq[raw] += 1; pages_seen.add(pg)

        # 5. create a node on every Title
        if cat == "Title":
            eid, pid, indent = d.metadata["element_id"], d.metadata.get("parent_id"), d.metadata.get("indent", 1)
            nodes[eid] = {"title": raw, "parent": pid, "indent": indent,
                          "parts": [], "order": order}
            order += 1; last_eid = eid
            (children[pid].append(eid) if pid in nodes else roots.append(eid))
            continue

        # 6. late header/footer suppression for docs where pattern missed
        if len(pages_seen) > 5 and cat in ("Text", "NarrativeText") \
                and text_freq[raw] > len(pages_seen) * 0.7:
            continue

        # 7. render the element into markdown
        if cat in ("Text", "NarrativeText", "ListItem"):
            part = raw
        elif cat == "Table":
            rows = [d.metadata.get("text_as_html", "")]
            part = "<-table-triplets->\n" + "\n".join(rows) + "\n<-table-triplets