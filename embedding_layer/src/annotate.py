from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils import get_sections_as_markdown, call_llm_with_structured, Document, log, config


class Annotation(BaseModel):
    reasoning: str = Field(description='Your thoughts and reasoning for why you chose these sections in a paragraph as plaintext with no formatting.')
    mapped_sections: List[str] = Field(description='The list of sections that can be written using this chunk.')


annotation_prompt = ChatPromptTemplate.from_template("""
You are a data-analyst who marks relevant content for a task.

## TASK
For each **chunk** (text or image-description) decide which **exact** section(s)
or sub-section(s) it belongs to.

### Rules
1. If the chunk maps to a *sub-section*, output only that sub-section name (not its parent).
2. Never invent or alter section names. Choose only from the list.
3. A chunk may map to *multiple* sections/sub-sections. Show them all in one list.
4. If the chunk is an image and it is not merely a logo/icon, tag it.
5. If no section applies, output an empty list `[]`.

---
# SECTIONS

```markdown
{sections}

Your output must strictly follow this given format and nothing else. Do not add any fillers or acknowledgments.
{format}

# CHUNK
```plaintext
{chunk}
```plaintext
""").partial(sections=get_sections_as_markdown(), format=JsonOutputParser(pydantic_object=Annotation).get_format_instructions(),
)

async def annotate_chunk(chunk_content: Document):
    if not chunk_content.page_content.strip():
        return None
    prompt = annotation_prompt.format(chunk=chunk_content.page_content)
    # await asyncio.sleep(random.uniform(1, 5))
    # start = time()
    try:
        data = await call_llm_with_structured(prompt, temperature=0,
                                             retries=config['embedding'].get('annotationRetries'),
                                             timeout=config['embedding'].get('annotationTimeout'))
    except Exception:
        log.error(f'Chunk failed to annotate: {chunk_content}')
        chunk_content.metadata['mapped_sections'] = []
        chunk_content.metadata['queryable'] = False
        return chunk_content
    # await asyncio.sleep(1)
    # log.debug(f"Annotate: {((time() - start)):.3f}s")
    chunk_content.metadata['mapped_sections'] = data.get('mapped_sections', [])
    chunk_content.metadata['queryable'] = len(chunk_content.metadata['mapped_sections']) > 0
    return chunk_content
