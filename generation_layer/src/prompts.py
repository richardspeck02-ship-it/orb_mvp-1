
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils import config


class Questions(BaseModel):
    queries: List[str] = Field(description='The queries you want to run to get information to write this section.')


class Feedback(BaseModel):
    feedback: str = Field(description='Feedback instructions on what to update for the writer.')
    needs_change: bool = Field(description='Flag to indicate whether the section content needs an update.')


query_prompt = ChatPromptTemplate.from_template("""
You are a technical writer who is writing the operational runbook for an application.
The runbook must follow a certain structure and you are writing one section of it, by adhering to the given template.
The section you are writing may be a sub-section of another one. In such cases, the parent section will also be
provided to you after the current section template.
# CURRENT TEMPLATE
```json
{section}
```

# TASK

If the retrieved material below isn't enough to write the section, you must write at most 5 search phrases to clarify and fill in the missing knowledge.
1. No filler words but maintain word diversity relevant to the section.
2. At least one query must be a *literal* phrase built from key nouns in
   the section title (no synonyms or paraphrase).
3. At least one query must be from the nouns and entities given in the section description.
4. Remaining queries must introduce **new tokens** not present in any
   earlier query (guarantee lexical diversity).
5. If domain-specific acronyms or command names are obvious embed them explicitly.
6. Use professional imperative phrasing
7. Be unique; avoid near-duplicates

## REVIEW INSTRUCTIONS - these are generated after a review of your previously generated content
{extra}

# RETRIEVED MATERIAL
{chunks}

---
# FORMAT

You must output in this format only and nothing else: {format}

""").partial(format=JsonOutputParser(pydantic_object=Questions).get_format_instructions(),
             parent=None, extra=None,
             app_context=config['appContext'])


section_write_prompt = ChatPromptTemplate.from_template("""
SYSTEM:
You are a senior SRE and technical writer.
Write **only** the section content, in Markdown, following the provided JSON template exactly.
Do not write or add any header to the section.
Never invent facts or embellish beyond the chunks given.
+If a data point (ID, e-mail, phone, URL, code, path) does **not** appear verbatim inside a
+chunk, you **MUST** omit it—do **NOT** fabricate placeholders like someone@example.com.
The current section maybe a sub-section - in that case, the parent content will be given as reference.

# CURRENT TEMPLATE
```json
{template}
```

# PARENT CONTENT  (read-only reference; do not copy entire paragraphs.):
{parent}

---

# TASK

1. You must use the given chunks to write this section of the operations run book following the template given above for the current section.
2. Write the <-table-triplets-> html tables into a valid Markdown table with headers and rows when using in your write up.
3. You are allowed to and must ensure to insert relevant tables at their appropriate places.
4. Any identifier (program ID, report ID, e-mail, URL, path) must be reproduced **EXACTLY**
   + as it appears in the chunks. Do not change case, add domains, URL encode or guess missing parts.
You must be grounded entirely to the given chunks and never make up or try to fill in missing information - if you find missing information, you may add that as a footnote at the end
of your write up.

---

# EXTRA INSTRUCTIONS AND CHUNKS

## EXTRA INSTRUCTIONS

{extra}

---
## CHUNKS

{chunks}

----

# PARAPHRASING RULE
+ • Paraphrase everything in your own words. You may use the facts, figures, and data from the chunks verbatim, but you must never reuse their exact phrasing or sentence structure.
+
# BACKGROUND
{app_context}

ALWAYS -
+ • When you draw on image-derived chunks (e.g. diagram captions, OCR'd text from figures), do **not** copy them verbatim.
Instead, **paraphrase** and **integrate** their key facts into your prose in your own words,
omitting any raw alt-text or "A flow diagram illustrating…" style sentences.
• Paraphrase all chunk text. You may quote identifiers (IDs, URLs, code snippets) exactly, but all other exposition must be freshly worded.

1. Remove any redundancy or contradiction.
2. If you find some information is ambiguous, add a caution note.
3. No duplication from parent: If parent content covers a fact, either (a) reference it with [Parent] or (b) summarize in one sentence, then cite parent.
4. Length cap: Entire output ≤ 1200 tokens. If you exceed, truncate less-important details (e.g., optional notes).
# FORMAT
Your output must be in markdown format. Output only the section content - no need to add the section header.
Do not output any explanation or analysis. Do not enclose your output in ```markdown...``` tags.
+If essential data is missing, add a footnote such as
+[*] E-mail address not present in source data.
+Never invent the missing value.
""").partial(extra="", parent="",
             app_context=config['appContext'])


image_insert_prompt = ChatPromptTemplate.from_template("""
You are a markdown editor assistant.

Below is the draft of section **{section}** (it may already contain images).

### DRAFT  (do not include this header in your output!)
{draft}

---

### Available images  (alias ➜ description)
```json
{images}
```

# TASK

Wherever each image belongs in the draft, insert exactly the tag <IMAGE1>, <IMAGE2>, or <IMAGE3> in place (i.e. immediately after the sentence or clause that refers to it).
If the section does not require an image, no need to add one.
Do not invent any other tags or alter the existing text in any way.
If an image ID from the list does not apply anywhere, simply do not insert that tag.
Make sure the <IMAGE#> tags are always on their individual lines and never in between a sentence or a paragraph.
If there are existing images in markdown format (exclamation)[caption](path), look at their caption -
    if the image you want to add is same as the image already present based on the caption, no need to duplicate it and add it again.
You can, re-arrange or replace existing images using their captions based on the relevancy but add duplicate images.
Output only the input markdown as is but updated with the <IMAGE#> tags in their relevant places.
Do not output or enclose in markdown tags. Also, do not keep or add the # DRAFT header in your output.

---
Example of desired behavior:
Original:

“Open port 443 on the security group.”
After:
“Open port 443 on the security group.
<IMAGE2>”
---
Now, apply this to the full draft.
""")


section_update_prompt = ChatPromptTemplate.from_template("""
You are a technical writer who is updating the operational runbook for an application.
The runbook must follow a certain structure and you are writing one section of it, by adhering to the given template.

# CURRENT TEMPLATE
```json
{template}
```

---

# PARENT CONTENT  (read-only reference; do not copy entire paragraphs):
{parent}
---

The application and document context for which we are writing is -
## APP CONTEXT
{app_context}

# TASK

Some operations were made to a document and as a result, this current section might need updates.
Your task is to look at the current section content and the latest chunks from the document.
The current section maybe a sub-section - in that case, the parent content will be given as reference.
Now, you must decide if the section content needs updating based on the operation and the chunk content.
Use the given chunks to update the section content, adhering to the given template.
You are allowed to and must ensure to insert relevant images and tables at their appropriate places.
You must be grounded totally to the given chunks and never make up or try to fill in missing information.
Always make sure you follow the context and what and why we are writing when writing your section.
Follow the guidelines given below to help you -

## OPERATION TYPES

1. UPDATE - for update operations, you must consider updating the section's existing content or add new information.
2. DELETE - for delete operations, you must consider updating the section's existing content by deleting information that got removed.

## GUIDE
1. Depending on the operation, decide carefully what kind of change you need to make to the section.
2. Look carefully in the chunks - some of them may contain images and tables.
   2.1 Images are marked within <image-tag-> tags. The images are in markdown format with the description and the local path.
       2.1.1 Carefully note this local path and remember it - it is absolute and you must never make it up
       2.1.2 You can use the image contents for your task
       2.1.3 When you want to insert an image somewhere in the section, insert it in markdown for so that it renders in markdown.
   2.2 Tables are stored as triplets and marked within <-table-triplets-> tags
       2.2.1 You can use the table contents for your task
       2.2.2 When you want to insert a table somewhere in the section, insert it as markdown so that it renders in markdown.
3. When you find that no changes are needed to the section, output the section as-is as you received
4. When you find that after a delete operation, the entire section needs deletion as the info it contained is getting removed,
   then you can delete the section content and output with warning emoticon - No knowledge found for this section.
5. Remove any redundancy or contradiction.
6. If information is ambiguous, add a caution note.
7. Length cap: Entire output ≤ 600 tokens. If you exceed, truncate less-important details (e.g., optional notes).

---
# CURRENT CONTENT
{section}
---
# EXTRA INSTRUCTIONS
{extra}

---
# FORMAT

Your output must be in markdown format. Output only the new version of the section content after making the changes necessary
- no need to add the section header. Do not enclose your output in ```markdown...``` tags.

---

# CHUNKS
{chunks}
---

# OPERATION
{event}
---
""").partial(extra=None, parent="",
             app_context=config['appContext'])


section_review_prompt = ChatPromptTemplate.from_template("""
You are acting as a strict SRE (Site Reliability Engineer) reviewing sections of an operations runbook.
The runbook must follow a certain structure, and the current section must adhere strictly to the provided template.

```json
{template}
```

---

The application and document context for which we are writing is -
## APP CONTEXT
{app_context}

#Task
You must examine the content of the section provided and critique whether the section adheres to the template or deviates from it. If the section content is off or missing information, output clear instructions specifying the exact issues. Use the criteria below for your critique:

# Fact References:

Check if any sentence in the section references facts unsupportable by the chunks (e.g., mentions software, version, or steps not present in the chunks).
Look for proper nouns or step names not found in the chunk text.

#Table Conversion:
Ignore column order differences; flag only when a row/column or triplet value is missing/incorrect.

#Duplication with Parent:
Flag only if ≥40 % of sentences are ≥80 % identical to the parent summary.

#Placeholders:
If any curly-brace placeholder ({template}, {format}, etc.) is still visible, flag `placeholder_not_filled`.


#Extra Instructions:
Always make sure you follow the context and what and why we are writing when writing your section.
{extra}

---

If the section passes all checks, output needs_change: false.
If the section fails any checks, clearly specify the necessary changes and output needs_change: true.
Output Format: Your output must strictly follow the format provided below and should not deviate (e.g., avoid extra explanations):

{format}
Section Content:
{content}

# Chunks:
{chunks}

# Parent Section Content:
{parent}
""").partial(format=JsonOutputParser(pydantic_object=Feedback).get_format_instructions(),
             parent="", extra="",
             app_context=config['appContext'])


summary_prompt = ChatPromptTemplate.from_template("""
We are currently writing the operational runbook for Transcend IM application following the standard below -
```json
{template}
```

The following sections have been written -
{generated}
---

# TASK

Your task is to write a short description and short summary capturing the entire runbook
written so far section by section in order. Dont make it too long.
This will be used to provide context to the next section writer so that it knows how
far we have proceeded.

---
Output only the summary and nothing else.
""")
