import asyncio
import pathlib
import re

import gradio as gr
import uvicorn

from ui_state import UIState
from fastapi import FastAPI

app = FastAPI()


async def save_chunk_update(state, mapped_sections):
    if not mapped_sections:
        mapped_sections = []
    if state.interaction_state['selected_chunk']['mapped_sections'] != mapped_sections:
        state = await state.update_mapped_sections(state.interaction_state['selected_doc_id'],
                                                  state.interaction_state['selected_chunk']['point_id'],
                                                  mapped_sections)
    return state, gr.Checkbox(value=len(mapped_sections) > 0), gr.Dropdown(choices=mapped_sections,
                                                                          value=mapped_sections)


async def load_chunks_content(state, processed_doc, chunk_id):
    doc_id = state.processed_documents[processed_doc]
    point = state.document_chunks[doc_id][chunk_id]
    state.set('select_doc_id', doc_id)
    state.set('selected_chunk', point)
    return state, gr.Checkbox(value=len(point['mapped_sections']) > 0), gr.Dropdown(value=point['mapped_sections'],
                                                                                  choices=point['mapped_sections']), \
        gr.Markdown(value=point['text_content'])


async def load_chunks(state, processed_doc):
    doc_id = state.processed_documents[processed_doc]
    chunks = state.document_chunks[doc_id]
    sel_chunk_id = list(chunks.keys())[0]
    chunk = chunks[sel_chunk_id]
    state.set('select_doc_id', doc_id)
    state.set('selected_chunk', chunk)
    return state, gr.Checkbox(value=len(chunk['mapped_sections']) > 0), gr.Dropdown(value=chunk['mapped_sections'],
                                                                                  choices=chunk['mapped_sections']), \
        gr.Markdown(value=chunk['text_content']), gr.Radio(choices=list(chunks.keys()), value=sel_chunk_id)


async def chunk_review_blocks(app_name):
    state = await UIState(app_name).load_latest_state()
    gr.Markdown(f'## Embedding Review - {app_name}')
    with gr.Blocks() as review:
        with gr.Column():
            processed_documents = gr.Dropdown(label='Processed Documents', choices=[],
                                              max_choices=1, interactive=True, visible=False)

            with gr.Row(scale=2):
                doc_point = gr.Radio(label='Chunks', choices=[],
                                     interactive=True, visible=False)

                with gr.Column(scale=8):
                    queryable = gr.Checkbox(label='Queryable', value=False, interactive=False, visible=False,
                                            info='Chunks not tagged to any sections are not used for writing the ORB.')
                    mapped_sections = gr.Dropdown(allow_custom_value=True, multiselect=True, interactive=True,
                                                  choices=[], visible=False, label='Mapped Sections',
                                                  info='ORB sections that can be written using this chunk.')
                    update_changes = gr.Button(value='Save Changes', visible=False)
                    with gr.Accordion(label='Chunk Content', open=True):
                        text_content = gr.Markdown(label='Content', value='',
                                                   show_copy_button=True, visible=False, show_label=True,
                                                   max_height=512)

        state = gr.State(state)
        review.load(load_review_content, inputs=[state], outputs=[processed_documents, doc_point, mapped_sections, queryable, text_content, update_changes, state],
                    show_progress='full')
        update_changes.click(save_chunk_update, inputs=[state, mapped_sections],
                             outputs=[state, queryable, mapped_sections])

        doc_point.input(load_chunks_content, inputs=[state, processed_documents, doc_point],
                        outputs=[state, queryable, mapped_sections, text_content])

        processed_documents.input(load_chunks, inputs=[state, processed_documents],
                                  outputs=[state, queryable, mapped_sections, text_content,
                                           doc_point])

    return review


async def load_review_content(state: UIState):
    state = await state.load_latest_state()
    if state.processed_documents:
        docs = list(state.processed_documents.keys())
        sel_doc_name = docs[0]
        sel_doc_id = state.processed_documents[sel_doc_name]
        state.set('selected_doc_id', sel_doc_id)
        sel_chunk_id = list(state.document_chunks[sel_doc_id].keys())[0]
        sel_doc_chunk = state.document_chunks[sel_doc_id][sel_chunk_id]
        state.set('selected_chunk', sel_doc_chunk)
        return gr.Dropdown(choices=docs, value=sel_doc_name, visible=True, interactive=True),\
            gr.Radio(choices=list(state.document_chunks[sel_doc_id].keys()), value=sel_chunk_id, visible=True),\
            gr.Dropdown(choices=sel_doc_chunk['mapped_sections'], value=sel_doc_chunk['mapped_sections'], visible=True),\
            gr.Checkbox(value=sel_doc_chunk['queryable'], visible=True), gr.Markdown(value=sel_doc_chunk['text_content'], visible=True),\
            gr.Button(visible=True), state
    else:
        return gr.Dropdown(visible=False, interactive=True), \
            gr.Radio(visible=False), \
            gr.Dropdown(visible=False), \
            gr.Checkbox(visible=False), gr.Markdown(visible=False), \
            gr.Button(visible=False), state


FILE_PREFIX = "file="


def fix_image_links(markdown: str) -> str:
    """
    Replace ![alt](/abs/path) -> ![alt](file/abs/path)
    (only if the target exists on disk).
    """
    def repl(match):
        alt, path = match.group(1), match.group(2)
        if path.startswith(FILE_PREFIX) or not pathlib.Path(path).exists():
            return match.group(0)                 # already good or not local
        return f"![{alt}]({FILE_PREFIX}{path})"
    return re.sub(r"!\[([^\]]*)\]\((/[^)]+)\)", repl, markdown)


def make_content(generated_map):
    lines = []
    for section in generated_map:
        lines.append(f'# {section}')
        lines.append(generated_map[section]['content'])
        if 'subsections' in generated_map[section]:
            for sub in generated_map[section]['subsections']:
                lines.append(f'## {sub}')
                lines.append(generated_map[section]['subsections'][sub]['content'])
    return '\n'.join(lines)


async def refresh_state(state):
    state = await state.load_latest_state()
    if state.orb_versions:
        all_versions = list(state.orb_versions.keys())
        selected_version = all_versions[0]
        selected_orb = state.orb_versions[selected_version]
        state.set('selected_orb_version', selected_version)
        state.set('selected_orb_content', selected_orb)
        return state, gr.Radio(choices=all_versions, value=selected_version, visible=True), \
            gr.Markdown(value=make_content(selected_orb['content']), visible=True), gr.File(value=None, visible=False),\
            gr.Button(visible=True)
    else:
        return state, gr.Radio(visible=False), \
            gr.Markdown(value='', visible=False), gr.File(value=None, visible=False), gr.Button(visible=False)


async def populate_orb(state, selected_version):
    selected_orb = state.orb_versions[selected_version]
    state.set('selected_orb_version', selected_version)
    state.set('selected_orb_content', selected_orb)
    return state, gr.Markdown(value=make_content(selected_orb['content']), visible=True), gr.File(value=None, visible=False), gr.Button(visible=True)


async def start_download(state):
    return gr.File(state.interaction_state['selected_orb_content']['path'], visible=True)


async def orb_view_blocks(app_name):
    state = await UIState(app_name).load_latest_state()
    gr.Markdown(f'## Generated ORB Versions - {app_name}')
    with gr.Blocks() as orb:
        with gr.Column():
            refresh = gr.Button(value='Refresh', interactive=True)
            # write_new = gr.Button(value='Write New ORB', interactive=True)
            versions = gr.Radio(label='Versions', choices=[], value=None, interactive=True,
                                visible=False)

            with gr.Row():
                download = gr.Button(value='Download', visible=False)
                d_file = gr.File(visible=False)
            with gr.Accordion(label='Rendered Content', open=True):
                rendered = gr.Markdown(value='', visible=False, max_height=512)

        state = gr.State(state)
        orb.load(refresh_state, inputs=[state], outputs=[state, versions, rendered, d_file, download], show_progress='full', )
        refresh.click(refresh_state, inputs=[state], outputs=[state, versions, rendered, d_file, download])
        versions.input(populate_orb, inputs=[state, versions], outputs=[state, rendered, d_file, download])
        download.click(start_download, inputs=[state], outputs=[d_file])
    return orb


async def load_tabs(app_name):
    with gr.Tabs() as tabs:
        with gr.Tab('Review Embeddings', visible=True):
            await chunk_review_blocks(app_name)
        with gr.Tab('Generated ORBs', visible=True):
            await orb_view_blocks(app_name)
    return tabs


async def run_ui():
    return gr.TabbedInterface([await chunk_review_blocks('Transcend-IM'), await orb_view_blocks('Transcend-IM')], ["🖊 Review Embeddings", "📖 Generated ORBs"])


if __name__ == '__main__':
    app = gr.mount_gradio_app(app, asyncio.run(run_ui()), path="/", allowed_paths=["/shared", "/", '.'])
    uvicorn.run(app, host='0.0.0.0', port=80)
