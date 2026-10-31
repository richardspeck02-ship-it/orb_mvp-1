import asyncio
import itertools

import aiohttp.client_exceptions

from utils import get_docs_for_app, get_points_for_doc, get_orb_history, log, WARNING, ENDC
from vector_db import update_document_sections


class UIState:
    def __init__(self, app_name):
        self.app_name = app_name
        self.processed_documents = {}
        self.document_chunks = {}
        self.interaction_state = {}
        self.orb_versions = {}

    async def load_latest_state(self):
        try:
            processed = await get_docs_for_app(self.app_name)
            self.processed_documents = {p['document_name']: p['document_id'] for p in processed}
            tasks = [get_points_for_doc(self.processed_documents[d_name])
                     for d_name in self.processed_documents]
            chunks_for_docs = await asyncio.gather(*tasks)
            chunks_for_docs = itertools.chain(*chunks_for_docs)
            for chunk in chunks_for_docs:
                if chunk['doc_id'] not in self.document_chunks:
                    self.document_chunks[chunk['doc_id']] = {chunk['point_id']: {'point_id': chunk['point_id'],
                                                                                 'text_content': chunk['text_content'],
                                                                                 'mapped_sections': chunk['mapped_sections'],
                                                                                 'queryable': chunk['queryable']}}

                else:
                    self.document_chunks[chunk['doc_id']] = self.document_chunks[chunk['doc_id']] | {chunk['point_id']: {'point_id': chunk['point_id'],
                                                                                                                        'text_content': chunk['text_content'],
                                                                                                                        'mapped_sections': chunk[
                                                                                                                            'mapped_sections'],
                                                                                                                        'queryable': chunk['queryable']}}

            self.orb_versions = await get_orb_history()
            return self
        except aiohttp.client_exceptions.ClientConnectionError:
            log.warning(f'{WARNING}Cannot load state...some upstream APIs are down...will wait.{ENDC}')
            await asyncio.sleep(1)
            return await self.load_latest_state()

    def set(self, key, value):
        self.interaction_state[key] = value

    async def update_mapped_sections(self, self, doc_id, point_id, sections):
        updates = await update_document_sections(point_id, sections)
        self.document_chunks[doc_id][point_id] = self.document_chunks[doc_id][point_id] | updates
        return self
