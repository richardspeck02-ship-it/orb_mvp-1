import tiktoken

import numpy as np
from utils import config, embeddings

MAX_TOTAL = config['retrieval']['maxTokens']          # CE limit
RESERVED = config['retrieval']['reservedTokens']      # room for query + [CLS]/[SEP]
WIN_TOKENS = MAX_TOTAL - RESERVED
STRIDE = config['retrieval'].get('trimWindowSize', 300)

FAST_TOKENIZER = tiktoken.encoding_for_model(config['llmModel']['defaultModel'])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

async def auto_trim(query: str, passage: str) -> str:
    token_ids = FAST_TOKENIZER.encode(passage)
    if len(token_ids) <= WIN_TOKENS:
        return passage
    n_tokens = len(token_ids)
    q_vec = np.asarray(await embeddings.aembed_query(query))

    best_score = -np.inf
    best_span: tuple[int, int] = (0, WIN_TOKENS)

    for start_idx in range(0, n_tokens, STRIDE):
        end_idx = min(start_idx + WIN_TOKENS, n_tokens)

        window_token_ids = token_ids[start_idx:end_idx]

        window_text = FAST_TOKENIZER.decode(window_token_ids)

        w_vec: np.ndarray = np.asarray(
            (await embeddings.aembed_documents([window_text]))[0]
        )

        score = _cosine_similarity(q_vec, w_vec)

        if score > best_score:
            best_score = score
            best_span = (start_idx, end_idx)

        if end_idx == n_tokens:
            break

    best_token_ids = token_ids[best_span[0]: best_span[1]]
    return FAST_TOKENIZER.decode(best_token_ids,)
