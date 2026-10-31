import gc
import random
import socket
from functools import lru_cache
import os
from multiprocessing import cpu_count

import aiohttp
import openai
import yaml

import asyncio
import functools
import inspect

import os.path
import re
import sys
import traceback
from typing import List
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from json import loads

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

from loguru import logger as log
from valkey import ResponseError
from valkey.asyncio import Valkey
from enum import Enum


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
BASE_DELAY = 1.5                # seconds before 1st retry
JITTER_RANGE = 0.3
cpu_limit = os.getenv('CPU_LIMIT', None)
if not cpu_limit:
    cpu_limit = cpu_count()
EXECUTOR = ThreadPoolExecutor(max_workers=int(float(cpu_limit)))


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class EmbedResult(Enum):
    SUCCESS = "success"
    REQUEUE_LATER = "requeue_later"
    SKIP_AND_ACK = "skip_and_ack"
    FAILURE = "failure"


class CostCounter(BaseModel):
    """
    An object providing helpers to track usage costs.
    This is mainly used in the track_token_usage function to log
    all usage costs.
    """
    input_costs: List[float] = Field(default_factory=list)
    output_costs: List[float] = Field(default_factory=list)
    total_costs: List[float] = Field(default_factory=list)

    def add_cost(self, inp, op):
        self.input_costs.append(inp)
        self.output_costs.append(op)
        self.total_costs.append(inp + op)

    def get_final_cost(self, as_str=True, with_color=True):
        if not as_str:
            return round(sum(self.input_costs), 4), round(sum(self.output_costs), 4), round(sum(self.total_costs), 4)
        if with_color:
            return f'{HEADER}Total Cost: ${round(sum(self.total_costs), 4)} (${round(sum(self.input_costs), 4)}, ${round(sum(self.output_costs), 4)}){ENDC}'
        return f'Total Cost: ${round(sum(self.total_costs), 4)} (${round(sum(self.input_costs), 4)}, ${round(sum(self.output_costs), 4)})'


def display_errors(gather_result):
    errors = [i for i in gather_result if isinstance(i, Exception)]
    if errors:
        log.warning(f'{WARNING}{len(errors)} tasks failed! Errors logged below.{ENDC}')
        for e in errors:
            log.warning(f'{FAIL}{traceback.print_exception(e.__class__, e, e.__traceback__)}{ENDC}')


@lru_cache(maxsize=1)
def get_config():
    return yaml.safe_load(open('/etc/orb-generation/config.yaml', 'r'))


image_context_prompt = ChatPromptTemplate.from_template("""
You are an Image Parser specializing in extracting the contents of an image.
You are currently working to extracting contents of an image in plain text so that it can be stored as a vector embeddings
which would be used to write new documents.

# Task

1. Watch the image very carefully and describe its content in a verbose manner such that anyone reading it can recreate
it.
   1.1 First write what the image is about and what it is showing.
2. Capture all possible details such as items, objects, their positionings, if they relate to each other etc so that anyone reading it can recreate it in their head.
   2.1 If the image is a table or has tables, you can add the tables exactly as they and where they appear in triplet format
3. Never mention it is an image, just describe it as per rules 1 and 2.
""")
config = get_config()
EMBED_CONSUMER_NAME = 'embed_' + config['redis'].get('workerName', f'consumer_{socket.gethostname()}')
MAX_CONCURRENCY = asyncio.Semaphore(config.get('maxConcurrency', cpu_count()))
log.add(sys.stderr, level=config.get('logLevel', 'warning').upper())
late_interaction_embedding_model = LateInteractionTextEmbedding(model_name=config['embeddingModels']['lateInteractionModel'],
                                                                specific_model_path=config['embeddingModels']['lateInteractionModelCache'])
                                                                # specific_model_path="./late")

# sparse_embedding_model = SparseTextEmbedding(model_name=config['embeddingModel']['sparseModel'],)

sparse_embedding_model = SparseTextEmbedding(model_name=config['embeddingModels']['sparseModel'],
                                             specific_model_path=config['embeddingModels']['sparseModelCache'])
# vision_model, _, vision_preprocessor = open_clip.create_model_and_transforms(config['embeddingModels']['visionModel'], pretrained=config['embeddingModels']['visionModelCheckpoint'],
#                                                                              device='cpu',
#                                                                              force_quick_gelu=True,)
# vision_model.eval()


async def get_points_for_doc(doc_id) -> List[dict]:
    async with aiohttp.ClientSession(headers={'Accept-Encoding': 'gzip', 'Accept': 'application/json'}) as sess:
        async with sess.get(config['generation']['pointsEndpoint'], params={'doc_id': doc_id, 'sections': ''}) as resp:
            resp.raise_for_status()
            points = (await resp.json())['data']
    return points


async def get_docs_for_app(app_name):
    async with aiohttp.ClientSession(headers={'Accept-Encoding': 'gzip', 'Accept': 'application/json'}) as sess:
        async with sess.get(config['embedding']['appMetadataEndpoint'], params={'app_name': app_name}) as resp:
            resp.raise_for_status()
            points = (await resp.json())
    return points


async def submit_write_request():
    async with aiohttp.ClientSession() as sess:
        async with sess.post(config['embedding']['writeEndpoint']) as resp:
            resp.raise_for_status()
            points = (await resp.json())
    return points


async def get_orb_history():
    async with aiohttp.ClientSession(headers={'Accept-Encoding': 'gzip', 'Accept': 'application/json'}) as sess:
        async with sess.get(config['embedding']['historyEndpoint']) as resp:
            resp.raise_for_status()
            points = (await resp.json())
    return points


async def transcribe_image(image_bytes, do_sleep=False, prompt=None):
    if not prompt:
        prompt = image_context_prompt.format()
    message = [SystemMessage(
        content=[
            {"type": "text", "text": prompt},
        ]
    ),
        HumanMessage(content=[{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64, {image_bytes}"}
        }])]
    if do_sleep:
        await asyncio.sleep(1)
    response = await call_llm(message, model=config['llmModel']['imageDescriptorModel'],
                              temperature=0.0)
    del image_bytes
    if not response:
        return ""
    return response


def garbage_collect():
    dead = gc.collect()
    log.debug(f"{OKCYAN}gc.collect() freed {dead} objects.{ENDC}")


def count_tokens(text, model=config['llmModel']['defaultModel']):
    """
    Use tiktoken to count tokens used for a particular piece of text.
    """
    enc = tiktoken.encoding_for_model(model)
    toks = len(enc.encode(text))
    # log.debug(f'{OKCYAN}Used {toks} tokens.{ENDC}')
    return toks


def record_cost(i_tokens, o_tokens, cost_counter: CostCounter, display=False):
    """
    Helper to log all token usages from the track_token_usage decorator.
    :param i_tokens:
    :param o_tokens:
    :param cost_counter:
    :param display:
    :return:
    """
    i_cost = round(float(config['llmModel']['inputCost']) * i_tokens, 4)
    o_cost = round(float(config['llmModel']['outputCost']) * o_tokens, 4)
    if display or cost_counter is None:
        log.info(f'{WARNING}Cost: ${i_cost + o_cost} (${i_cost}, ${o_cost}){ENDC}')
    if cost_counter:
        cost_counter.add_cost(i_cost, o_cost)


def track_token_usage(func):
    """Decorator to track input and output tokens for sync, async, and streaming functions."""

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            input_prompt = kwargs.get("prompt", args[0] if args else "")
            if isinstance(input_prompt, str):
                input_tokens = count_tokens(input_prompt)
            else:
                input_tokens = 0

            response = await func(*args, **kwargs)

            output_tokens = count_tokens(response)
            record_cost(input_tokens, output_tokens, kwargs.get('cost_counter'))

            return response

        return async_wrapper

    elif inspect.isasyncgenfunction(func):
        @functools.wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            input_prompt = kwargs.get("prompt", args[0] if args else "")
            if isinstance(input_prompt, str):
                input_tokens = count_tokens(input_prompt)
            else:
                input_tokens = 0
            output_tokens = 0

            async for chunk in func(*args, **kwargs):
                if chunk:
                    output_tokens += count_tokens(chunk)
                    yield chunk

            record_cost(input_tokens, output_tokens, kwargs.get('cost_counter'))

        return async_gen_wrapper

    else:  # If function is sync, use regular wrapper
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            input_prompt = kwargs.get("prompt", args[0] if args else "")
            if isinstance(input_prompt, str):
                input_tokens = count_tokens(input_prompt)
            else:
                input_tokens = 0

            response = func(*args, **kwargs)  # Call the sync function

            output_tokens = count_tokens(response)
            record_cost(input_tokens, output_tokens, kwargs.get('cost_counter'))

            return response

        return sync_wrapper


def initialize_embeddings(model='text-embedding-3-large'):
    """
    Initializes an instance of OpenAIEmbeddings.

    Returns:
        OpenAIEmbeddings: An initialized instance of the OpenAIEmbeddings class.
    """
    return AzureOpenAIEmbeddings(
        # azure_deployment=settings.OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        model=model,
        chunk_size=1,
        openai_api_version=config['embeddingModels']['apiVersion'],
        azure_endpoint=config['embeddingModels']['endpoint'],
        openai_api_type='azure',
        openai_api_key=os.getenv(config['embeddingModels']['apiKey']).strip(),
    )


embeddings = initialize_embeddings()


def initialize_llm(model='', temperature=config['llmModel'].get('defaultTemperature', 0.7)) -> AzureChatOpenAI:
    """
    Function used to create an instance of AzureChatOpenAI.

    Returns:
        AzureChatOpenAI: An instance of AzureChatOpenAI.
    """
    if model.startswith('o'):
        llm_chat = AzureChatOpenAI(
            deployment_name=model or config['llmModel']['defaultModel'],
            azure_endpoint=config['llmModel']['baseUrl'],
            openai_api_version=config['llmModel']['apiVersion'],
            openai_api_key=os.getenv(config['llmModel']['apiKey']).strip())
    else:
        llm_chat = AzureChatOpenAI(
            deployment_name=model or config['llmModel']['defaultModel'],
            azure_endpoint=config['llmModel']['baseUrl'],
            openai_api_version=config['llmModel']['apiVersion'],
            temperature=temperature,
            openai_api_key=os.getenv(config['llmModel']['apiKey']).strip())
    return llm_chat


# @track_token_usage
async def call_llm(prompt, model=config['llmModel']['defaultModel'], temperature=None, **kwargs):
    """
    Common wrapper used all over the application to invoke the LLM in non-streaming mode.
    :param prompt:
    :param model:
    :param temperature:
    :param kwargs:
    :return:
    """
    max_retries = kwargs.get('retries') or config['llmModel'].get('timeoutRetries', 3)
    for attempt in range(1, max_retries + 1):
        llm = initialize_llm(model, temperature)

        try:
            res = await asyncio.wait_for(llm.ainvoke(prompt), timeout=kwargs.get('timeout') or config['llmModel'].get('generationTimeout', 30))
            return res.content
        except (asyncio.TimeoutError,
                openai._exceptions.APITimeoutError,
                openai._exceptions.APIConnectionError,
                openai._exceptions.RateLimitError) as err:

            if attempt == max_retries:
                log.error(f'{FAIL}Retries exhausted while waiting for LLM response. {ENDC}')
                return None

            # log & back-off before retrying
            sleep_for = BASE_DELAY * (2 ** (attempt - 1))
            sleep_for *= random.uniform(1 - JITTER_RANGE, 1 + JITTER_RANGE)
            log.warning(
                f"[call_llm] attempt {attempt}/{max_retries} failed "
                f"({err.__class__.__name__}: {err}); retrying in {sleep_for:.1f}s"
            )
            await asyncio.sleep(sleep_for)

        except Exception:
            log.error(f'{FAIL}Unknown exception when calling LLM: {traceback.format_exc()}{ENDC}')
            return None

    return None


@track_token_usage
async def stream(prompt, model=config['llmModel']['defaultModel'], temperature=None, **kwargs):
    """
    Common wrapper used all over the application to invoke the LLM in streaming mode.
    :param prompt:
    :param model:
    :param temperature:
    :param kwargs:
    :return:
    """
    llm = initialize_llm(model, temperature)
    async for chunk in llm.astream(prompt):
        if chunk:
            yield chunk.content


async def call_llm_with_structured(prompt, model=config['llmModel']['defaultModel'], temperature=None, **kwargs):
    """
    Common wrapper used all over the application to invoke the LLM with structured output.
    The LLMs raw output is parsed before returning.
    :param prompt:
    :param model:
    :param temperature:
    :param kwargs:
    :return:
    """
    res = await call_llm(prompt, model=model, temperature=temperature, **kwargs)
    return await extract_json_response(res, model=model, **kwargs)


async def extract_json_response(llm_data, retries=config['llmModel']['structuredOutputRetries'],
                                model=config['llmModel']['defaultModel'],
                                **kwargs):
    """
    Parses the LLM output and extracts the structured JSON in its response.
    Whenever there are errors in the LLM's output structure, it performs retries.
    Runtime Error is raised when retries fail as subsequent steps may not proceed
    if the structured output could not be generated.
    :param llm_data:
    :param retries:
    :param model:
    :param kwargs:
    :return:
    """
    tries = 1
    while tries <= retries:
        tries += 1
        # try naively to read json as LLM can sometimes not send the ```json...``` tags
        try:
            if not llm_data:
                return {}
            return loads(llm_data)
        except Exception:
            pass
        match = re.search(r"```\s*([\[\s\S]*?)\s*```", llm_data)
        if match:
            json_content = match.group(1).strip().lstrip('json ')
            try:
                return loads(json_content)
            except Exception as e:
                log.error(f'{FAIL}JSON parse error! {e}{ENDC}')
                prompt = f"""The content {json_content} you generated which should have a JSON within ```json...``` tags failed to parse due to this error - {traceback.format_exc()}
You must fix this JSON and output it properly within ```json...``` tags. Output only the fixed JSON in your output and nothing else."""
                llm_data = await call_llm(prompt, temperature=0.3, model=model, **kwargs)
    raise RuntimeError(f'Retries {retries} exhausted trying to fix LLM output. Final output: {llm_data}')


@lru_cache(1)
def get_sections_as_markdown(no_desc=True):
    lines = []
    sections = sorted(config['generateSections'], key=lambda x: int(x['position']))
    for section in sections:
        lines.append(f'# SECTION NAME: {section["name"]}')
        lines.append(f'# SECTION POSITION - {section["position"]}')
        lines.append('# SECTION FORMAT: ' + section['description'])
        if section.get('subSections'):
            for sub in section['subSections']:
                lines.append(f'## SUB-SECTION OF {section["name"]} NAME: {sub["name"]}')
                lines.append(f'## SUB-SECTION OF {section["name"]} POSITION - {sub["position"]}')
                lines.append(f'## SUB-SECTION OF {section["name"]} FORMAT: ' + sub['description'])
    lines.append('')
    return '\n'.join(lines)


redis_client = Valkey(host=config['redis']['host'], port=config['redis']['port'], decode_responses=True)

redis_conf = config['redis']
queues = {
    redis_conf['dlQueueName']: None,
    redis_conf['embedQueueName']: redis_conf['embedConsumerGroupName'],
    redis_conf['generatorQueueName']: redis_conf['generatorConsumerGroupName'],
    # redis_conf['preprocessQueueName']: redis_conf['preprocessConsumerGroupName']
}


async def ensure_stream_exists(name, maxlen=10000, approximate=True):
    """
    Ensure a Redis stream exists with optional maxlen trimming.
    Creates an empty entry if missing.
    """
    try:
        await redis_client.xinfo_stream(name)
        log.info(f"Stream already exists: {name}")
    except ResponseError as e:
        if "no such key" in str(e).lower():
            log.info(f"Creating missing stream: {name}")
            try:
                await redis_client.xadd(
                    name,
                    fields={"init": "init"},
                    maxlen=maxlen,
                    approximate=approximate,
                )
                log.info(f"Stream created: {name}.")
                if queues[name]:
                    log.info(f'Adding its group: {queues[name]}')
                    try:
                        await redis_client.xgroup_create(name=name, groupname=queues[name], id="1",
                                                         mkstream=True)
                        log.info(f"Consumer group '{queues[name]}' created.")
                    except Exception as e:
                        if "BUSYGROUP" in str(e):
                            log.info(f"Consumer group '{queues[name]}' already exists.")
                        else:
                            log.exception("Failed to create consumer group.")
                            raise
            except Exception as ex:
                log.exception(f"Failed to create stream {name}: {ex}")
        else:
            log.exception(f"Unexpected error checking stream {name}: {e}")


async def create_streams():
    tasks = [ensure_stream_exists(queue_name) for queue_name in queues]
    await asyncio.gather(*tasks)
    await redis_client.aclose()
