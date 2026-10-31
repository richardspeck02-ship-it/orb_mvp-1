import os
import secrets
import string

import valkey.asyncio as redis
from langchain_openai import AzureChatOpenAI
from loguru import logger as log
from yaml import safe_load

current_dir = os.path.dirname(os.path.realpath(__file__))
CFG_PATH = os.path.join('/etc/orb-generation', 'config.yaml')
# CFG_PATH = os.path.join(current_dir, '..', 'config.yaml')

if not os.path.exists(CFG_PATH):
    CFG_PATH = os.path.join(current_dir, '..', 'config.yaml')
config = safe_load(open(CFG_PATH, 'r'))
alphabet = string.ascii_letters + string.digits


def random_seed():
    return ''.join(secrets.choice(alphabet) for _ in range(16))


pool = redis.ConnectionPool(
    host=config['redis']['host'],
    port=config['redis'].get('port', 6379),
    db=config['redis'].get('db', 0),
    decode_responses=True,
    health_check_interval=30,
    max_connections=9999
)
redis_client = None


def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(connection_pool=pool)
    return redis_client


default_db = None
self_db = None

llm = AzureChatOpenAI(
    api_key=os.environ.get(config['llmModel'].get('apiKey', 'llm_key')),
    # api_key='abcd',
    api_version=str(config['llmModel']['apiVersion']),
    azure_endpoint=config['llmModel']['baseUrl'],
    azure_deployment=config['llmModel']['defaultModel'],
    temperature=0,
    streaming=True,
    cache=False,
    verbose=True
)

context_windows = {
    'gpt4': 8192,
    'gpt-4': 8192,
    'gpt-4-turbo': 128000,
    'gpt-4o': 128000
}

base_directory = config['s3'].get('inputPath', current_dir)
output_directory = config['s3'].get('tempPath', current_dir)
output_directory_s3 = config['s3'].get('sharedPath', current_dir)
output_directory_s3 = os.path.join(output_directory_s3, config["s3"]["rootDirectory"])


def clean_json_data(data):
    if data.startswith("```"):
        data = data.lstrip("```").rstrip("```").strip()
    if data.lower().startswith('final answer'):
        data = data.lstrip('Final Answer:')
    if '```' in data:
        data = data.split('```', 1)[1].rstrip('```').lstrip('json').strip()
    return data


def clean_yaml_data(data):
    if data.startswith("```"):
        data = data.lstrip("```").rstrip("```").strip()
    if data.lower().startswith('final answer'):
        data = data.lstrip('Final Answer:')
    if '```' in data:
        data = data.split('```', 1)[1].rstrip('```').lstrip('yaml').strip()
    print(data)
    return data


log.debug('All external systems - Redis, Qdrant, Infinity - are up. App starting.')
