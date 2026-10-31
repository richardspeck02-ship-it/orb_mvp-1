import os
from langchain_openai import AzureChatOpenAI
model_name = 'gpt-40'
endpoint_url = 'https://genaiapimna.jnj.com/openai-chat'
11m AzureChatOpenAI(
openai_api_key='',
openai_api_version='2024-06-01',
azure_endpoint=endpoint_url,
deployment_name=model_name,
temperature=0.7,
streaming-True,
cache=False,
verbose=True
)
# Example chat completion request
prompt = "Hello, how can I help you today?"
response = 11m.invoke(prompt)
print(response.content)