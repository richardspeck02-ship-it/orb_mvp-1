from functools import lru_cache
import os
import yaml
import os.path
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from langchain_openai import AzureOpenAIEmbeddings

@lru_cache(maxsize=1)
def get_config():
    return yaml.safe_load(open('/etc/orb-generation/config.yaml', 'r'))

config = get_config()

late_interaction_embedding_model = LateInteractionTextEmbedding(model_name=config['embeddingModels']['lateInteractionModel'],
                                                                specific_model_path=config['embeddingModels']['lateInteractionModelCache'])
                                                                # specific_model_path="./late")

# sparse_embedding_model = SparseTextEmbedding(model_name=config['embeddingModel']['sparseModel'],)

sparse_embedding_model = SparseTextEmbedding(model_name=config['embeddingModels']['sparseModel'],
                                             specific_model_path=config['embeddingModels']['sparseModelCache'])

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
