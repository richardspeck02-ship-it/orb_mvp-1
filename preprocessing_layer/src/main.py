from preprocess import DocumentPreprocessor
import os
import utils
from database.redis_operations import RedisOperations
from pprint import pprint
from fastapi import FastAPI, Query, Body
from typing import Optional
import uvicorn
import sys
from utils import config as conf, log

base_directory = utils.base_directory
redis_ops = RedisOperations()
log = utils.log
app = FastAPI(title="ORB Generation API",
              description="This API allows users to work with documents for automated ORB generation",
              version="1.0.0")

@app.get('/healthz', status_code=200)
async def check_redis():
    try:
        await redis_ops.ping()
    except Exception:
        log.error(f'Ping failed! Redis is not available at {conf["redis"]["host"]}:'
                  f'{conf["redis"].get("port", 6379)}! A running instance of Redis is required for this service.')
        sys.exit(1)

@app.get("/getDocuments/", summary="Fetch the Document Details (Name, Updation Status,"
                                   " Checksum, Delta etc.) from Redis DB")
async def document_details(
    app_name: Optional[str] = Query(None, description="Name of the application to fetch documents for",
                                    title="Application Name"),
    document_names: Optional[str] = Query(None, description="Comma-separated list of "
                                                            "document names to fetch details for",
                                          title="Document Names"),
    include_metadata: Optional[bool] = Query(False, description="Whether to include metadata"
                                                                " in the response",
                                            title="Include Metadata")
):
    """
    API endpoint to get document details from Redis.
    Optionally accepts an S3 event payload in the request body.
    """
    result = await DocumentPreprocessor.get_document_details(app_name, document_names, include_metadata)
    return result

@app.delete("/clearDB/", summary="Clear the Document Details from Redis DB")
async def clear_db(key_prefix: str = Query(None, description="Prefix to filter keys to delete",
                                          title="Key Prefix")):
    """
    API endpoint to clear document details from Redis.
    """
    if not key_prefix:
        return {"error": "Key prefix is required to clear the database.", 'deleted': False}
    key_prefix = key_prefix.strip().split(',')
    redis_ops = RedisOperations()

    try:
        await redis_ops.clear_db(key_prefix)
    except Exception as e:
        log.error(f"Error clearing database: {e}")
        return {"error": str(e), 'deleted': False}

    return {'deleted': True, 'error': None}

@app.post("/processDocuments/", summary="Process the Document Metadata and Insert/Update in Redis DB")
async def handle_preprocessing(app_name: Optional[str] = Query(None, description="Name of the application"
                                                                                 " to process documents for",
                                                               title="Application Name"),
                               document_names: Optional[str] = Query(None, description="Comma-separated list of "
                                                                                       "document names to process details for",
                                                                     title="Document Names"),
                               s3_event: Optional[dict] = Body(None, example={
                                   "Records": [
                                       {
                                           "eventName": "ObjectCreated:Put",
                                           "s3": {
                                               "bucket": {
                                                   "name": "example-bucket"
                                               },
                                               "object": {
                                                   "key": "App name/Document name",
                                               }
                                           }
                                       }
                                   ]
                               }, description="S3 Event JSON", title="S3 Event")):
    """
    API to process the document before embedding.
    """
    # Define the source directory containing document folders
    source_directory = os.path.join(base_directory, conf["s3"]["rootDirectory"])
    state = {"source_directory": source_directory, 'came_from': 'start',
             'app_name': app_name, 'document_names': document_names, 's3_event': s3_event}

    # Initialize the preprocessor
    preprocessor = DocumentPreprocessor(state)

    # Preprocess all documents
    state = await preprocessor.preprocess()
    os.path.join(source_directory, "documents_metadata.json")
    # PDFProcessor.reconstruct_pdf_from_metadata(metadata_file, source_directory)
    log.info("Preprocessing completed.")

    response = state['changes']
    # log.info(f"Response: {response}")

    return response


async def main():
    """
    Main function to orchestrate testing.
    """
    # Uncomment the function you want to test
    # await handle_preprocessing()
    # await handle_chunking()
    # await test_redis_operations()
    # response = await DocumentPreprocessor.get_document_details('LEGAL DATA & ANALYTICS PLATFORM')
    # pprint(response, indent=4)
    print("Hello ORB")


if __name__ == "__main__":
    # asyncio.run(main())
    uvicorn.run(app, host=conf["uvicorn"]["host"], port=conf["uvicorn"]["port"])