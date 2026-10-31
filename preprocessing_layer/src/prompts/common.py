from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field


class ImageDesc(BaseModel):
    description: str = Field(description="Detailed context or insights of the image content.")
    is_logo: bool = Field(description="Boolean True or False if the image is appears to be a Logo or icon")


image_out_format = JsonOutputParser(pydantic_object=ImageDesc).get_format_instructions()

image_context_prompt = ChatPromptTemplate.from_template("""
You are an Image Parser specializing in extracting the contents of an image.
You are currently working to extracting contents of an image in plain text so that it can be stored as a vector embeddings which would be used to write new documents.

# Task

1. Watch the image very carefully and describe its content in a verbose manner such that anyone reading it can recreate it.
2. Ensure to capture intricate details such as item, objects, their positioning relative to each other, colors etc.
   2.1 If the image is a table or has tables, you can add the tables exactly as they and where they appear in triplet format
3. Also, determine if the image is a sort of a logo or not - a logo can be a company logo, a technology logo, an icon etc.

Output Format:

You must strictly adhere to the JSON format provided in {format} and output nothing else. Ensure that your JSON is valid and follows the specified structure.
""").partial(format=image_out_format + ' - This JSON must be output as a single-line json without any truncations.')
