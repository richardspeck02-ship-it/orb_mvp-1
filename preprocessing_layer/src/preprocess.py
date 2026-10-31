import asyncio
import gc
import os
import json
import shutil
from datetime import datetime
from pdfplumber import open as open_pdf
from docx import Document
from PIL import Image
import io
from fpdf import FPDF
import base64
from langchain_core.messages import HumanMessage
import subprocess
import hashlib
import ast
import aioboto3
import urllib.parse
import time

from utils import log, llm, clean_json_data, config
import utils
from database.redis_operations import RedisOperations
from prompts.common import image_context_prompt


class DocumentPreprocessor:
    def __init__(self, state):
        app_name = state['app_name']
        if app_name:
            app_name = [app.strip() for app in str(app_name).split(',')]

        self.app_name = app_name

        document_name = state['document_names']
        if document_name:
            document_name = [doc.strip() for doc in str(document_name).split(',')]
        self.document_name = document_name

        s3_event = state['s3_event']
        if not s3_event:
            s3_event = {}
        else:
            s3_info = s3_event.get('Records', [{}])[0].get("s3", {})
            bucket_info = s3_info.get("bucket", {})
            if not bucket_info or bucket_info.get('name', 'example-bucket') == 'example-bucket':
                s3_event = {}

        self.s3_event = s3_event
        log.info(f'S3 event detected: {s3_event}')
        log.info(f'App name detected: {self.app_name}')
        log.info(f'Document name detected: {self.document_name}')

        self.state = state
        self.s3_client = None
        self.bucket = None
        self.event_type = None
        self.current_file = None
        self.version_timestamp = int(datetime.fromisoformat(
            s3_event.get('Records', [{}])[0].get("eventTimestamp", datetime.now().isoformat())).timestamp())

    @staticmethod
    async def get_document_details(app_name, document_names=None, include_metadata=False):
        """
        Function to retrieve and display data from Redis for documents.
        """
        redis_ops = RedisOperations()
        response = []

        # Ping Redis to ensure it's available
        try:
            await redis_ops.ping()
            log.info("Redis is available.")
        except RuntimeError as e:
            log.info(f"Error: {e}")
            return

        if document_names:
            document_names = [document_name.strip() for document_name in document_names.split(',')]

        # Retrieve all keys with the prefix
        prefix = redis_ops.prefix
        keys = await redis_ops.redis_client.keys(f"{prefix}:{app_name}*")

        if not keys:
            log.info(f"No data found for app: {app_name}")
            return response

        for key in keys:
            key = key.lstrip(f"{prefix}:")
            log.info(f"Key: {key}")
            data = await redis_ops.get_data(key)
            if not include_metadata:
                del data['metadata']

            if (document_names and data.get('document_name') in document_names) or not document_names:
                response.append(data)

        return response

    async def preprocess(self):
        """
        Preprocessing node for extracting metadata from documents.
        """
        state = self.state
        source_directory = state.get("source_directory")
        state['came_from'] = 'preprocess'

        if not any([self.s3_event, self.app_name, self.document_name]):
            state['changes'] = {"message": "No S3 event or input detected"}
            return state

        redis_ops = RedisOperations()
        # await redis_ops.clear_db()

        try:
            await self.prepare_files_and_directory(source_directory)
            log.info(f"Event type: {self.event_type}")
        except BlockingIOError:
            state['changes'] = {"message": "Event is not related to ORB Documents"}
            return state
        except NotImplementedError:
            state['changes'] = {"message": "Unrecognized S3 Event type"}
            return state

        all_documents_metadata = {}
        changes = {}
        log.info(f"Source directory: {source_directory}")

        unique_subdir = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shared_output_dir = os.path.join(utils.output_directory_s3, unique_subdir)
        os.makedirs(shared_output_dir, exist_ok=True)
        state["shared_output_dir"] = shared_output_dir
        log.info(f"Shared unique directory: {shared_output_dir}")

        for folder in os.listdir(source_directory):
            folder_path = os.path.join(source_directory, folder)
            log.info(f"Folder: {folder_path}")

            if os.path.isdir(folder_path) and (not self.app_name or folder in self.app_name):
                folder_metadata, folder_changes = await self.process_folder(folder_path, redis_ops)
                all_documents_metadata[folder] = folder_metadata
                changes[folder] = folder_changes

        output_file = DocumentPreprocessor.save_metadata_to_json(source_directory, all_documents_metadata)
        state["metadata_file"] = output_file
        state["changes"] = changes
        return state

    async def download_s3_object(self, object_key):
        object_key_temp = object_key.replace(f'{config["s3"]["rootDirectory"]}/', '')
        self.app_name = [os.path.dirname(object_key_temp)]
        source_directory = os.path.join(utils.output_directory, os.path.dirname(object_key_temp))
        os.makedirs(source_directory, exist_ok=True)
        file_name = os.path.basename(object_key)
        self.document_name = [os.path.splitext(file_name)[0]]
        local_path = os.path.join(source_directory, file_name)
        await self.bucket.download_file(object_key, local_path)
        log.info(f"Downloaded S3 object to {local_path}")

        return source_directory, file_name

    async def initialize_s3(self, bucket_name):
        session_sts = aioboto3.Session()
        role_arn = config['s3']["role_arn"]

        async with session_sts.client("sts") as sts_client:
            try:
                # Assume the role to get temporary credentials
                assumed_role = await sts_client.assume_role(
                    RoleArn=role_arn,
                    RoleSessionName="EKSClusterSession"
                )
                credentials = assumed_role['Credentials']

                access_key = credentials['AccessKeyId']
                secret_key = credentials['SecretAccessKey']
                session_key = credentials['SessionToken']

            except Exception as e:
                log.info(e)
                access_key, secret_key, session_key = None, None, None

        session = aioboto3.Session()
        async with session.resource(
            service_name='s3',
            region_name=config['s3']["region"],
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_key
        ) as s3_obj:
            self.bucket = await s3_obj.Bucket(bucket_name)

        async with session.client(
            "s3",
            region_name=config['s3']["region"],
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_key
        ) as s3_client:
            self.s3_client = s3_client

    async def prepare_files_and_directory(self, source_directory):
        """Validate the source directory."""
        s3_event = self.s3_event

        if s3_event:
            # Extract bucket and object key from S3 event
            record = s3_event.get("Records", [{}])[0]
            s3_info = record.get("s3", {})
            bucket_info = s3_info.get("bucket", {})
            object_info = s3_info.get("object", {})
            event_name = record.get('eventName', '')
            log.info(f"S3 Event Name: {event_name}")

            bucket_name = bucket_info.get("name")
            object_key = object_info.get("key")

            if not bucket_name or not object_key:
                raise ValueError("S3 event missing bucket name or object key.")

            # Decode object key if needed
            object_key = urllib.parse.unquote_plus(object_key)
            self.current_file = object_key

            # try:
            #     await self.initialize_s3(bucket_name)
            #     await self.download_s3_object(object_key)
            # except Exception as e:
            #     log.error(f"S3 error: {e}")
            if not object_key.startswith(f'{config["s3"]["rootDirectory"]}/'):
                raise BlockingIOError("Event is not related to ORB Documents")
            if event_name.startswith('ObjectCreated:'):
                self.event_type = 'create'
            elif event_name.startswith('ObjectRemoved:'):
                self.event_type = 'delete'
            else:
                raise NotImplementedError("Unrecognized S3 Event type")

            object_key = object_key.replace(f'{config["s3"]["rootDirectory"]}/', '')
            app_name = os.path.dirname(object_key)
            self.app_name = [app_name]
            file_name = os.path.basename(object_key)
            self.document_name = [os.path.splitext(file_name)[0]]
            log.info(f'App: {self.app_name}')
            log.info(f'Document: {self.document_name}')

        elif not source_directory:
            raise ValueError("Missing application name")
        else:
            os.makedirs(source_directory, exist_ok=True)
            self.event_type = 'create'

    async def process_folder(self, folder_path, redis_ops):
        """Process each folder and extract metadata."""
        folder_metadata = {"pdfs": [], "docx": [], "tds": None}
        folder_changes = {"change_type": "no change", "documents": []}

        if self.event_type == 'delete':
            folder_metadata, folder_changes = \
                await self.handle_delete_files(folder_path, redis_ops,
                                               folder_metadata, folder_changes)

            return folder_metadata, folder_changes

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if (file.endswith(".pdf") or file.endswith(".docx")) and \
                    not os.path.basename(file_path).startswith("~") and \
                    (not self.document_name or os.path.splitext(file)[0] in self.document_name):
                log.info(f"Processing file: {file_path}")
                metadata, is_tds, file_change = await self.process_file(file_path, folder_path,
                                                                        redis_ops)

                folder_metadata, folder_changes = \
                    DocumentPreprocessor.add_metadata_to_list(is_tds, folder_metadata, file,
                                                             metadata, file_change, folder_changes)

        # Determine overall folder change type
        if any(file["change_type"] in ["add", "update", "delete"] for file in folder_changes["documents"]):
            folder_changes["change_type"] = "update"

        return folder_metadata, folder_changes

    async def handle_delete_files(self, folder_path, redis_ops, folder_metadata, folder_changes):
        document = self.document_name[0]
        app_name = os.path.basename(folder_path)
        file_key = f"{app_name}:{document}"
        log.info(f"Deleting document {file_key} from Redis")
        await redis_ops.clear_db([file_key])
        metadata, is_tds, file_change = None, False, {"change_type": "delete", "document_name": document}
        file = document + os.path.splitext(self.current_file)[1]
        folder_metadata, folder_changes = \
            DocumentPreprocessor.add_metadata_to_list(is_tds, folder_metadata, file,
                                                     metadata, file_change, folder_changes)

        shared_path = os.path.join(self.state["shared_output_dir"], app_name, file)
        document_id = hashlib.md5(file_key.encode()).hexdigest()
        self.state['doc_path'] = shared_path
        self.state['doc_id'] = document_id

        await self.add_data_to_redis_stream(redis_ops)

        return folder_metadata, folder_changes

    @staticmethod
    def add_metadata_to_list(is_tds, folder_metadata, file,
                              metadata, file_change, folder_changes):
        if is_tds:
            folder_metadata["tds"] = {"document_name": file, "metadata": metadata}
        elif file.endswith(".pdf"):
            folder_metadata["pdfs"].append({"document_name": file, "metadata": metadata})
        elif file.endswith(".docx"):
            folder_metadata["docx"].append({"document_name": file, "metadata": metadata})

        folder_changes["documents"].append(file_change)

        return folder_metadata, folder_changes

    async def process_file(self, file_path, folder_name, redis_ops):
        """Process individual files and extract metadata."""
        file_key = f"{os.path.basename(folder_name)}:{os.path.basename(os.path.splitext(file_path)[0])}"
        existing_data = await redis_ops.get_data(file_key)

        checksum = DocumentPreprocessor.generate_pdf_checksum(file_path)
        log.info(f"Current checksum: {checksum}")
        log.info(f"Previous checksum: {existing_data.get('checksum', '')}")

        if existing_data and existing_data.get("checksum", '') == checksum:
            metadata, is_tds = existing_data.get('metadata', {}), \
                ast.literal_eval(existing_data.get('is_tds'))
            file_change = await self.update_redis(redis_ops, file_key,
                                                 folder_name, file_path, metadata,
                                                 is_tds, checksum)

            return metadata, is_tds, file_change

        if file_path.endswith(".pdf"):
            processor = PDFProcessor(file_path, folder_name, self.state)
        elif file_path.endswith(".docx"):
            processor = DOCXProcessor(file_path, folder_name, self.state)
        else:
            return None, False, None

        metadata = processor.process()

        is_tds = "Technical Design Specifications" in os.path.basename(file_path)

        file_change = await self.update_redis(redis_ops, file_key,
                                             folder_name, file_path, metadata,
                                             is_tds, checksum)

        return metadata, is_tds, file_change

    @staticmethod
    def generate_pdf_checksum(file_path: str) -> str:
        """
        Generate checksum for a PDF file using SHA-256 algorithm.

        :param file_path: Path to the PDF document
        :return: The SHA-256 checksum as a hex string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read the file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            return sha256_hash.hexdigest()

        except FileNotFoundError:
            log.error("Error: File not found.")
            return ""
        except Exception as e:
            log.error(f"An error occurred: {e}")
            return ""

    async def update_redis(self, redis_ops, file_key, folder_name,
                           file_path, metadata, is_tds, checksum):
        """Update Redis with metadata and calculate deltas."""
        existing_data = await redis_ops.get_data(file_key)
        delta = None
        change_type = "add"

        if existing_data and existing_data.get("checksum", '') != checksum:
            previous_metadata = existing_data.get("metadata", {})
            delta = {
                "added": {k: v for k, v in metadata.items() if k not in previous_metadata},
                "removed": {k: v for k, v in previous_metadata.items() if k not in metadata},
                "modified": {k: v for k, v in metadata.items() if k in previous_metadata and v != previous_metadata[k]},
            }
            if delta["added"] or delta["removed"] or delta["modified"]:
                change_type = "update"
            else:
                change_type = "no change"
        elif existing_data and existing_data.get("checksum", '') == checksum:
            change_type = "no change"

        app_name = os.path.basename(folder_name)
        document_name = os.path.basename(file_path)
        shared_path = os.path.join(self.state["shared_output_dir"], app_name, document_name)
        document_id = hashlib.md5(file_key.encode()).hexdigest()
        self.state['doc_path'] = shared_path
        self.state['doc_id'] = document_id

        redis_data = {
            "document_id": document_id,
            "app_name": app_name,
            "document_name": os.path.basename(os.path.splitext(file_path)[0]),
            "last_updated": datetime.now().isoformat() if change_type != 'no change'
            else existing_data.get('last_updated', datetime.now().isoformat()),
            "extension": os.path.splitext(file_path)[1].lstrip('.'),
            "is_tds": str(is_tds),
            "metadata": json.dumps(metadata),
            "delta": json.dumps(delta),
            "checksum": checksum,
            "latest_shared_path": shared_path
        }

        await redis_ops.insert_or_update_data(file_key, redis_data)
        redis_data['change_type'] = change_type
        redis_data['metadata'] = json.loads(redis_data['metadata'])
        if change_type != 'no change':
            await self.add_data_to_redis_stream(redis_ops, metadata=redis_data['metadata'], delta=delta)

        return redis_data

    async def add_data_to_redis_stream(self, redis_ops, metadata=None, delta=None):
        stream = config['redis']['embedQueueName']

        payload = {
            "images": [] if not metadata else metadata.get('images', []),
            "event_type": self.event_type,
            "timestamp": self.version_timestamp,
            "doc_path": self.state['doc_path'].replace('.docx', '.pdf'),
            "doc_id": self.state['doc_id'],
            "origin": "preprocess",
            "delta": delta
        }

        await redis_ops.add_to_stream(stream, payload)

        log.info(f"Added to Redis stream (showing keys): {payload.keys()}")

    @staticmethod
    def save_metadata_to_json(source_directory, all_documents_metadata):
        """Save the extracted metadata to a JSON file."""
        source_directory = os.path.join(utils.output_directory,
                                        os.path.relpath(source_directory,
                                                        os.path.join(utils.base_directory,
                                                                     config["s3"]["rootDirectory"])))
        output_file = os.path.join(source_directory, "documents_metadata.json")
        with open(output_file, "w+", encoding="utf-8") as f:
            json.dump(all_documents_metadata, f, indent=4, ensure_ascii=False)
        log.info(f"Metadata saved to {output_file}")
        return output_file


class DocumentProcessor:
    """Base class for processing documents."""

    def __init__(self, file_path, base_folder, state):
        self.file_path = file_path
        self.base_folder_initial = base_folder
        self.base_folder = os.path.join(utils.output_directory,
                                        os.path.relpath(base_folder,
                                                        os.path.join(utils.base_directory,
                                                                     config["s3"]["rootDirectory"])))
        self.state = state
        self.output_dir_final = os.path.join(self.state["shared_output_dir"],
                                             os.path.relpath(base_folder,
                                                             os.path.join(utils.base_directory,
                                                                          config["s3"]["rootDirectory"])))

        os.makedirs(self.output_dir_final, exist_ok=True)
        log.info(f'Final output folder: {self.output_dir_final}')
        log.info(f'Shared output folder: {self.state["shared_output_dir"]}')
        log.info(f'Base folder: {base_folder}')

    @staticmethod
    def move_files_to_shared_location(source, target):
        try:
            shutil.copyfile(source, target)
            log.info(f'Moved file: {source} to shared location')
        except PermissionError as e:
            log.error(f'PermissionError: {e}')

    @staticmethod
    def extract_image_context(image_path):
        """Extract context from images using an LLM."""

        try:
            # Convert image to bytes for LLM processing (if required)
            with open(image_path, "rb") as image_file:
                image_bytes = base64.b64encode(image_file.read()).decode('utf-8')

            # Send image to LLM for context extraction
            message = HumanMessage(
                content=[
                    {"type": "text", "text": image_context_prompt.format()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_bytes}"}
                    }
                ]
            )

            response = llm.invoke([message])
            context = json.loads(clean_json_data(response.content))
            del image_bytes
            # Extract and return the response text
            log.info(f"Image: {os.path.basename(image_path)}, Context: {context}")
            return context

        except Exception as e:
            log.error(f"Error extracting context for image: {e}")
            return None
    async def process(self):
        """Process the doocument. To be implemented by subclasses. """
        raise NotImplementedError("Subclasses must implement the `process` method.")

class PDFProcessor(DocumentProcessor):
    """Class for processing PDF documents."""

    def process(self):
        gc.collect()
        metadata = {"text": [], "tables": [], "images": []}

        DocumentProcessor. \
            move_files_to_shared_location(self.file_path,
                                         os.path.join(self.output_dir_final,
                                                      os.path.basename(self.file_path)))

        with open_pdf(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text()
                if text:
                    metadata["text"].append({
                        "page": page_num,
                        "content": text.strip(),
                        "formatting": {
                            "font_size": None,
                            "font_color": None,
                            "position": None
                        }
                    })

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    metadata["tables"].append({
                        "page": page_num,
                        "content": table,
                        "formatting": {
                            "border": "default",
                            "position": None
                        }
                    })

                # Extract images
                for image_index, image in enumerate(page.images):
                    image_data = image.get("stream")
                    if image_data:
                        log.info(f"Extracting image on page {page_num}, index {image_index}")
                        img, image_bytes = PDFProcessor. \
                            get_image_from_page(image_data, page, image, page_num, image_index)

                        output_dir = os.path.join(self.base_folder, "extracted_images")
                        os.makedirs(output_dir, exist_ok=True)
                        image_path = os.path.join(output_dir, f"page_{page_num}_image_{image_index}.png")
                        img.save(image_path)
                        img.close()  # <-- frees C buffers
                        del image_bytes  # <-- frees Python bytes
                        gc.collect()
                        image_directory_shared = os.path.join(self.output_dir_final, "extracted_images")
                        os.makedirs(image_directory_shared, exist_ok=True)
                        image_path_shared = os.path.join(image_directory_shared,
                                                         f"page_{page_num}_image_{image_index}.png")
                        log.info(f'Shared image path: {image_path_shared}')
                        try:
                            DocumentProcessor.move_files_to_shared_location(image_path, image_path_shared)
                        except (shutil.SameFileError, PermissionError, FileNotFoundError) as e:
                            log.info(f"Error: {e}")

                        metadata["images"].append({
                            "page": page_num,
                            "image_index": image_index,
                            "image_path": image_path_shared,
                            "context": None
                        })

        # for img_meta in metadata['images']:
        #     img_meta['context'] = DocumentProcessor.extract_image_context(img_meta['image_path'])
        gc.collect()
        return metadata

    @staticmethod
    def get_image_from_page(image_data, page, image, page_num, image_index):
        image_data_final = image_data.get_data()
        image_bytes = io.BytesIO(image_data_final)
        image_bytes.seek(0)

        try:
            img = Image.open(image_bytes)
            img.verify()
            img = Image.open(image_bytes)
        except Exception as e:
            log.error(f"First read attempt failed for image on page {page_num}, index {image_index}: {e}")
            dpi = 200
            page_image = page.to_image(resolution=dpi).original
            # Calculate scaling factor between PDF points (default 72 DPI) and the target DPI
            scale = dpi / 72.0

            pdf_height = page_image.height
            left = int(image["x0"] * scale)
            right = int(image["x1"] * scale)
            lower = int(image["y0"] * scale)
            upper = int(image["y1"] * scale)
            # Swap and flip y-coordinates
            pillow_upper = int(pdf_height - upper)
            pillow_lower = int(pdf_height - lower)
            bbox = (left, pillow_upper, right, pillow_lower)
            log.info(f"Image bounding box after: {bbox}")

            cropped_image = page_image.crop(bbox)
            img = cropped_image
            img.verify() if hasattr(img, 'verify') else None

        return img, image_bytes

    @staticmethod
    def reconstruct_pdf_from_metadata(metadata_file, output_directory, prefix="reconstructed_"):
        """
        Reconstruct a PDF using metadata JSON, replacing images with their context as text.

        Args:
            metadata_file (str): Path to the metadata JSON file.
            output_directory (str): Directory to save the reconstructed PDF.
            prefix (str): Prefix for the reconstructed PDF file name.
        """
        reconstructed_pdf_dir = os.path.join(output_directory, "reconstructed_pdfs")
        os.makedirs(reconstructed_pdf_dir, exist_ok=True)

        # Load metadata from JSON file
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)

            for app_name, app_data in metadata.items():
                # Process each PDF file in the metadata
                for pdf_data in app_data.get("pdfs", []):
                    pdf_file = pdf_data["document_name"]
                    metadata = pdf_data["metadata"]

                    # Reconstruct PDF using metadata
                    PDFProcessor.reconstruct_pdf_child(metadata,
                                                       os.path.join(reconstructed_pdf_dir,
                                                                    f"{prefix}{pdf_file}"))

    @staticmethod
    def reconstruct_pdf_child(metadata, output_path):
        """
        Reconstruct a PDF using metadata JSON, replacing images with their context as text.

        Args:
            metadata (dict): Metadata dictionary containing text, tables, and images.
            output_path (str): Path to save the reconstructed PDF.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        dejavu_font_path = os.path.join("fonts", "DejaVuSans.ttf")
        pdf.add_font("DejaVu", "", dejavu_font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        # Add text content
        for text in metadata["text"]:
            pdf.multi_cell(0, 10, text["content"])

        # Add tables
        for table in metadata["tables"]:
            pdf.set_font("DejaVu", size=10)

            for row in table["content"]:
                row = [PDFProcessor.remove_null(cell) for cell in row]
                pdf.cell(0, 10, ', '.join(row), ln=True)

        # Add image contexts
        for image in metadata["images"]:
            pdf.set_font("DejaVu", size=10)
            pdf.multi_cell(0, 10, f"Image Context: {image['context']}")

        # Save the reconstructed PDF
        pdf.output(output_path)
        log.info(f"Reconstructed PDF saved to {output_path}")

    @staticmethod
    def remove_null(data):
        if not data:
            return ""

        return str(data)


class DOCXProcessor(DocumentProcessor):
    """Class for processing DOCX documents."""

    @staticmethod
    def convert_docx_to_pdf(input_path, output_path=None):
        """
        Converts a .docx file to a PDF using LibreOffice.

        Args:
            input_path (str): Path to the input .docx file.
            output_path (str): Path to save the output PDF file. If None, saves in the same directory as the input file.

        Returns:
            str: Path to the generated PDF file.
        """
        # Set the output path
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".pdf"
        env = os.environ.copy()
        env["UNO_EXIT_ON_TERMINATION"] = "1"
        # env["SAL_USE_VCLPLUGIN"] = "gen"  # no GTK junk

        # LibreOffice command for conversion
        command = [
            "soffice",  # LibreOffice executable
            "--headless",  # Run in headless mode (no GUI)
            "--convert-to", "pdf",  # Convert to PDF
            "--outdir", f'{os.path.dirname(output_path)}',  # Output directory
            f'{input_path}'  # Input file
        ]

        try:
            # Run the LibreOffice command
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           env=env)
            log.info(f"PDF successfully created: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            log.error(f"Error during conversion: {e.stderr.decode().strip()}")

    def process(self):
        output_dir = os.path.join(self.base_folder, "Converted Docx")
        os.makedirs(output_dir, exist_ok=True)
        pdf_output_path = os.path.join(output_dir,
                                       os.path.basename(os.path.
                                                        splitext(self.file_path)[0]) + '.pdf')

        DOCXProcessor.convert_docx_to_pdf(self.file_path, pdf_output_path)

        DocumentProcessor. \
            move_files_to_shared_location(pdf_output_path,
                                         os.path.join(self.output_dir_final,
                                                      os.path.basename(pdf_output_path)))

        log.info(
            f'Moved converted PDF file: {os.path.basename(self.file_path)} to shared location {os.path.join(self.output_dir_final, os.path.basename(pdf_output_path))}')
        metadata = PDFProcessor(pdf_output_path, self.base_folder_initial, self.state).process()

        return metadata

    @staticmethod
    def _get_font_color(para):
        """Extract font color from a paragraph."""
        try:
            run = para.runs[0]
            if run.font.color and run.font.color.rgb:
                return str(run.font.color.rgb)
        except IndexError:
            pass
        return None

    def _extract_images(self):
        """Extract images from the DOCX file, convert them to PNG, and save them."""
        images_metadata = []
        output_dir = os.path.join(self.base_folder, "extracted_images")
        os.makedirs(output_dir, exist_ok=True)

        for rel in Document(self.file_path).part.rels.values():
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                try:
                    image = Image.open(io.BytesIO(image_data))
                    image = image.convert("RGBA")
                    image_name = os.path.splitext(os.path.basename(rel.target_ref))[0] + ".png"
                    image_path = os.path.join(output_dir, image_name)
                    image.save(image_path, format="PNG")

                    images_metadata.append({
                        "image_path": image_path,
                        "width": image.width,
                        "height": image.height,
                        "format": "PNG",
                        "context": None
                    })
                except Exception as e:
                    log.error(f"Error processing image {rel.target_ref}: {e}")

        return images_metadata
