from typing import Optional, Sequence

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import storage
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import os
import warnings
from vertexai.language_models import TextEmbeddingModel
from vertexai.language_models import TextEmbeddingInput

# Suppress all warnings and timeouts
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
warnings.filterwarnings('ignore')

# empty object to store chunks
chunks = []

project_id = "project-pathfinder-447802"
location = "us" 
processor_id = "3b6a24e6b08c4d47" 
processor_version = "rc" # Refer to https://cloud.google.com/document-ai/docs/manage-processor-versions for more information
# file_path = "gs://pathfinder_poc_pdf/pdf_output/"
file_path = "gs://pathfinder_poc_pdf/test_small/"

# file_path = "gs://pathfinder_poc_pdf/pdf_output/page_16.pdf"
# file_path = "https://storage.cloud.google.com/pathfinder_poc_pdf/pdf_output/page_1.pdf"
# file_path = "./pdf_output_page_16.pdf"
mime_type = "application/pdf" # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types



# def process_document(
#     project_id: str,
#     location: str,
#     processor_id: str,
#     processor_version: str,
#     file_path: str,
#     mime_type: str,
#     process_options: Optional[documentai.ProcessOptions] = None,
# ) -> documentai.Document:
#     # You must set the `api_endpoint` if you use a location other than "us".
#     client = documentai.DocumentProcessorServiceClient(
#         client_options=ClientOptions(
#             api_endpoint=f"{location}-documentai.googleapis.com"
#         )
#     )
    
#     name = client.processor_version_path(
#         project_id, location, processor_id, processor_version
#     )

#     # Handle both GCS and local files
#     if file_path.startswith('gs://'):
#         # Parse GCS URI
#         bucket_name = file_path.split('/')[2]
#         prefix = '/'.join(file_path.split('/')[3:])
        
#         # Initialize storage client
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(bucket_name)
        
#         # List all PDF files in the directory
#         blobs = bucket.list_blobs(prefix=prefix)
#         documents = []
        
#         for blob in blobs:
#             if blob.name.endswith('.pdf'):
#                 print(f"Processing: gs://{bucket_name}/{blob.name}")
#                 # Download to memory
#                 image_content = blob.download_as_bytes()
                
#                 # Configure the process request
#                 request = documentai.ProcessRequest(
#                     name=name,
#                     raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
#                     process_options=process_options,
#                 )

#                 result = client.process_document(request=request)
#                 documents.append(result.document)
                
#         return documents
#     else:
#         # Local file
#         with open(file_path, "rb") as image:
#             image_content = image.read()

#         # Configure the process request
#         request = documentai.ProcessRequest(
#             name=name,
#             raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
#             process_options=process_options,
#         )

#         result = client.process_document(request=request)
#         return [result.document]

# # process_document_layout_sample(project_id, location, processor_id, processor_version, file_path, mime_type)
# documents = process_document(project_id, location, processor_id, processor_version, file_path, mime_type)
# chunks.extend(documents)


#from chunks extract document_layout.blocks.text_block.text
# def extract_text(chunks):
#     sentences = []
#     for document in chunks:
#         for layout in document.document_layout.blocks:
#             if layout.text_block:
#                 sentences.append(layout.text_block.text)
#     return sentences

def read_from_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

sentences = read_from_file('sentences.txt')

def generate_text_embeddings(sentences) -> list: 
    vertexai.init(project=project_id, location="us-central1")
    embedding_model_config = rag.EmbeddingModelConfig(publisher_model="publishers/google/models/text-embedding-004")
    embeddings = embedding_model_config.get_embeddings(sentences)
    print(embeddings)
    # Example response:
    # [[0.006135190837085247, -0.01462465338408947, 0.004978656303137541, ...], [0.1234434666, ...]],
    return [embedding.values for embedding in embeddings]

vectors = generate_text_embeddings(sentences)
print(vectors)
