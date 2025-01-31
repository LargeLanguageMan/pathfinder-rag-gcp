from vertexai import rag
import vertexai
import os
from google.cloud import storage

PROJECT_ID = "project-pathfinder-447802"
vector_search_index_name = "projects/288327094814/locations/us-central1/indexes/4586419241221095424"
vector_search_index_endpoint_name = "projects/288327094814/locations/us-central1/indexEndpoints/3534582437625462784"
display_name = "test_corpus"
description = "Corpus Description"
paths = ["gs://pathfinder_poc_pdf/pdf_output/"]

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Configure embedding model (Optional)
embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-004"
)

# Configure Vector DB
vector_db = rag.VertexVectorSearch(
    index=vector_search_index_name, index_endpoint=vector_search_index_endpoint_name
)

def create_corpus():
    # Initialize GCS client
    storage_client = storage.Client()
    bucket_name = "pathfinder_poc_pdf"
    blob_name = "corpus/corpus_name.txt"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Check if corpus name already exists in GCS
    try:
        corpus_name = blob.download_as_text()
        return corpus_name
    except Exception:
        # Create new corpus if file doesn't exist
        corpus = rag.create_corpus(
            display_name=display_name,
            description=description
            # embedding_model_config=embedding_model_config,
            # vector_db=vector_db,
        )
        # Upload corpus name to GCS
        blob.upload_from_string(corpus.name)
        return corpus.name

transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(
        chunk_size=512,
        chunk_overlap=100,
    ),
)

def import_files():
    response = rag.import_files(
        corpus_name=create_corpus(),
        paths=paths,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=900,  # Optional
)
    print(f"Imported {response.imported_rag_files_count} files.")

import_files()



