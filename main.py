from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import os
import warnings
from builtins import input  # Add this import for terminal input

# Suppress all warnings and timeouts
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
warnings.filterwarnings('ignore')

# Create a RAG Corpus, Import Files, and Generate a response


PROJECT_ID = "project-pathfinder-447802"
display_name = "test_corpus"
paths = ["gs://pathfinder_poc_pdf"]
#["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]  
#  Supports Google Cloud Storage and Google Drive Links



# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create RagCorpus
# Configure embedding model, for example "text-embedding-004".
embedding_model_config = rag.EmbeddingModelConfig(
  publisher_model="publishers/google/models/text-embedding-004"
)

backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=backend_config,
)

# Use the corpus name directly from the created corpus
corpus_name = rag_corpus.name

# Optional: List corpora to verify
corpora = rag.list_corpora()


transformation_config = rag.TransformationConfig(
      chunking_config=rag.ChunkingConfig(
          chunk_size=512,
          chunk_overlap=100,
      ),
  )

rag.import_files(
    corpus_name,
    paths,
    transformation_config=transformation_config, # Optional
    max_embedding_requests_per_min=1000,  # Optional
)

user_query = input("Ask something about the files: ")

# List the files in the rag corpus
rag.list_files(corpus_name)

# Direct context retrieval
rag_retrieval_config=rag.RagRetrievalConfig(
    top_k=3,  # Optional
    filter=rag.Filter(vector_distance_threshold=0.5)  # Optional
)
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=corpus_name,
        )
    ],
    text=user_query,
    rag_retrieval_config=rag_retrieval_config,
)

print(response)

# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)
# Create a gemini model instance
rag_model = GenerativeModel(
        # Available models include:
    # - gemini-1.0-pro: Most capable model for complex tasks
    # - gemini-1.0-pro-vision: Multimodal model that can process text and images
    # - gemini-1.5-pro: Latest version with improved capabilities
    # - gemini-1.5-flash-001: Optimized for faster response times
    # Sources:
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
    # https://ai.google.dev/models/gemini
    model_name="gemini-1.5-flash-001", tools=[rag_retrieval_tool]
)

# Generate response
response = rag_model.generate_content(user_query)
print(response.text)
# Example response:
#   RAG stands for Retrieval-Augmented Generation.
#   It's a technique used in AI to enhance the quality of responses
# ...