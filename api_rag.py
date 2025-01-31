from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import os
import warnings
from flask import Flask, request, jsonify

# Suppress all warnings and timeouts
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
warnings.filterwarnings('ignore')

app = Flask(__name__)

PROJECT_ID = "project-pathfinder-447802"
display_name = "test_corpus"
paths = ["gs://pathfinder_poc_pdf"]
#["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]  
#  Supports Google Cloud Storage and Google Drive Links



# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

@app.route('/rag', methods=['POST'])
def process_rag():
    try:
        data = request.get_json()
        user_query = data.get('user_query')
        
        if not user_query:
            return jsonify({'error': 'user_query is required'}), 400

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

        # Direct context retrieval
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=3,  # Optional
            filter=rag.Filter(vector_distance_threshold=0.5)  # Optional
        )

        # Get contexts
        contexts_response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,
                )
            ],
            text=user_query,
            rag_retrieval_config=rag_retrieval_config,
        )

        # Enhance generation with RAG
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
            model_name="gemini-1.5-flash-001", 
            tools=[rag_retrieval_tool]
        )

        # Generate response
        response = rag_model.generate_content(user_query)
        
        # Prepare the response
        # contexts = []
        # for context in contexts_response.contexts:
        #     contexts.append({
        #         'text': context.text,
        #         'source': context.source
        #     })

        return jsonify({
            # 'contexts': contexts,
            'generated_response': response.text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)