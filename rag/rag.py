from vertexai import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
import vertexai
import os
from google.cloud import storage

PROJECT_ID = "project-pathfinder-447802"

def get_corpus_name():
    storage_client = storage.Client()
    bucket_name = "pathfinder_poc_pdf"
    blob_name = "corpus/corpus_name.txt"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text().strip()

corpus_name = get_corpus_name()
print(corpus_name)


vertexai.init(project=PROJECT_ID, location="us-central1")
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,  # Optional
    filter=rag.Filter(vector_distance_threshold=0.5)  # Optional
)
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=corpus_name,
        )
    ],
    rag_retrieval_config=rag_retrieval_config,
    text="What is the purpose of the document?",
)
print("*************************************************")
print("this is the retrieved documents based on user query")
print(response)
print("*************************************************")
vertexai.init(project=PROJECT_ID, location="us-central1")

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

rag_model = GenerativeModel(
    model_name="gemini-1.5-flash-001", tools=[rag_retrieval_tool]
)
response = rag_model.generate_content("give me a list of all data subject rights, then summarise them. if possible tell me on what page this is on ")
print(response.text)
