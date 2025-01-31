

from vertexai import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
import vertexai
import os

PROJECT_ID = "project-pathfinder-447802"
def get_corpus_name():
    with open('corpus_name.txt', 'r') as f:
        corpus_name = f.read()
    return corpus_name

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
response = rag_model.generate_content("what does OCR stand for?")
print(response.text)
