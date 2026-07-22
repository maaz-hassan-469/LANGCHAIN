from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
langchain_docs = [
    Document(
        page_content="Document Loaders in LangChain are responsible for loading data from various sources like PDFs, Text files, Web pages, and Databases into standard Document objects.",
        metadata={
            "doc_id": 1,
            "topic": "Document Loaders",
            "category": "Data Ingestion",
            "module": "langchain_community",
            "difficulty": "Beginner"
        }
    ),
    Document(
        page_content="Text Splitters break down large documents into smaller, semantically meaningful chunks. SemanticChunker uses embedding models to detect topic shifts rather than hard character limits.",
        metadata={
            "doc_id": 2,
            "topic": "Text Splitting",
            "category": "Preprocessing",
            "module": "langchain_experimental",
            "difficulty": "Intermediate"
        }
    ),
    Document(
        page_content="Embeddings convert textual data into high-dimensional numerical vectors. Models like sentence-transformers/all-MiniLM-L6-v2 map semantic similarity into vector space.",
        metadata={
            "doc_id": 3,
            "topic": "Embeddings",
            "category": "Vector Representations",
            "module": "langchain_huggingface",
            "difficulty": "Intermediate"
        }
    ),
    Document(
        page_content="Vector Stores like ChromaDB store embeddings along with original document text and metadata, enabling high-speed similarity search and vector retrieval for RAG pipelines.",
        metadata={
            "doc_id": 4,
            "topic": "Vector Databases",
            "category": "Storage & Retrieval",
            "module": "langchain_chroma",
            "difficulty": "Intermediate"
        }
    ),
    Document(
        page_content="LangGraph extends LangChain by allowing developers to build stateful, multi-actor, and cyclic workflows (graphs) for advanced agentic decision-making and self-correcting RAG.",
        metadata={
            "doc_id": 5,
            "topic": "LangGraph",
            "category": "Agentic Workflows",
            "module": "langgraph",
            "difficulty": "Advanced"
        }
    )
]

vector_store=Chroma.from_documents(
    documents=langchain_docs,
    embedding=model
)

retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":2,"lamba_mult":1}
)


query="what is langchain"
result=retriever.invoke(query)
print(result)