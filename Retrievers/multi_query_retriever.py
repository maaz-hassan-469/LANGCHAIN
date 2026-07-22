from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

health_docs = [
    Document(
        page_content="Drinking 8 to 10 glasses of water daily keeps the body properly hydrated, aids digestion, enhances skin health, and supports optimal brain function.",
        metadata={
            "doc_id": 1,
            "topic": "Hydration",
            "category": "Physical Health",
            "importance": "High"
        }
    ),
    Document(
        page_content="A balanced diet rich in whole grains, leafy greens, lean proteins, and healthy fats provides essential vitamins and strengthens the immune system.",
        metadata={
            "doc_id": 2,
            "topic": "Nutrition",
            "category": "Diet",
            "importance": "High"
        }
    ),
    Document(
        page_content="Engaging in at least 30 minutes of moderate aerobic exercise like walking, running, or cycling daily reduces the risk of cardiovascular diseases.",
        metadata={
            "doc_id": 3,
            "topic": "Exercise",
            "category": "Physical Health",
            "importance": "High"
        }
    ),
    Document(
        page_content="Getting 7 to 9 hours of continuous sleep each night is critical for cellular repair, memory consolidation, hormone regulation, and mental clarity.",
        metadata={
            "doc_id": 4,
            "topic": "Sleep Hygiene",
            "category": "Recovery",
            "importance": "High"
        }
    ),
    Document(
        page_content="Practicing mindfulness, meditation, or deep breathing exercises helps lower cortisol levels, reducing anxiety and chronic psychological stress.",
        metadata={
            "doc_id": 5,
            "topic": "Mindfulness",
            "category": "Mental Health",
            "importance": "Medium"
        }
    ),
    Document(
        page_content="Limiting intake of refined sugars, artificial sweeteners, and ultra-processed foods lowers the risk of developing type 2 diabetes and metabolic disorders.",
        metadata={
            "doc_id": 6,
            "topic": "Sugar Intake",
            "category": "Diet",
            "importance": "High"
        }
    ),
    Document(
        page_content="Scheduling annual preventive medical checkups and routine blood work enables early detection and management of potential health issues.",
        metadata={
            "doc_id": 7,
            "topic": "Preventive Care",
            "category": "Medical Checkups",
            "importance": "Medium"
        }
    ),
    Document(
        page_content="Maintaining personal hygiene, such as frequent handwashing with soap for 20 seconds, significantly stops the transmission of infectious pathogens.",
        metadata={
            "doc_id": 8,
            "topic": "Hygiene",
            "category": "Preventive Health",
            "importance": "High"
        }
    ),
    Document(
        page_content="Getting 10 to 15 minutes of natural sunlight exposure early in the morning boosts Vitamin D synthesis, strengthening bone density and mood regulation.",
        metadata={
            "doc_id": 9,
            "topic": "Sunlight & Vitamin D",
            "category": "Wellness",
            "importance": "Medium"
        }
    ),
    Document(
        page_content="Avoiding prolonged sitting by taking short walking breaks every hour improves blood circulation, posture, and long-term metabolic health.",
        metadata={
            "doc_id": 10,
            "topic": "Ergonomics",
            "category": "Physical Health",
            "importance": "Medium"
        }
    )
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model=llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task="text-generation")

model=ChatHuggingFace(llm=llm)

vector_store=Chroma.from_documents(
    embedding=embeddings,
    documents=health_docs
)

multi_query_retriever=MultiQueryRetriever.fromllm(
    retriever=vector_store.as_retriever(search_kwargs={"k":5}),
    llm=model
)

result=multi_query_retriever.invoke("how to maintain health")

print(result)