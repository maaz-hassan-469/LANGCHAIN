from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
player_docs = [
    Document(
        page_content="Babar Azam is a top-order batter known for his exquisite cover drive and consistent scoring across all formats in international cricket.",
        metadata={
            "player_id": 101,
            "name": "Babar Azam",
            "country": "Pakistan",
            "role": "Batter",
            "batting_style": "Right-hand bat",
            "bowling_style": "Right-arm offbreak",
            "matches_played": 300,
            "category": "International"
        }
    ),
    Document(
        page_content="Shaheen Shah Afridi is a premier left-arm fast bowler known for generating lethal swing and early breakthroughs in the opening overs.",
        metadata={
            "player_id": 102,
            "name": "Shaheen Afridi",
            "country": "Pakistan",
            "role": "Bowler",
            "batting_style": "Left-hand bat",
            "bowling_style": "Left-arm fast-medium",
            "matches_played": 180,
            "category": "International"
        }
    ),
    Document(
        page_content="Rashid Khan is a world-class leg-spin bowler known for his rapid arm action, unpickable googly, and tight economy rates in T20 leagues.",
        metadata={
            "player_id": 103,
            "name": "Rashid Khan",
            "country": "Afghanistan",
            "role": "Bowler",
            "batting_style": "Right-hand bat",
            "bowling_style": "Right-arm legbreak",
            "matches_played": 250,
            "category": "International"
        }
    ),
    Document(
        page_content="Ben Stokes is an aggressive all-rounder capable of turning matches single-handedly with crucial lower-order runs, seam bowling, and high-energy fielding.",
        metadata={
            "player_id": 104,
            "name": "Ben Stokes",
            "country": "England",
            "role": "All-rounder",
            "batting_style": "Left-hand bat",
            "bowling_style": "Right-arm fast-medium",
            "matches_played": 260,
            "category": "International"
        }
    ),
    Document(
        page_content="Shadab Khan is a dynamic leg-spinning all-rounder who contributes heavily in middle-overs bowling and provides power-hitting in the lower middle-order.",
        metadata={
            "player_id": 105,
            "name": "Shadab Khan",
            "country": "Pakistan",
            "role": "All-rounder",
            "batting_style": "Right-hand bat",
            "bowling_style": "Right-arm legbreak",
            "matches_played": 210,
            "category": "International"
        }
    )
]

vector_store=Chroma(
    embedding_function=model,
    persist_directory='chroma_db',
    collection_name='sample'
)

#vector_store.add_documents(player_docs)

# print(vector_store.get(include=['embeddings','documents','metadatas']))


# result=vector_store.similarity_search(
#     query="which one of them is batter?",
#     k=1
# )

result=vector_store.similarity_search(
    query="",
    filter={"name":"Babar Azam"}
)

print(result)

# print("final answer: ",result)