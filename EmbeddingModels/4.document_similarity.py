from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large')

docs=["virat is the captain of india",
      "gayle is the captain of westindies",
      "babar is the captain of pakistan",
      "cummins is the captin of australia"]

query="who is the captain of pakistan"

docs_embedding=embedding.embed_documents(docs)
query_embedding=embedding.embed_query(query)

scores=(cosine_similarity([query_embedding],docs_embedding))[0]

index,score=(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])
print(docs[index])
print("similarity score is: ", score)
