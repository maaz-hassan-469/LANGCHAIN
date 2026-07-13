from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=30)

result=embedding.embed_query("islamabad is the capital of pakistan")

print(str(result))

