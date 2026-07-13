from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=30)
docs=["rohit is the captain of india"
      "babar is the captain of pakistan"
      "morgan is the captain of england"
      "gayle is the captain of westindies"]

result=embedding.embed_documents(docs)

print(str(result))
