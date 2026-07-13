from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatmodel=ChatOpenAI(model="gpt-4",temperature=0)

result=chatmodel.invoke("what is the capital of pakistan")

print(result.content)

