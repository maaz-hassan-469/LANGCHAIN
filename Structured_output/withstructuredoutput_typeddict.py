from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

class review(TypedDict):

    summary:str
    sentiment:str


llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task="text-generation")

model=ChatHuggingFace(llm=llm)

structured_model=model.with_structured_output(review)

result=structured_model.invoke("""Disappointed, stopped working after two days
At first, it seemed fine, but on the second day of use, it just completely stopped turning on. I tried changing the batteries and checking the connection, but nothing worked. I will be returning this for a refund""")

print(result)
print(result['summary'])
print(result['sentiment'])