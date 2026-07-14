from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from  langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()


llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Flash',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content="Tell me about langchain")
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)