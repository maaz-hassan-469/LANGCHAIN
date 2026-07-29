from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.messages import HumanMessage,AIMessage
import requests

#tool binding
@tool
def multiply(a:int,b:int)->int:
    """give 2 numbers a and b to this tool to return the product"""
    return a*b

llm=ChatOllama(
    model="llama3.2:1b",
    temperature=0.2
)

llm_with_tool=llm.bind_tools([multiply])

#tool calling

result=llm_with_tool.invoke("can you multiply 3 with 10 ").tool_calls[0]

print(llm_with_tool)

#tool execution
multiply.invoke(result.tool_calls[0])






