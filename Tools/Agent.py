from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
#Now langchain create react_agent_replace with langgraph
from langchain.agents import create_react_agent ,AgentExecutor
from langchain import hub

search_tools=DuckDuckGoSearchRun()

llm=ChatOllama(
    model="llama3.2:1b",
    temperature=0.2
)

prompt=hub.pull("hwchase17/react")

agent=create_react_agent(
    llm=llm,
    tools=[search_tools],
    prompt=prompt
)


agent_executor=AgentExecutor(
    agent=agent,
    tools=[search_tools],
    verbose=True
)

response=agent_executor.invoke({"input":"3 ways to reach islamabad from lahore"})

print(response)