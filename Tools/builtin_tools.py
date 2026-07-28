from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
search_tool=DuckDuckGoSearchRun()
shell=ShellTool()


result=search_tool.invoke("latest news of today")
result2=shell.invoke("whoami")

print(result)
print(result2)