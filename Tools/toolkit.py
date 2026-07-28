from langchain_core.tools import tool,StructuredTool


# Method 1
@tool
def multiply(a:int,b:int) -> int:

    """desciption: Multiply two numbers """
    return a*b

def add(a:int,b:int) -> int:

    """desciption: add two numbers """
    return a+b


class mathtoolkit:
    def get_tool(self):
        return [add,multiply]


toolkit=mathtoolkit()

tools=toolkit.get_tool()
for tool in tools:
    print(tool.name, "->" , tool.desciption)

    