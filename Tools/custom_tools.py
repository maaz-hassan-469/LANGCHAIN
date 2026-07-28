from langchain_core.tools import tool,StructuredTool
from pydantic import BaseModel,Field


# Method 1
@tool
def multiply(a:int,b:int) -> int:

    """desciption: Multiply two numbers """
    return a*b

result=multiply.invoke({"a":3,"b":2})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)

#Method 2

class MultiplyInput(BaseModel):
    a: int=Field(description="the first number to add")
    b: int=Field(description="the second number to add")

def multiply_func(a:int, b:int) ->int:
    return a*b

multiply_tool=StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="multiply two numbers",
    args_schema=MultiplyInput
)

result2=multiply_tool.invoke({"a":3,"b":2})

print(result2)

#method 3

class multplyTool(BaseModel):
    name:str="multiply"
    description:str="multiply two numbers"
    args_schema: type[BaseModel]=MultiplyInput

    def _run(self, a:int,b:int)->int:
        return a*b

multiply_tool=multplyTool()

result3=multiply_tool.invoke()
print (result3)



