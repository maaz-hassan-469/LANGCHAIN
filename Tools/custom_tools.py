from langchain_core.tools import tool

@tool
def multiply(a:int,b:int) -> int:

    """desciption: Multiply two numbers """
    return a*b

result=multiply.invoke({"a":3,"b":2})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)


