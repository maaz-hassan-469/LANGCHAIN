from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.output_parsers import ResponseSchema, StructuredOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str=Field(description="name of the fictional perosn"),
    age:  int=Field(gt=0 ,description="age of that person"),
    city: str=Field(description="city of that person")

parser=PydanticOutputParser(pydantic_object="person")

template=PromptTemplate(
    template='generate the name,age and city of a fictional {place} person\n {format_instruction}',
    input_variables=['place'],
    partial_variables={'formar_instruction':parser.get_format_instructions()}
)

chain=template|model|parser

result=chain.invoke({"place":"pakistan"})
print(result)