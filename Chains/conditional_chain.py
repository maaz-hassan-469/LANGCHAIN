from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

llm1=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm1)

parser=StrOutputParser()

class Feedback(BaseModel):

    sentiment :Literal['positive','negative']=Field(description="give the sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="classify the following feedback into positive or negative\n {feedback}\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

classifier_chain=prompt1|model1|parser2

prompt2=PromptTemplate(
    template="write an appropriate response to this positive feedback {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="write an appropriate response to this negative feedback {feedback}",
    input_variables=["feedback"]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive",prompt2|model1|parser),
    (lambda x:x.sentiment=="negative",prompt3|model1|parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain=classifier_chain|branch_chain

print(chain.invoke({"feedback":"this is terrible phone"}))

