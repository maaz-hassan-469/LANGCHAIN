from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

template=PromptTemplate(
    template="generate a report on this {topic}",
    input_variables=["topic"],
)

chain=template|model|parser

result=chain.invoke({"topic":"output parsers"})

print(result)