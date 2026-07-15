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

template1=PromptTemplate(
    template="generate a report on this {topic}",
    input_variables=["topic"],
)

template2=PromptTemplate(
    template="generate the summary of the following text {text}",
    input_variables=["text"]
)

chain=template1|model|parser|template2|model|parser
result=chain.invoke({"topic":"unemployement"})

print(result)

chain.get_graph().print_ascii()