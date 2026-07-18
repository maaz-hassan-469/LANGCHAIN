from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation")
model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

file_path=r"d:\LANGCHAIN\Document Loader\text.txt"
loader=TextLoader(file_path,
                  encoding='utf-8')

docs=loader.load()

prompt=PromptTemplate(template="summarize this document {docs}",
                      input_variables=["docs"])

chain=prompt|model|parser

result=chain.invoke({"docs":docs})
print(result)

# print(docs[0])
# print(docs[0].metadata)
# print(docs[0].page_content)
# print(len(docs))