from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation")
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(template="generate a tweet on this {topic}",
                       input_variables=['topic'])

prompt2=PromptTemplate(template="create a post for linkdin on this {topic}",
                       input_variables=['topic'])

parser=StrOutputParser()

parallel_chains=RunnableParallel({
    "tweet":RunnableSequence(prompt1,model,parser),
    "linkdin":RunnableSequence(prompt2,model,parser)}
)

result=parallel_chains.invoke({"topic":"runnable"})
print(result["tweet"])

