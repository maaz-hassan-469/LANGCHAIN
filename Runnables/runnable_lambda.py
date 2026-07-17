from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation")
model=ChatHuggingFace(llm=llm)

def count(text):
    return len(text.split())


prompt1=PromptTemplate(template="create a joke on this {topic}",
                       input_variables=['topic'])

parser=StrOutputParser()

joke_chain=RunnableSequence(prompt1,model,parser)

count_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "count":RunnableLambda(count)}
)

final_chain=joke_chain|count_chain

result=final_chain.invoke({"topic":"coding"})
print(result)