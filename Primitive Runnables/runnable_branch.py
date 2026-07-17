from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation")
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(template="create a report on this {topic}",
                       input_variables=['topic'])

prompt2=PromptTemplate(template="summarize the topic {text}",
                       input_variables=['text'])


parser=StrOutputParser()

topic_chain=RunnableSequence(prompt1,model,parser)

count_chain=RunnableBranch(
    (lambda x:len(x.split())> 100,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
final_chain=topic_chain|count_chain

result=final_chain.invoke({"topic":"war"})
print(result)
