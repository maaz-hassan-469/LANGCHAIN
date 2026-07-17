from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation")
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(template="give me joke on this {topic}",
                       input_variables=['topic'])

parser=StrOutputParser()

prompt2=PromptTemplate(template="give me explanation on the joke {text}",
                       input_variables=['text'])


first_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explanation":RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(first_chain,parallel_chain)

result=final_chain.invoke({"topic":"coding"})

print(result)