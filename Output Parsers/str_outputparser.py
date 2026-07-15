from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="write a 5 line summary on {text}",
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'balckhole'})

result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result})

result2=model.invoke(prompt2)

print(result2.content)