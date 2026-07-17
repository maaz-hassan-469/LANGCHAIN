from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation")
model=ChatHuggingFace(llm=llm)

