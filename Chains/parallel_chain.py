from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
llm2=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)
model1=ChatHuggingFace(llm=llm1)
model2=ChatHuggingFace(llm=llm2)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="generate a short and simple notes on this {text}",
    input_variables=["text"],
)

prompt2=PromptTemplate(
    template="generate the quiz of the following text {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="merge the provided notes and quiz into a single documents notes-> {notes} and {quiz}",
    input_variables=["notes","quiz"]
)

parallel_chain=RunnableParallel({
"notes":prompt1|model1|parser,
"quiz":prompt2|model2|parser
})

merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain

text=""". Large Language Models (Base LLMs)
Definition: A Base LLM is the raw, foundational AI model that has been trained on massive amounts of internet text. Its primary objective is simple: predict the next most logical word in a sentence.

It doesn't actually understand that it is talking to a human. It just acts like the world’s most advanced autocomplete.

The Autocomplete Analogy
If you start typing an text message on your phone and just keep tapping the suggested next words, your phone creates a sentence based on patterns. A Base LLM does this on a massive, highly intelligent scale.

Example of Base LLM Behavior
If you give a Base LLM a prompt like:

Your Prompt: "Write a recipe for chocolate chip cookies."

Base LLM Output: "Ingredients needed for chocolate chip cookies. Step 1: Preheat the oven. Step 2..."

However, look what happens if you ask it a question:

Your Prompt: "What is the capital of France?"

Base LLM Output: "What is the capital of Spain? What is the capital of Italy? What is the capital of Germany?"

Why? Because on the internet, geography quizzes often list questions one after another. The LLM thinks you are starting a list of questions, so it autocompletes the next logical question instead of answering you!"""
result=chain.invoke({"text":text})

print(result)