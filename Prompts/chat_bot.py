from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Flash',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)

while True:
    user_input=input("you:")
    if user_input=="exit":
        break
    result=model.invoke(user_input)
    print(result.content)

