from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Flash',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)

chat_history=[]
while True:
    user_input=input("you:")
    chat_history.append(user_input)
    if user_input=="exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(result)
    print("AI:",result.content)

print(chat_history)