from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id='deepseek-ai/DeepSeek-V4-Pro',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0,
        max_new_tokens=100
    )
)
model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the capital of pakistan")

print(result.content)