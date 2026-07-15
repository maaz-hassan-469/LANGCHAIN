from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template="give me the name ,age and city of some fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instruction()}
    )

prompt=template.invoke({})

result=model.invoke(prompt)

final_result=parser.parse(result.content)

print(final_result)