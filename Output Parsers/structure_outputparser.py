from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task='text-generation')
model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name="fact 1",description="fact 1 about the topic"),
    ResponseSchema(name="fact 1",description="fact 1 about the topic"),
    ResponseSchema(name="fact 1",description="fact 1 about the topic"),
]
parser=StructuredOutputParser.from_response_scehmas(schema)
template=PromptTemplate(
    template="give 3 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

prompt=template.invoke({"topic":'black hole'})

result=model.invoke(template)

final_result=parser.parse(result.content)

print(final_result)
