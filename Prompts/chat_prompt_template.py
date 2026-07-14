from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([('system','you are a helpful {domain}'),
                                  ('human','tell me about {topic}')])

prompt=chat_template.invoke({'domain':'cricket assistant','topic':'dusra'})

print(prompt)