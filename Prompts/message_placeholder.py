from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,SystemMessage

chat_template=ChatPromptTemplate([('system','you are a helpful assistance'),
                                  MessagesPlaceholder(variable_name='chat_history'),
                                  ('human','{query}')])

chat_history=[]

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
prompt=chat_template.invoke({'chat_history':'chat_history','query':HumanMessage(content='where is my refund?')})