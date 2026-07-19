from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text="""from typing import TypedDict

class person(TypedDict):

    name: str
    age: int

new_person: person={'name':  'str','age':int}  

print(new_person)"""


splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=30,
    chunk_overlap=0
)

chunks=splitter.split_text(text)

print(chunks)