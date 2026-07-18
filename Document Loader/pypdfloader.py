from langchain_community.document_loaders import PyPDFLoader

file_path=r"d:\LANGCHAIN\Document Loader\PDF.pdf"
loader=PyPDFLoader(file_path)

docs=loader.load()

print(docs[1])
# print(len(docs))