from langchain_community.document_loaders import WebBaseLoader

url="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash"
loader=WebBaseLoader(url)

docs=loader.load()

print(docs[0].page_content)

# print(len(docs))
#similarly we can do it for csv loader like we did for text and pdf the main difference is that
#csv loader treat every row as individual document