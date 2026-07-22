from langchain_community.retrievers import WikipediaRetriever
import os

query="economy of pakistan"

retriever=WikipediaRetriever(top_k_results=2,
                             lang="en")

docs=retriever.invoke(query)

print(docs)
# for i,doc in enumerate(docs):
#     print(f"\n---Result{i+1}---")
#     print(f"Content:\n{doc.page_content}")
