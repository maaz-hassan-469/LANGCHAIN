
from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""Fine-tuning an existing model involves taking a model, such as an Amazon Titan, Mistral, or Llama 
model, and then adapting the model to your custom data. There are various techniques for fine-tuning, most of which involve modifying only a few parameters instead of modifying all of the parameters in the model. This is called parameter-efficient fine-tuning. There are two primary methods for fine-tuning:• Supervised fine-tuning uses labeled data and helps you train the model for a new kind of task. For example, if you wanted to generate a report based on a PDF form, then you might have to teach the model how to do that by providing enough examples."""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks=splitter.split_text(text)

print(chunks)
print(len(chunks))