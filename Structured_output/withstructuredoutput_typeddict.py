from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional,Literal
from dotenv import load_dotenv

load_dotenv()

class review(TypedDict):

    key_themes:Annotated[list[str],'write down all the themes that are mentioned in the review']
    summary:Annotated[str,'A brief summary of the review']
    sentiment:Annotated[Literal["pos","neg"],'return sentiment of the review']
    pros:Annotated[Optional[list[str]],'write all the pros that are mentioned in the review']



llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task="text-generation")

model=ChatHuggingFace(llm=llm)

structured_model=model.with_structured_output(review)

result=structured_model.invoke("""Right out of the box, the first thing you notice is the premium feel. The matte finish looks fantastic and does a wonderful job of resisting fingerprint smudges, which was a huge issue on the previous generation's model. It has a reassuring weight to it—not so heavy that it becomes tedious to carry around all day, but substantial enough that it doesn't feel like a cheap plastic toy.

The buttons are incredibly tactile and give a satisfying, clicky feedback when pressed. My only minor complaint about the physical design is the placement of the secondary ports. They are located on the bottom left edge, which makes plugging in accessories a bit awkward if you are using it while it charges.""")

print(result)
print(result['summary'])
print(result['sentiment'])