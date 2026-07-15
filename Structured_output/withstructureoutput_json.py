from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

json_schema={
  "review_id": "rev_2026_99812x",
  "product": {
    "product_id": "prod_77321",
    "category": "Electronics",
    "name": "ApexWireless Pro Headphones",
    "sku": "APX-WRLS-PRO-BLK"
  },
  "author": {
    "user_id": "usr_44102",
    "username": "TechEnthusiast99",
    "is_verified_buyer": True,
    "reviewer_tier": "Top Contributor"
  },
  "metadata": {
    "submission_timestamp": "2026-07-15T01:45:22Z",
    "platform": "Mobile App (iOS)",
    "language": "en-US"
  },
  "ratings": {
    "overall_score": 4.2,
    "breakdown": {
      "build_quality": 5,
      "sound_performance": 5,
      "battery_life": 2.5,
      "value_for_money": 4
    }
  },
  "content": {
    "title": "Incredible sound and build, but terrible battery life",
    "text_body": "After using the ApexWireless Pro for 3 months, I have mixed feelings. The active noise cancellation and overall sound profile are elite—deep bass without muddiness. The aluminum frame feels premium and durable. However, the battery life is an absolute letdown. It claims 20 hours, but I barely get 8 hours before needing a charge. If you don't mind charging it daily, it's a stellar product.",
    "pros": [
      "Premium aluminum build quality",
      "Studio-grade sound with deep bass response",
      "Exceptional active noise cancellation (ANC)"
    ],
    "cons": [
      "Battery life falls significantly short of advertised specs",
      "The companion app requires too many permissions",
      "Charging cable included in the box is too short"
    ]
  },
  "engagement": {
    "helpful_votes": 342,
    "unhelpful_votes": 12,
    "total_comments": 4,
    "reported": False
  }
}


llm=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-V4-Pro',
                        task="text-generation")

model=ChatHuggingFace(llm=llm)

structured_model=model.with_structured_output(json_schema)

result=structured_model.invoke("""Right out of the box, the first thing you notice is the premium feel. The matte finish looks fantastic and does a wonderful job of resisting fingerprint smudges, which was a huge issue on the previous generation's model. It has a reassuring weight to it—not so heavy that it becomes tedious to carry around all day, but substantial enough that it doesn't feel like a cheap plastic toy.

The buttons are incredibly tactile and give a satisfying, clicky feedback when pressed. My only minor complaint about the physical design is the placement of the secondary ports. They are located on the bottom left edge, which makes plugging in accessories a bit awkward if you are using it while it charges.""")

print(result)
print(result["product"]["name"])
