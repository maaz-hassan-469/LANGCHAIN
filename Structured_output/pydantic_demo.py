from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str='maaz'
    email:EmailStr="abc@gmail.com"
    age:Optional[str]=32
    cgpa:float=Field(gt=0,lt=10,default=5,description="a decimal value representing the cgpa of student")

new_student={"age":32}

student=Student(**new_student)

print(Student)
