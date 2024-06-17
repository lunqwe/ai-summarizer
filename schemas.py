from pydantic import BaseModel
from enum import Enum

class TextInput(BaseModel):
    text_input: str