
from fastapi import FastAPI, HTTPException

from schemas import TextInput
from summarizer import Summarizer

app = FastAPI()


@app.post("/summarize")
async def summarize(data: TextInput):
    summary = Summarizer.summarize(input=data.text_input)
    return {"summary": summary}
    
    

    