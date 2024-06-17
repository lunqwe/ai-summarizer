import asyncio
from fastapi import FastAPI, HTTPException

from schemas import TextInput
from summarizer import Summarizer

app = FastAPI()


@app.post("/summarize")
async def summarize_endpoint(data: TextInput):
    try:
        summary = await asyncio.to_thread(Summarizer.summarize, data.text_input)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

    