from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag_core import get_rag_response

app = FastAPI(title="API Zélia - Assistente Acadêmica")

class Message(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []

@app.post("/perguntar")
def perguntar_ao_manual(request: QuestionRequest):
    try:
        history_dict = [{"role": msg.role, "content": msg.content} for msg in request.history]
        answer = get_rag_response(request.query, history_dict)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))