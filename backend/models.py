from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    answer: str
    source: Optional[str] = None