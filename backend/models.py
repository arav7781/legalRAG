from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Sequence
from langchain_core.messages import BaseMessage
from typing import Annotated
from langgraph.graph.message import add_messages

# Pydantic models
class PDFUploadRequest(BaseModel):
    pdf_url: HttpUrl
    collection_name: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    collection_name: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: dict = {}

# Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    collection_name: str