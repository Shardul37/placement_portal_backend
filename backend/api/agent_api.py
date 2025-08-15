from fastapi import APIRouter
from pydantic import BaseModel
from core.agent import handle_query

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/chat")
def chat_with_agent(request: QueryRequest):
    """
    Endpoint for the Streamlit frontend to send queries to the agent.
    """
    response = handle_query(request.query)
    return response