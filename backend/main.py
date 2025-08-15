from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from core.agent import get_agent_response
from core.database import initialize_db_pool, close_db_pool
import os
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the lifespan of the application, initializing and closing the DB connection pool.
    """
    await initialize_db_pool()
    yield
    await close_db_pool()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/agent/chat")
async def chat_with_agent(request: QueryRequest):
    """
    Endpoint for the Streamlit frontend to send queries to the LangChain agent.
    """
    try:
        response = await get_agent_response(request.query)
        return {"response": response}
    except Exception as e:
        return {"response": f"An error occurred: {e}"}

@app.get("/")
def read_root():
    return {"message": "Placement AI Backend is running!"}