# backend/core/tools/rag_tool.py
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector  # <-- Corrected import
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# The actual RAG implementation
class RAGTool(BaseTool):
    name: str = "RAG_Search_Tool"
    description: str = (
        "This tool performs semantic search on job descriptions, company information, and requirements. "
        "Use this tool for questions about: job skills and technologies required, company culture and work environment, "
        "specific job responsibilities, qualification requirements, company background and history, "
        "work policies (WFH, bonds, training), selection processes, and any text-based information. "
        "Examples: 'What skills does Google look for?', 'Tell me about Microsoft's work culture', "
        "'Which companies require Python skills?', 'What is the selection process for TCS?'."
    )

    async def _arun(self, query: str) -> str:
        # 1. Initialize the LLM and Embedding Model
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

        # 2. Connect to the vector store (pgvector) with the correct parameter
        vector_store = PGVector(
            connection_string=os.getenv("DATABASE_URL"),
            embedding_function=embeddings,
            collection_name="job_embeddings",  # <-- Corrected parameter name
        )

        # 3. Perform a similarity search (using the async version)
        docs = await vector_store.asimilarity_search(query, k=5)
        retrieved_text = "\n\n".join([doc.page_content for doc in docs])

        # 4. Use the retrieved documents to generate a final answer
        prompt_template = PromptTemplate.from_template(
            "Based on the following context, answer the user's question. If you don't know the answer, say so. "
            "\n\nContext:\n{context}\n\nQuestion:\n{question}\n"
        )
        chain = prompt_template | llm
        
        final_response = chain.invoke({"context": retrieved_text, "question": query})

        return final_response.content

    def _run(self, query: str) -> str:
        raise NotImplementedError("RAGTool does not support synchronous calls.")
