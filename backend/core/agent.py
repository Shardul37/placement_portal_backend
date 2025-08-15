from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder # <-- Import MessagesPlaceholder
from core.tools.sql_tool import SQLTool, DB_SCHEMA
from core.tools.rag_tool import RAGTool
import os

# Define the tools the agent can use
tools = [SQLTool(), RAGTool()]

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# Create the prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are an AI assistant for a college placement portal. Your job is to answer user questions accurately. Use the tools provided to get information from the database.\n\n"
               f"Database schema:\n{DB_SCHEMA}\n\n"
               "**Important Rule:** If a company's `ctc_btech` or `gross_salary_btech` is NULL, it means they did not offer a job for B.Tech students. Probably they want Mtech, phd or Bdes students.You must inform the user of this when relevant.\n\n"
               "When querying the database, be flexible and use partial and case-insensitive matching for company names and roles. "
               "For example, if the user mentions 'Google', query using 'ILIKE '%Google%'' to find 'Google India Pvt Ltd'. "
               "Whenever you are giving any number for money value, make sure to mention the currency in the output and write the number with commas. "
               "Also, for names and locations, prefer using the ILIKE operator with wildcards ('%') instead of the '=' operator.\n\n"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") # <-- Add the scratchpad placeholder
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def get_agent_response(query: str):
    """
    Function to run the agent with a user query asynchronously.
    """
    # Use 'ainvoke' to run the agent in an async context
    response = await agent_executor.ainvoke({"input": query, "chat_history": []})
    return response["output"]