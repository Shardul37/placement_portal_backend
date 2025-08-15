from langchain.tools import BaseTool
from core.database import get_db_connection

class SQLTool(BaseTool):
    name: str = "SQL_Query_Executor"
    description: str = (
        "This tool executes SQL queries on the placements database. "
        "Use this tool for questions that require filtering, counting, aggregating, or comparing numerical data. "
        "Examples: 'How many companies visited in Phase 1?', 'What is the average CTC for software roles?', "
        "'Which companies offer jobs in Mumbai?', 'Show me all companies with CTC above 15 lakhs', "
        "'Companies that visited in Phase 2 vs Phase 1', 'Highest paying companies by location'."
    )

    async def _arun(self, query: str) -> str:
        """
        Executes a SQL query asynchronously and returns the results.
        """
        async with get_db_connection() as conn:
            try:
                results = await conn.fetch(query)
                
                # Format the results into a human-readable string
                return f"Query executed successfully. Results: {results}"
            except Exception as e:
                return f"Error executing SQL query: {e}"

    def _run(self, query: str) -> str:
        raise NotImplementedError("This tool does not support synchronous calls.")

# We will provide the agent with the database schema in the prompt to help it generate
# accurate queries.
DB_SCHEMA = """
Table: placements
Columns:
- company_name (TEXT)
- job_role (TEXT)
- job_location (TEXT[])
- gross_salary_btech (BIGINT)
- ctc_btech (BIGINT)
- currency (TEXT)
- job_requirements (TEXT[])
- company_info (TEXT[])
- additional_details (TEXT[])
- phase (INT)
"""
