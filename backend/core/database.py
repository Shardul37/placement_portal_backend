import asyncpg
import os
from contextlib import asynccontextmanager

# We will use a connection pool, which is the recommended way
# to handle database connections in an async application.
_pool = None

@asynccontextmanager
async def get_db_connection():
    """Provides an asynchronous database connection from the pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call initialize_db_pool first.")
    
    conn = await _pool.acquire()
    try:
        yield conn
    finally:
        await _pool.release(conn)

async def initialize_db_pool():
    """Initializes the database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
        print("Database connection pool initialized.")

async def close_db_pool():
    """Closes the database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("Database connection pool closed.")