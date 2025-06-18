"""
LLM service for SQL query generation using Gemini.
"""

from typing import Optional, Dict, Any
from ..shared.gemini_base import GeminiBase, GeminiConfig


class SQLQueryGenerator:
    """Service for generating SQL queries from natural language using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.gemini = GeminiBase(model_name=model_name)
        self.system_prompt = """You are an expert SQL query generator. Your task is to convert natural language questions into valid SQL queries.

Rules:
1. Only return the SQL query, no explanations
2. Use proper SQL syntax and formatting
3. Consider database schema and relationships
4. Handle edge cases and invalid requests gracefully
5. If the request is unclear, ask for clarification

Database Schema Context:
- Always assume standard table and column naming conventions
- Use JOIN operations when multiple tables are involved
- Apply appropriate WHERE clauses for filtering
- Use proper aggregation functions when needed"""
    
    def generate_sql(
        self, 
        natural_language_query: str,
        schema_context: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Generate SQL query from natural language.
        
        Args:
            natural_language_query: User's question in natural language
            schema_context: Optional database schema information
            temperature: Model temperature (lower = more deterministic)
            
        Returns:
            Generated SQL query
        """
        prompt = f"Natural language query: {natural_language_query}"
        
        if schema_context:
            prompt = f"Database Schema:\n{schema_context}\n\n{prompt}"
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    async def generate_sql_async(
        self, 
        natural_language_query: str,
        schema_context: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """Async version of generate_sql."""
        prompt = f"Natural language query: {natural_language_query}"
        
        if schema_context:
            prompt = f"Database Schema:\n{schema_context}\n\n{prompt}"
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return await self.gemini.generate_async(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )