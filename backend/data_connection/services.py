"""
Data connection service with Gemini integration for schema analysis.
"""

from typing import Optional, Dict, Any, List
from ..shared.gemini_base import GeminiBase, GeminiConfig


class SchemaAnalysisService:
    """Service for analyzing database schemas and suggesting optimizations using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.gemini = GeminiBase(model_name=model_name)
        self.system_prompt = """You are a database schema expert. Your role is to:

1. Analyze database schemas and table relationships
2. Suggest optimal query strategies
3. Identify potential performance issues
4. Recommend schema improvements
5. Help users understand data structure

Provide clear explanations and practical recommendations for database optimization and query efficiency."""
    
    def analyze_schema(
        self,
        schema_info: Dict[str, Any],
        temperature: float = 0.3
    ) -> str:
        """
        Analyze database schema and provide recommendations.
        
        Args:
            schema_info: Dictionary containing schema information (tables, columns, relationships)
            temperature: Model temperature for analysis consistency
            
        Returns:
            Schema analysis and recommendations
        """
        prompt = f"""
Database Schema Information:
{schema_info}

Please analyze this schema and provide:
1. Overview of the data structure
2. Key relationships between tables
3. Potential query optimization opportunities
4. Suggestions for common query patterns
5. Any schema design recommendations
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    def suggest_query_optimization(
        self,
        sql_query: str,
        schema_context: str,
        temperature: float = 0.4
    ) -> str:
        """
        Suggest optimizations for a given SQL query.
        
        Args:
            sql_query: The SQL query to optimize
            schema_context: Relevant schema information
            temperature: Model temperature for optimization suggestions
            
        Returns:
            Query optimization suggestions
        """
        prompt = f"""
SQL Query to Optimize:
{sql_query}

Database Schema Context:
{schema_context}

Please provide:
1. Analysis of the current query performance characteristics
2. Specific optimization suggestions (indexes, query rewriting, etc.)
3. Alternative query approaches if applicable
4. Expected performance improvements
5. Any potential trade-offs to consider
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    def explain_query_plan(
        self,
        query_plan: str,
        sql_query: str,
        temperature: float = 0.2
    ) -> str:
        """
        Explain a database query execution plan in plain language.
        
        Args:
            query_plan: The database query execution plan
            sql_query: The original SQL query
            temperature: Model temperature for explanation consistency
            
        Returns:
            Plain language explanation of the query plan
        """
        prompt = f"""
SQL Query:
{sql_query}

Query Execution Plan:
{query_plan}

Please explain this query execution plan in plain language:
1. How the database will execute this query step by step
2. Which operations are most expensive
3. Whether the query is efficient or needs optimization
4. Key performance bottlenecks if any
5. Simple recommendations for improvement
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    async def analyze_schema_async(
        self,
        schema_info: Dict[str, Any],
        temperature: float = 0.3
    ) -> str:
        """Async version of analyze_schema."""
        prompt = f"""
Database Schema Information:
{schema_info}

Please analyze this schema and provide:
1. Overview of the data structure
2. Key relationships between tables
3. Potential query optimization opportunities
4. Suggestions for common query patterns
5. Any schema design recommendations
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return await self.gemini.generate_async(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )