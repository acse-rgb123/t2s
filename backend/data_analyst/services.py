"""
Data analysis service using Gemini for insights and recommendations.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
from ..shared.gemini_base import GeminiBase, GeminiConfig


class DataAnalysisService:
    """Service for analyzing data and generating insights using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.gemini = GeminiBase(model_name=model_name)
        self.system_prompt = """You are a data analyst expert. Your role is to:

1. Analyze data patterns and trends
2. Generate insights and recommendations
3. Explain statistical findings in plain language
4. Suggest follow-up questions or analyses
5. Identify anomalies or interesting patterns

Always provide actionable insights and explain your reasoning. Use clear, non-technical language when possible, but be precise with statistical concepts."""
    
    def analyze_query_results(
        self,
        query_results: List[Dict[str, Any]],
        query_description: str,
        temperature: float = 0.4
    ) -> str:
        """
        Analyze SQL query results and provide insights.
        
        Args:
            query_results: Results from SQL query execution
            query_description: Description of what the query was trying to find
            temperature: Model temperature for analysis creativity
            
        Returns:
            Analysis and insights
        """
        # Convert results to a readable format
        if not query_results:
            results_summary = "No data returned from the query."
        else:
            # Create a summary of the data
            num_rows = len(query_results)
            columns = list(query_results[0].keys()) if query_results else []
            
            # Sample data (first few rows)
            sample_data = query_results[:5]
            
            results_summary = f"""
Data Summary:
- Number of rows: {num_rows}
- Columns: {', '.join(columns)}
- Sample data: {sample_data}
"""
        
        prompt = f"""
Query Description: {query_description}

{results_summary}

Please analyze this data and provide:
1. Key insights and patterns
2. Summary statistics if relevant
3. Recommendations or follow-up questions
4. Any anomalies or interesting findings
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    def suggest_followup_queries(
        self,
        original_query: str,
        results_summary: str,
        temperature: float = 0.6
    ) -> str:
        """
        Suggest follow-up queries based on analysis results.
        
        Args:
            original_query: The original SQL query or natural language question
            results_summary: Summary of the results from the analysis
            temperature: Model temperature for creativity in suggestions
            
        Returns:
            Suggested follow-up queries
        """
        prompt = f"""
Original Query: {original_query}
Results Summary: {results_summary}

Based on these results, suggest 3-5 follow-up questions or queries that would provide additional insights. Focus on:
1. Drilling down into interesting patterns
2. Comparing with different time periods or segments
3. Finding root causes or correlations
4. Exploring trends or forecasting opportunities
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=512)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    async def analyze_query_results_async(
        self,
        query_results: List[Dict[str, Any]],
        query_description: str,
        temperature: float = 0.4
    ) -> str:
        """Async version of analyze_query_results."""
        # Convert results to a readable format
        if not query_results:
            results_summary = "No data returned from the query."
        else:
            # Create a summary of the data
            num_rows = len(query_results)
            columns = list(query_results[0].keys()) if query_results else []
            
            # Sample data (first few rows)
            sample_data = query_results[:5]
            
            results_summary = f"""
Data Summary:
- Number of rows: {num_rows}
- Columns: {', '.join(columns)}
- Sample data: {sample_data}
"""
        
        prompt = f"""
Query Description: {query_description}

{results_summary}

Please analyze this data and provide:
1. Key insights and patterns
2. Summary statistics if relevant
3. Recommendations or follow-up questions
4. Any anomalies or interesting findings
"""
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return await self.gemini.generate_async(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )