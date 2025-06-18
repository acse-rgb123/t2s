"""
Spider 2 specific service for text-to-SQL operations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from .bigquery_client import BigQueryClient
from ..shared.gemini_base import GeminiBase, GeminiConfig


class Spider2Service:
    """
    Service for working with the Spider 2 dataset for text-to-SQL benchmarking.
    
    Spider 2 is a large-scale cross-domain text-to-SQL dataset that contains
    databases from various domains and complex SQL queries.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        service_account_path: Optional[str] = None,
        gemini_model: str = "gemini-1.5-pro"
    ):
        """
        Initialize Spider 2 service.
        
        Args:
            project_id: Google Cloud project ID
            service_account_path: Path to service account JSON
            gemini_model: Gemini model for SQL generation
        """
        self.bigquery_client = BigQueryClient(
            project_id=project_id,
            service_account_path=service_account_path
        )
        self.gemini = GeminiBase(model_name=gemini_model)
        
        # Spider 2 specific configuration
        self.spider2_project = "spider2-public-data"
        self.spider2_dataset = "spider2_1_0"
        
        # Cache for schema information
        self._schema_cache = {}
    
    def get_available_databases(self) -> List[str]:
        """
        Get list of available databases in Spider 2 dataset.
        
        Returns:
            List of database names
        """
        try:
            tables = self.bigquery_client.list_spider2_tables()
            # Extract unique database names from table names
            databases = set()
            for table in tables:
                if '_' in table:
                    db_name = table.split('_')[0]
                    databases.add(db_name)
            return sorted(list(databases))
        except Exception as e:
            raise Exception(f"Failed to get Spider 2 databases: {str(e)}")
    
    def get_database_schema(self, database_name: str) -> Dict[str, Any]:
        """
        Get schema for a specific database in Spider 2.
        
        Args:
            database_name: Name of the database
            
        Returns:
            Database schema information
        """
        if database_name in self._schema_cache:
            return self._schema_cache[database_name]
        
        try:
            # Get all tables for this database
            all_tables = self.bigquery_client.list_spider2_tables()
            db_tables = [t for t in all_tables if t.startswith(f"{database_name}_")]
            
            schema_info = {
                "database_name": database_name,
                "tables": {},
                "table_count": len(db_tables)
            }
            
            for table_name in db_tables:
                table_schema = self.bigquery_client.get_table_schema(table_name)
                schema_info["tables"][table_name] = table_schema
            
            # Cache the schema
            self._schema_cache[database_name] = schema_info
            return schema_info
            
        except Exception as e:
            raise Exception(f"Failed to get database schema for {database_name}: {str(e)}")
    
    def generate_sql_for_database(
        self,
        natural_language_query: str,
        database_name: str,
        temperature: float = 0.3
    ) -> str:
        """
        Generate SQL query for a specific Spider 2 database.
        
        Args:
            natural_language_query: Question in natural language
            database_name: Target database name
            temperature: Model temperature for generation
            
        Returns:
            Generated SQL query
        """
        try:
            # Get database schema
            schema_info = self.get_database_schema(database_name)
            
            # Prepare schema context for Gemini
            schema_context = self._format_schema_for_gemini(schema_info)
            
            system_prompt = f"""You are an expert SQL query generator for the Spider 2 dataset. 
Your task is to convert natural language questions into valid SQL queries for the {database_name} database.

Database Schema Context:
{schema_context}

Important Rules:
1. Only use tables and columns that exist in the provided schema
2. Use proper BigQuery SQL syntax
3. Include the full table references: `{self.spider2_project}.{self.spider2_dataset}.table_name`
4. Handle JOIN operations carefully based on foreign key relationships
5. Use appropriate aggregation functions when needed
6. Consider data types when writing WHERE clauses
7. Return only the SQL query, no explanations

Generate a SQL query that answers the natural language question."""
            
            config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
            
            sql_query = self.gemini.generate(
                prompt=f"Natural language query: {natural_language_query}",
                system_prompt=system_prompt,
                config=config
            )
            
            return sql_query.strip()
            
        except Exception as e:
            raise Exception(f"Failed to generate SQL for {database_name}: {str(e)}")
    
    def execute_and_analyze(
        self,
        natural_language_query: str,
        database_name: str,
        execute_query: bool = True
    ) -> Dict[str, Any]:
        """
        Generate SQL, execute it, and provide analysis.
        
        Args:
            natural_language_query: Question in natural language
            database_name: Target database name
            execute_query: Whether to actually execute the query
            
        Returns:
            Dictionary with SQL, results, and analysis
        """
        try:
            # Generate SQL
            sql_query = self.generate_sql_for_database(
                natural_language_query, 
                database_name
            )
            
            result = {
                "natural_language_query": natural_language_query,
                "database_name": database_name,
                "generated_sql": sql_query,
                "execution_results": None,
                "analysis": None,
                "error": None
            }
            
            if execute_query:
                try:
                    # Execute the query
                    df_results = self.bigquery_client.query(sql_query)
                    result["execution_results"] = {
                        "success": True,
                        "row_count": len(df_results),
                        "columns": list(df_results.columns),
                        "sample_data": df_results.head(5).to_dict('records') if len(df_results) > 0 else []
                    }
                    
                    # Generate analysis of results
                    analysis = self._analyze_query_results(
                        natural_language_query,
                        sql_query,
                        df_results
                    )
                    result["analysis"] = analysis
                    
                except Exception as e:
                    result["execution_results"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            return result
            
        except Exception as e:
            return {
                "natural_language_query": natural_language_query,
                "database_name": database_name,
                "generated_sql": None,
                "execution_results": None,
                "analysis": None,
                "error": str(e)
            }
    
    def benchmark_query(
        self,
        natural_language_query: str,
        expected_sql: str,
        database_name: str
    ) -> Dict[str, Any]:
        """
        Benchmark generated SQL against expected SQL (for evaluation).
        
        Args:
            natural_language_query: Question in natural language
            expected_sql: Expected/ground truth SQL
            database_name: Target database name
            
        Returns:
            Benchmark results with comparison
        """
        try:
            # Generate SQL
            generated_sql = self.generate_sql_for_database(
                natural_language_query,
                database_name
            )
            
            # Execute both queries
            try:
                generated_results = self.bigquery_client.query(generated_sql)
                expected_results = self.bigquery_client.query(expected_sql)
                
                # Compare results
                results_match = generated_results.equals(expected_results)
                
                return {
                    "natural_language_query": natural_language_query,
                    "database_name": database_name,
                    "generated_sql": generated_sql,
                    "expected_sql": expected_sql,
                    "results_match": results_match,
                    "generated_row_count": len(generated_results),
                    "expected_row_count": len(expected_results),
                    "execution_success": True
                }
                
            except Exception as e:
                return {
                    "natural_language_query": natural_language_query,
                    "database_name": database_name,
                    "generated_sql": generated_sql,
                    "expected_sql": expected_sql,
                    "results_match": False,
                    "execution_error": str(e),
                    "execution_success": False
                }
                
        except Exception as e:
            return {
                "natural_language_query": natural_language_query,
                "database_name": database_name,
                "generation_error": str(e),
                "execution_success": False
            }
    
    def _format_schema_for_gemini(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for Gemini context."""
        schema_text = f"Database: {schema_info['database_name']}\\n\\n"
        
        for table_name, table_info in schema_info["tables"].items():
            schema_text += f"Table: {table_name}\\n"
            schema_text += f"Rows: {table_info.get('num_rows', 'N/A')}\\n"
            schema_text += "Columns:\\n"
            
            for column in table_info["columns"]:
                schema_text += f"  - {column['name']} ({column['type']}) {column.get('mode', '')}\\n"
            
            schema_text += "\\n"
        
        return schema_text
    
    def _analyze_query_results(
        self,
        natural_query: str,
        sql_query: str,
        results_df: pd.DataFrame
    ) -> str:
        """Analyze query results using Gemini."""
        try:
            analysis_prompt = f"""
Natural Language Query: {natural_query}
Generated SQL: {sql_query}
Result Summary: {len(results_df)} rows returned
Columns: {list(results_df.columns)}
Sample Data: {results_df.head(3).to_dict('records') if len(results_df) > 0 else 'No data'}

Analyze these query results and provide:
1. Whether the SQL correctly answers the natural language question
2. Key insights from the results
3. Any potential issues or improvements
"""
            
            config = GeminiConfig(temperature=0.4, max_output_tokens=512)
            return self.gemini.generate(
                prompt=analysis_prompt,
                system_prompt="You are a SQL analysis expert. Provide concise analysis of query results.",
                config=config
            )
        except Exception:
            return "Analysis could not be generated."
    
    def close(self):
        """Close connections."""
        self.bigquery_client.close()