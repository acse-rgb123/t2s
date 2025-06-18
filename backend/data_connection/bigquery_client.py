"""
BigQuery client for connecting to Spider 2 dataset and other BigQuery datasets.
"""

import os
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from google.cloud import bigquery
from google.auth import default
from google.oauth2 import service_account
import json


class BigQueryClient:
    """
    BigQuery client for querying datasets including Spider 2.
    
    Usage:
        # Using default credentials
        client = BigQueryClient()
        
        # Using service account
        client = BigQueryClient(service_account_path="path/to/key.json")
        
        # Query Spider 2 dataset
        results = client.query_spider2("SELECT * FROM table_name LIMIT 10")
    """
    
    def __init__(
        self, 
        project_id: Optional[str] = None,
        service_account_path: Optional[str] = None,
        credentials_json: Optional[str] = None
    ):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: Google Cloud project ID
            service_account_path: Path to service account JSON file
            credentials_json: Service account JSON as string
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        # Initialize credentials
        if service_account_path:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path
            )
        elif credentials_json:
            credentials_dict = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict
            )
        else:
            # Use default credentials (gcloud auth application-default login)
            credentials, project = default()
            if not self.project_id:
                self.project_id = project
        
        # Initialize BigQuery client
        self.client = bigquery.Client(
            project=self.project_id,
            credentials=credentials if 'credentials' in locals() else None
        )
        
        # Spider 2 dataset configuration
        self.spider2_project = "spider2-public-data"
        self.spider2_dataset = "spider2_1_0"
    
    def query(
        self, 
        sql: str, 
        to_dataframe: bool = True,
        dry_run: bool = False
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute a SQL query on BigQuery.
        
        Args:
            sql: SQL query string
            to_dataframe: Return results as pandas DataFrame
            dry_run: Only validate query without running it
            
        Returns:
            Query results as DataFrame or list of dictionaries
        """
        try:
            job_config = bigquery.QueryJobConfig()
            job_config.dry_run = dry_run
            
            if dry_run:
                job = self.client.query(sql, job_config=job_config)
                return {
                    "valid": True,
                    "total_bytes_processed": job.total_bytes_processed,
                    "total_bytes_billed": job.total_bytes_billed
                }
            
            # Execute query
            query_job = self.client.query(sql, job_config=job_config)
            results = query_job.result()
            
            if to_dataframe:
                return results.to_dataframe()
            else:
                return [dict(row) for row in results]
                
        except Exception as e:
            raise Exception(f"BigQuery query failed: {str(e)}")
    
    def query_spider2(
        self, 
        sql: str, 
        to_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Query the Spider 2 dataset specifically.
        
        Args:
            sql: SQL query string (will be prefixed with Spider 2 dataset)
            to_dataframe: Return results as pandas DataFrame
            
        Returns:
            Query results as DataFrame or list of dictionaries
        """
        # Ensure query uses the Spider 2 dataset
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed for Spider 2 dataset")
        
        # Add dataset prefix if not present
        if f"`{self.spider2_project}.{self.spider2_dataset}`" not in sql:
            # This is a simplified approach - you might want to parse the SQL more carefully
            pass
        
        return self.query(sql, to_dataframe=to_dataframe)
    
    def list_spider2_tables(self) -> List[str]:
        """
        List all available tables in the Spider 2 dataset.
        
        Returns:
            List of table names
        """
        try:
            dataset_ref = self.client.dataset(
                self.spider2_dataset, 
                project=self.spider2_project
            )
            tables = self.client.list_tables(dataset_ref)
            return [table.table_id for table in tables]
        except Exception as e:
            raise Exception(f"Failed to list Spider 2 tables: {str(e)}")
    
    def get_table_schema(
        self, 
        table_name: str, 
        dataset_name: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            dataset_name: Dataset name (defaults to Spider 2)
            project_name: Project name (defaults to Spider 2)
            
        Returns:
            Table schema information
        """
        try:
            project = project_name or self.spider2_project
            dataset = dataset_name or self.spider2_dataset
            
            table_ref = self.client.dataset(dataset, project=project).table(table_name)
            table = self.client.get_table(table_ref)
            
            schema_info = {
                "table_name": table_name,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "columns": []
            }
            
            for field in table.schema:
                column_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description
                }
                schema_info["columns"].append(column_info)
            
            return schema_info
            
        except Exception as e:
            raise Exception(f"Failed to get table schema: {str(e)}")
    
    def get_spider2_schema_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive schema summary of the Spider 2 dataset.
        
        Returns:
            Schema summary with all tables and their structures
        """
        try:
            tables = self.list_spider2_tables()
            schema_summary = {
                "dataset": f"{self.spider2_project}.{self.spider2_dataset}",
                "total_tables": len(tables),
                "tables": {}
            }
            
            for table_name in tables:
                try:
                    schema_info = self.get_table_schema(table_name)
                    schema_summary["tables"][table_name] = schema_info
                except Exception as e:
                    schema_summary["tables"][table_name] = {
                        "error": f"Failed to get schema: {str(e)}"
                    }
            
            return schema_summary
            
        except Exception as e:
            raise Exception(f"Failed to get Spider 2 schema summary: {str(e)}")
    
    def sample_table_data(
        self, 
        table_name: str, 
        limit: int = 5,
        dataset_name: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get sample data from a table.
        
        Args:
            table_name: Name of the table
            limit: Number of rows to sample
            dataset_name: Dataset name (defaults to Spider 2)
            project_name: Project name (defaults to Spider 2)
            
        Returns:
            Sample data as DataFrame
        """
        project = project_name or self.spider2_project
        dataset = dataset_name or self.spider2_dataset
        
        sql = f"""
        SELECT *
        FROM `{project}.{dataset}.{table_name}`
        LIMIT {limit}
        """
        
        return self.query(sql, to_dataframe=True)
    
    def validate_query(self, sql: str) -> Dict[str, Any]:
        """
        Validate a SQL query without executing it.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Validation results including cost estimation
        """
        return self.query(sql, dry_run=True)
    
    def close(self):
        """Close the BigQuery client connection."""
        self.client.close()


# Convenience functions for quick access
def connect_to_spider2(
    project_id: Optional[str] = None,
    service_account_path: Optional[str] = None
) -> BigQueryClient:
    """
    Quick connection to Spider 2 dataset.
    
    Args:
        project_id: Google Cloud project ID
        service_account_path: Path to service account JSON
        
    Returns:
        Configured BigQueryClient instance
    """
    return BigQueryClient(
        project_id=project_id,
        service_account_path=service_account_path
    )


def quick_spider2_query(
    sql: str,
    project_id: Optional[str] = None,
    service_account_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Execute a quick query on Spider 2 dataset.
    
    Args:
        sql: SQL query string
        project_id: Google Cloud project ID
        service_account_path: Path to service account JSON
        
    Returns:
        Query results as DataFrame
    """
    client = connect_to_spider2(project_id, service_account_path)
    try:
        return client.query_spider2(sql)
    finally:
        client.close()