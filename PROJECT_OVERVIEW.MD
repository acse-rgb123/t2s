The project aims to do several things but can be split into three broad aspects.

1. Data ingestion and augmented context
	- We expect to be able to take in a variety of data connections, focusing primarily on Cloud SQL databases, and BigQuery databases (ignoring excel spreadsheets, mysql, nosql and other data connection types)
	- We expect the data to be unlabeled, messy, and slightly unstructured, riddled with errors. This means we must explore the dataset (likely using hard coded SQL or BigQuery SQL code to explore and obtain properties of each column and how they are related to each other), add descriptions to the column names and rename column names for clarity, begin data engineering by identifying percentage of null values, or inconsistent types within a column, etc. The result of this process is expected to be two fold: A clean and well engineered, and descriptive dataset, along with a report on what changes have been made in case the user wishes to revise or add context and correct changes themselves.

2. Text to SQL
	- 

3. Conversational mechanism
	- 
