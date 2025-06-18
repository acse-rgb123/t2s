The project aims to do several things but can be split into three broad aspects.

1. Data ingestion and augmented context
	- We expect to be able to take in a variety of data connections, focusing primarily on Cloud SQL databases, and BigQuery databases (ignoring excel spreadsheets, mysql, nosql and other data connection types)
	- We expect the data to be unlabeled, messy, and slightly unstructured, riddled with errors. This means we must explore the dataset (likely using hard coded SQL or BigQuery SQL code to explore and obtain properties of each column and how they are related to each other), add descriptions to the column names and rename column names for clarity, begin data engineering by identifying percentage of null values, or inconsistent types within a column, etc. The result of this process is expected to be two fold: A clean and well engineered, and descriptive dataset, along with a report on what changes have been made in case the user wishes to revise or add context and correct changes themselves.

2. Text to SQL
	- this query engine is built on ny llms and will take in a question or instructions from the analysis module. the purpose is to follow the instruction or answer the question by querying the data layaer that it will be connected to. initially this will be sql but moving forward as multiple datalayers are attached different querying languages for non relational databases
	- the output will be the data from running the query which will be passed to the analysis agent
	- this tool can be called multiple time in order accumulate different pieces of data essential to answering the question

3. Conversational mechanism
	- 

4. Analysis module
	- this module takes in the question answereed by the user and maps out the attack plan to answering the data. it is able to use the text to sql querying engine to retrieve facts and information about the dataset which in turn will help it perform arithmetic and other analytical pieces to answer the question. it can then output visuals, dashboard inforgrpahics etc responsible for answering the question in the users perferred format
