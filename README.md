# AI-Powered SQL Query Generator

## Project Overview
This project implements an AI-powered SQL query generator that converts natural language questions into SQL queries. It utilizes advanced language models and vector stores to understand and process user queries efficiently.

## Demo Video

[Watch the AI-Powered SQL Query Generator Demo on Loom](https://www.dropbox.com/scl/fi/n6z3gy7h2pu26wkkcxtjd/Text-to-SQL.mp4?rlkey=a69gegnqrecbghqniqc76lo2g&st=h5i6kj0c&dl=0)

[![AI-Powered SQL Query Generator Demo](https://path-to-your-thumbnail-image.jpg)](https://www.dropbox.com/scl/fi/n6z3gy7h2pu26wkkcxtjd/Text-to-SQL.mp4?rlkey=a69gegnqrecbghqniqc76lo2g&st=h5i6kj0c&dl=0)


## Tech Stack
- **Language**: Python
- **AI Model**: LLaMa 3 70B (via Groq API)
- **Vector Store**: ChromaDB
- **Database**: PostgreSQL
- **API Framework**: Flask

## Implementation Steps

### Setup
1. **Database**: 
   - PostgreSQL database with mock tables, such as:
     - Users
     - Employees

2. **CSV Data**: 
   - Utilize CSV files containing mock data, including:
     - Sales data (e.g., `sales_data.csv`)
     - Inventory data (e.g., `inventory.csv`)
   - These CSV files will be used for training and as sample data sources.

### Business Logic
1. **LLM Integration**: 
   - Utilize the Groq API to access the LLaMa 3 70B model for natural language processing and query generation.

2. **Vector Store Setup**: 
   - Implement ChromaDB as the vector store for efficient similarity searches and data retrieval.

3. **Text to Query Conversion**: 
   - Implement logic to convert natural language queries into SQL queries using the LLaMa model.

4. **Query Execution**: 
   - Execute the generated queries against the PostgreSQL database and return the results.

5. **Training**: 
   - Implement a training process that can use either CSV data or database schema information.

### REST API
- **Endpoints**: 
  - **GET /train**: Initiates the training process
  - **POST /ask**: Accepts a question and returns the generated SQL query and results

## Key Components

### ChromaDB (Vector Store)
- Manages collections for SQL queries, DDL, documentation, and CSV metadata.
- Provides methods for adding and retrieving similar questions, related DDL, and documentation.

### Groq API Integration
- Utilizes the Groq API to access the LLaMa 3 70B model.
- Handles prompt submission and response processing.

### CSV Integration
- Processes CSV files to extract metadata and store it in the vector database.
- Allows for querying and retrieval of CSV-related information.

### PostgreSQL Integration
- Connects to a PostgreSQL database for executing SQL queries and retrieving schema information.

## Getting Started
### Setup and Running the Application

1. **Install Dependencies**:
   - Run the following command to install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Configure Database Connection**:
   - Update the PostgreSQL connection details in the application code:
     ```python
     app_instance.connect_to_postgres(host='localhost', dbname='indexer', user='postgres', password='your_password', port='5432')
     ```

3. **Run the Application**:
   - Navigate to the project folder and run:
     ```bash
     python app.py
     ```

4. **Using the API**:
   - To train the model:
     ```bash
     GET /train?csv=path_to_csv_file  # For CSV-based training
     GET /train  # For database schema-based training
     ```
   - To ask a question:
     ```bash
     POST /ask
     Content-Type: application/json
     
     {
         "question": "Your natural language question here"
     }
     ```

   Example using curl:

   For training:
   ```bash
   curl "http://localhost:5000/train?csv=path/to/your/csvfile.csv"
   ```
   or
   ```bash
   curl "http://localhost:5000/train"
   ```

   For asking a question:
   ```bash
   curl -X POST "http://localhost:5000/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is the total sales for the last quarter?"}'
   ```

   The response will be a JSON object containing the generated SQL query and the query results.
