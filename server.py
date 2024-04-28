from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelWithLMHead, AutoTokenizer
import mysql.connector
import sqlparse
from fastapi.middleware.cors import CORSMiddleware
import re
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Add OPTIONS method
    allow_headers=["*"],
)

# Initialize T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

class QueryRequest(BaseModel):
    username: str
    password: str
    dbname: str
    queryname: str

def get_sql(query):
    input_text = "translate English to SQL: %s </s>" % query
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'])

    return tokenizer.decode(output[0])
def parse_sql(sql_query):
    parsed = sqlparse.parse(sql_query)
    return parsed
def execute_sql(db_config, sql_query):
    try:
        # Connect to MySQL database
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        db.close()
        return result
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {err.msg}")
def add_parentheses(string):
    # Define the words that need parentheses
    functions = ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT']
    
    # Regular expression pattern to find the words and their next set of words
    pattern = r'\b(' + '|'.join(functions) + r')\b\s+(\w+)\b'
    
    # Define a function to insert parentheses
    def replace(match):
        return f"{match.group(1)}({match.group(2)})"

    # Replace the matches with the updated string
    result = re.sub(pattern, replace, string)
    
    return result
def get_all_tables(db_config):
    try:
        # Connect to MySQL database
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        
        # Execute query to get all table names
        cursor.execute("SHOW TABLES")
        
        # Fetch all table names
        table_names = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        db.close()
        
        return table_names
    
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {err.msg}")

@app.post("/sql/")
async def generate_sql_and_fetch_results(request: QueryRequest):
    username = request.username
    password = request.password
    dbname = request.dbname
    query = request.queryname
    print(username)
    print(password)
    print(dbname)
    # Generate SQL query
    db_config = {
        "host": "localhost",
        "user": username,
        "passwd": password,
        "database": dbname
    }
    sql_query = get_sql(query)
    print(sql_query)
    sql_query = sql_query.strip("<pad>").strip("</s>").strip()

    print(sql_query)

    parsed_query = parse_sql(sql_query)
    t_name = get_all_tables(db_config)
    print(t_name[0])
    # Extract SQL query string from the parsed query
    query_strings = [str(stmt) for stmt in parsed_query]

    # Perform replacement on each query string
    queryfinals = [query_str.replace("table", t_name[0]) for query_str in query_strings]

    # Join modified query strings back into a single string
    final_query = ";".join(queryfinals)
    print(final_query)

# Split the string by '='
    if "=" in final_query:
        parts = final_query.split('=')

        # Check if the second part contains any alphabets
        if any(c.isalpha() for c in parts[1]):
            # If it contains alphabets, add single quotes around it
            final_query = parts[0] + "='" + parts[1].strip() + "'"

    print(final_query)

    f_query = add_parentheses(final_query)
    print(f_query)
    # MySQL connection configuration

    # Execute SQL query
    result = execute_sql(db_config, f_query)
    print(result)
    res = {
        "query": sql_query,
        "table": result
    }
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
