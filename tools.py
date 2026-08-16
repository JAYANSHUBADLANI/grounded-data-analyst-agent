import sqlite3

DB_PATH = "olist.db"

def get_schema() -> str:
    """
    Returns the database schema: tables and their columns.
    Used by the agent to understand what data is available.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            return "Error: No tables found in the database. Ensure data_loader.py has been run."
            
        schema_info = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            col_details = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                col_details.append(f"{col_name} ({col_type})")
                
            schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_details)}")
            
        conn.close()
        return "\n\n".join(schema_info)
    except Exception as e:
        return f"Error retrieving schema: {e}"

def execute_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the database.
    Returns the result rows as a string, or an error message if it fails.
    """
    try:
        # Basic safety check (though SQLite sandboxing is already quite limited)
        forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "create", "replace"]
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in forbidden_keywords):
            return "Error: Only read-only SELECT queries are allowed."
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "Query executed successfully, but returned 0 rows."
            
        # Format the result
        result_str = f"Columns: {', '.join(columns)}\n"
        for row in rows[:50]: # Cap the output to 50 rows to avoid blowing up context window
            result_str += f"{row}\n"
            
        if len(rows) > 50:
            result_str += f"... and {len(rows) - 50} more rows."
            
        return result_str
        
    except Exception as e:
        return f"Error executing query: {e}"
