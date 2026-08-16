import json
import sqlite3

def populate_ground_truth(db_path="olist.db", json_path="benchmark_qa.json"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
        
    for qa in qa_data:
        try:
            cursor.execute(qa["query"])
            result = cursor.fetchone()
            
            if result:
                # Assumes every benchmark query returns a single scalar value
                val = result[0]
                # Format floats to 2 decimal places to avoid precision mismatches, else str
                if isinstance(val, float):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                qa["ground_truth_answer"] = val_str
            else:
                qa["ground_truth_answer"] = "None"
        except Exception as e:
            print(f"Error executing query for question ID {qa['id']}: {qa['query']}\n{e}")
            qa["ground_truth_answer"] = "ERROR"
            
    conn.close()
    
    with open(json_path, 'w') as f:
        json.dump(qa_data, f, indent=2)
        
    print("Populated ground_truth_answer in benchmark_qa.json")

if __name__ == "__main__":
    populate_ground_truth()
