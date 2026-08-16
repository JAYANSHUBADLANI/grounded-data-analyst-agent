import os
import sqlite3
import pandas as pd
import glob

def load_data(db_path="olist.db", raw_data_dir="data/raw"):
    print(f"Loading data from {raw_data_dir} into {db_path}...")
    conn = sqlite3.connect(db_path)
    
    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {raw_data_dir}.")
        print("Please ensure the Kaggle dataset is downloaded and unzipped into data/raw/")
        return False
        
    for file_path in csv_files:
        # Extract table name from filename (e.g., olist_orders_dataset.csv -> orders)
        filename = os.path.basename(file_path)
        table_name = filename.replace("olist_", "").replace("_dataset.csv", "").replace(".csv", "")
        
        print(f"Loading {filename} into table '{table_name}'...")
        # Read CSV and load into SQLite
        try:
            df = pd.read_csv(file_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Successfully loaded {len(df)} rows into '{table_name}'.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    conn.close()
    print("Database creation complete.")
    return True

if __name__ == "__main__":
    load_data()
