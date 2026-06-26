# %pip install mysql-connector-python 

import mysql.connector
import pandas as pd
import os

database_name = 'ccs'
# Connection credentials
config = {
    'host': 'relational.fel.cvut.cz',
    'port': 3306,
    'user': 'guest',
    'password': 'ctu-relational',
    'database': database_name  # The database name
}

# Output directory for CSVs
output_dir = f'../data/{database_name}'
os.makedirs(output_dir, exist_ok=True)

# Connect to MariaDB and export all tables to CSV
def download_dataset():
    try:
        print("Connecting to MariaDB...")
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print(f"Found {len(tables)} tables.")

        for (table_name,) in tables:
            print(f"Exporting table: {table_name}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            csv_path = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(csv_path, index=False)

        print(f"All tables exported to '{output_dir}'")
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()

# Run
download_dataset()
