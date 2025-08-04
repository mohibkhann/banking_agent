import sqlite3
import pandas as pd

def test_queries(db_path: str):
    # 1) Connect
    conn = sqlite3.connect(db_path)
    print(f"Connected to {db_path}\n")

    # 2) List all tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables in database:")
    print(tables, "\n")

    # 3) Show schema for client_transactions
    schema_client = pd.read_sql("PRAGMA table_info(client_transactions);", conn)
    print("client_transactions schema:")
    print(schema_client, "\n")

    # 4) Show schema for overall_transactions
    schema_overall = pd.read_sql("PRAGMA table_info(overall_transactions);", conn)
    print("overall_transactions schema:")
    print(schema_overall, "\n")

    # 5) Peek at the first 5 rows of each table
    print("First 5 client_transactions rows:")
    print(pd.read_sql("SELECT * FROM client_transactions LIMIT 5;", conn), "\n")

    print("First 5 overall_transactions rows:")
    print(pd.read_sql("SELECT * FROM overall_transactions LIMIT 5;", conn), "\n")

    # 6) Run a simple aggregate: total spending for client_id=430 last month
    sql_total = """
        SELECT
            SUM(amount) AS total_spent,
            COUNT(*) AS txn_count,
            AVG(amount) AS avg_txn
        FROM client_transactions
        WHERE client_id = 430
          AND date >= '2023-07-01'
          AND date < '2023-08-01';
    """
    print("Client 430 spending in July 2023:")
    print(pd.read_sql(sql_total, conn), "\n")

    # 7) Run a benchmark query: average overall transaction
    sql_bench = "SELECT AVG(amount) AS avg_overall FROM overall_transactions;"
    print("Average overall transaction amount:")
    print(pd.read_sql(sql_bench, conn), "\n")

    conn.close()
    print("Connection closed.")

if __name__ == "__main__":
    test_queries("C:/Users/mohib.alikhan/Desktop/Banking-Agent/banking_data.db")
