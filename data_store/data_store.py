import os
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


class DataStore:
    """
    Optimized DataStore that intelligently manages SQLite database creation
    and CSV loading. Only loads CSV data when necessary (first run, database
    missing, CSVs updated, or forced).
    """

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        db_path: str = "banking_data.db",
        force_reload: bool = False,
        check_csv_modified: bool = True,
    ):
        self.db_path = db_path
        self.client_csv_path = client_csv_path
        self.overall_csv_path = overall_csv_path
        self.conn: sqlite3.Connection = None  # type: ignore
        self.schema_info: Dict[str, Any] = {}
        self._query_cache: Dict[str, Any] = {}  # Simple query cache

        # Initialize connection
        self._initialize_connection()

        # Determine if we need to load CSV data
        need_to_load = self._should_load_data(force_reload, check_csv_modified)

        if need_to_load:
            print("📥 Loading CSV data into database...")
            self._setup_fresh_database()
        else:
            print("✅ Using existing database (CSV loading skipped)")
            self._build_schema_info()

        logger.info("✅ DataStore ready for queries.")

    def _initialize_connection(self):
        """Initialize SQLite connection with optimizations."""
        try:
            self.conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            # SQLite optimizations
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = 10000")
            self.conn.execute("PRAGMA temp_store = MEMORY")
            logger.info(f"✅ Connected to SQLite DB at {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Cannot open database: {e}")
            raise

    def _should_load_data(
        self, force_reload: bool, check_csv_modified: bool
    ) -> bool:
        """Intelligently determine if CSV data needs to be loaded."""
        # Force reload requested
        if force_reload:
            print("🔄 Force reload requested")
            return True

        # Database file doesn't exist
        if not os.path.exists(self.db_path):
            print("🆕 Database doesn't exist, will create")
            return True

        # Check required tables
        try:
            result = self.conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN 
                  ('client_transactions', 'overall_transactions')
                """
            ).fetchall()
            if len(result) != 2:
                print("📋 Missing required tables, will reload")
                return True
        except Exception as e:
            print(f"❌ Error checking tables: {e}, will reload")
            return True

        # Check table contents
        try:
            client_count = self.conn.execute(
                "SELECT COUNT(*) FROM client_transactions"
            ).fetchone()[0]
            overall_count = self.conn.execute(
                "SELECT COUNT(*) FROM overall_transactions"
            ).fetchone()[0]
            if client_count == 0 or overall_count == 0:
                print("📊 Tables empty, will reload")
                return True
            print(
                f"✅ DB has data: {client_count:,} client + "
                f"{overall_count:,} overall transactions"
            )
        except Exception as e:
            print(f"❌ Error checking contents: {e}, will reload")
            return True

        # CSV newer than DB?
        if check_csv_modified:
            try:
                db_mtime = os.path.getmtime(self.db_path)
                client_mtime = os.path.getmtime(self.client_csv_path)
                overall_mtime = os.path.getmtime(self.overall_csv_path)
                if client_mtime > db_mtime or overall_mtime > db_mtime:
                    print("📅 CSV files are newer than DB, will reload")
                    return True
            except FileNotFoundError as e:
                print(f"⚠️ CSV not found: {e}, using existing DB")
            except Exception as e:
                print(f"⚠️ Timestamp check error: {e}, using existing DB")

        # Verify integrity
        if not self._verify_database_integrity():
            print("🔧 Integrity check failed, will reload")
            return True

        return False

    def _verify_database_integrity(self) -> bool:
        """Verify database structure and metadata."""
        try:
            meta = self.conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='_database_metadata'
                """
            ).fetchone()
            if not meta:
                return False
            status = self.conn.execute(
                "SELECT value FROM _database_metadata WHERE key = 'status'"
            ).fetchone()
            return status and status[0] == "ready"
        except Exception:
            return False

    def _setup_fresh_database(self):
        """Set up database from scratch with CSV data."""
        try:
            # Drop old tables
            self.conn.execute("DROP TABLE IF EXISTS client_transactions")
            self.conn.execute("DROP TABLE IF EXISTS overall_transactions")
            self.conn.execute("DROP TABLE IF EXISTS _database_metadata")
            self.conn.commit()

            # Create & load
            self.create_tables()
            self.load_client_data(self.client_csv_path)
            self.load_overall_data(self.overall_csv_path)
            self._mark_database_ready()

            # Optimize
            self._optimize_database()
        except Exception as e:
            logger.error(f"❌ Failed to setup fresh database: {e}")
            raise

    def create_tables(self):
        """Create tables and indexes for client & overall transactions."""
        client_sql = """
        CREATE TABLE IF NOT EXISTS client_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            client_id INTEGER NOT NULL,
            card_id INTEGER,
            card_type TEXT,
            card_brand TEXT,
            amount REAL NOT NULL,
            merchant_id INTEGER,
            merchant_city TEXT,
            merchant_state TEXT,
            zip TEXT,
            mcc_number INTEGER,
            use_chip BOOLEAN,
            credit_limit REAL,
            acct_open_date TEXT,
            card_on_dark_web BOOLEAN,
            current_age INTEGER,
            gender TEXT,
            per_capita_income REAL,
            yearly_income REAL,
            total_debt REAL,
            credit_score INTEGER,
            num_credit_cards INTEGER,
            day_name TEXT,
            is_weekend BOOLEAN,
            mcc_original TEXT,
            mcc_category TEXT,
            month TEXT,
            txn_time TEXT,
            txn_hour INTEGER,
            txn_date TEXT,
            is_night_txn BOOLEAN,
            txn_count_per_day INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        overall_sql = client_sql.replace(
            "client_transactions", "overall_transactions"
        ).replace(
            "client_id INTEGER NOT NULL,", ""
        ).replace(
            "transaction_id INTEGER NOT NULL,", ""
        )

        indexes = [
            # client_transactions indexes
            "CREATE INDEX IF NOT EXISTS idx_client_id ON client_transactions(client_id);",
            "CREATE INDEX IF NOT EXISTS idx_client_date ON client_transactions(date);",
            "CREATE INDEX IF NOT EXISTS idx_client_category ON client_transactions(mcc_category);",
            "CREATE INDEX IF NOT EXISTS idx_client_amount ON client_transactions(amount);",
            # overall_transactions indexes
            "CREATE INDEX IF NOT EXISTS idx_overall_date ON overall_transactions(date);",
            "CREATE INDEX IF NOT EXISTS idx_overall_category ON overall_transactions(mcc_category);",
            "CREATE INDEX IF NOT EXISTS idx_overall_amount ON overall_transactions(amount);",
            "CREATE INDEX IF NOT EXISTS idx_overall_age ON overall_transactions(current_age);",
            "CREATE INDEX IF NOT EXISTS idx_overall_gender ON overall_transactions(gender);",
            # compound indexes
            "CREATE INDEX IF NOT EXISTS idx_client_compound ON client_transactions(client_id, date, mcc_category);",
            "CREATE INDEX IF NOT EXISTS idx_overall_compound ON overall_transactions(date, mcc_category, current_age);",
        ]

        try:
            self.conn.execute(client_sql)
            self.conn.execute(overall_sql)
            for stmt in indexes:
                self.conn.execute(stmt)
            self.conn.commit()
            self._build_schema_info()
            logger.info("✅ Tables & indexes created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create tables/indexes: {e}")
            raise

    def _build_schema_info(self):
        """Build schema metadata for LLM-driven query generation."""
        common_cols = {
            "date": "TEXT (YYYY-MM-DD) – Transaction date",
            "card_id": "INTEGER – Card identifier",
            # … add the rest of the columns here …
        }

        self.schema_info = {
            "client_transactions": {
                "description": "Client-specific transaction data",
                "columns": {
                    "client_id": "INTEGER – Unique client identifier",
                    "transaction_id": "INTEGER – Unique transaction identifier",
                    **common_cols,
                },
                "sample_queries": [
                    "SELECT SUM(amount) FROM client_transactions "
                    "WHERE client_id = 430 AND date >= '2023-01-01'",
                    # …
                ],
            },
            "overall_transactions": {
                "description": "Market benchmark data",
                "columns": common_cols,
                "sample_queries": [
                    "SELECT AVG(amount) FROM overall_transactions "
                    "WHERE mcc_category = 'restaurants'",
                    # …
                ],
            },
        }

    def get_schema_info(self) -> Dict[str, Any]:
        """Return stored schema information."""
        return self.schema_info

    def load_client_data(self, csv_path: str) -> Dict[str, Any]:
        """Load and preprocess client CSV into the database."""
        logger.info(f"📥 Loading client CSV: {csv_path}")
        df = pd.read_csv(csv_path, dtype={"client_id": "int32"})
        df = self._preprocess_dataframe(df, include_client_id=True)
        self.conn.execute("DELETE FROM client_transactions")
        df.to_sql(
            "client_transactions",
            self.conn,
            if_exists="append",
            index=False,
        )
        self.conn.commit()
        return self._get_table_stats("client_transactions")

    def load_overall_data(self, csv_path: str) -> Dict[str, Any]:
        """Load and preprocess overall CSV into the database."""
        logger.info(f"📥 Loading overall CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df = self._preprocess_dataframe(df, include_client_id=False)
        self.conn.execute("DELETE FROM overall_transactions")
        df.to_sql(
            "overall_transactions",
            self.conn,
            if_exists="append",
            index=False,
            chunksize=10000,
        )
        self.conn.commit()
        return self._get_table_stats("overall_transactions")
    

    def _preprocess_dataframe(
        self, df: pd.DataFrame, include_client_id: bool
    ) -> pd.DataFrame:
        
        """Enhanced preprocessing with data type optimization."""
        if not include_client_id and "client_id" in df.columns:
            df = df.drop(columns=["client_id"])

        # Normalize date columns
        for col in ("date", "txn_date"):
            if col in df.columns:
                df[col] = (
                    pd.to_datetime(df[col], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                )

        # Ensure time is string
        if "txn_time" in df.columns:
            df["txn_time"] = df["txn_time"].astype(str)

        # Boolean flags
        for col in (
            "use_chip",
            "is_weekend",
            "card_on_dark_web",
            "is_night_txn",
        ):
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Fill missing categories
        df = df.fillna(
            {
                "mcc_original": "Unknown",
                "mcc_category": "Other",
                "card_type": "Unknown",
                "card_brand": "Unknown",
                "gender": "Unknown",
            }
        )

        # Downcast numerics
        numeric_types = {
            "amount": "float32",
            "credit_score": "int16",
            "num_credit_cards": "int8",
            "current_age": "int8",
            "merchant_id": "int32",
            "mcc_number": "int16",
            "credit_limit": "float32",
            "per_capita_income": "float32",
            "yearly_income": "float32",
            "total_debt": "float32",
            "txn_hour": "int8",
            "txn_count_per_day": "int16",
        }
        for col, dtype in numeric_types.items():
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0)
                    .astype(dtype)
                )

        return df

    def _mark_database_ready(self):
        """Mark database as properly set up with metadata."""
        try:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _database_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            metadata = [
                ("status", "ready"),
                ("version", "1.0"),
                ("last_loaded", datetime.now().isoformat()),
                ("client_csv_path", self.client_csv_path),
                ("overall_csv_path", self.overall_csv_path),
            ]
            self.conn.executemany(
                "INSERT OR REPLACE INTO _database_metadata (key, value) VALUES (?, ?)",
                metadata,
            )
            self.conn.commit()
            logger.info("✅ Database marked as ready")
        except Exception as e:
            logger.warning(f"Could not set database metadata: {e}")

    def _optimize_database(self):
        """Optimize database after data loading."""
        try:
            print("🔧 Optimizing database...")
            self.conn.execute("ANALYZE")
            self.conn.execute("VACUUM")
            self.conn.commit()
            print("✅ Database optimization complete")
        except Exception as e:
            logger.warning(f"Database optimization failed: {e}")

    def _get_table_stats(self, table: str) -> Dict[str, Any]:
        """Get comprehensive table statistics."""
        stats: Dict[str, Any] = {}

        # Row count
        stats["row_count"] = self.conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        # Date range
        date_min, date_max = self.conn.execute(
            f"SELECT MIN(date), MAX(date) FROM {table}"
        ).fetchone()
        stats["date_range"] = {"min": date_min, "max": date_max}

        # Amount stats
        total, avg, mn, mx = self.conn.execute(
            f"SELECT SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM {table}"
        ).fetchone()
        stats["amount_stats"] = {
            "total": total,
            "average": round(avg, 2) if avg else 0,
            "min": mn,
            "max": mx,
        }

        # Top categories
        top = self.conn.execute(
            f"""
            SELECT mcc_category, COUNT(*) AS count
            FROM {table}
            GROUP BY mcc_category
            ORDER BY count DESC
            LIMIT 5
            """
        ).fetchall()
        stats["top_categories"] = dict(top)

        return stats

    def execute_sql_query(
        self, sql: str, params: Tuple = ()
    ) -> Tuple[List[tuple], List[str]]:
        """Execute SQL with safety checks and optional caching."""
        cache_key = f"{sql}:{params}"
        is_select = sql.strip().upper().startswith("SELECT")

        # Return cached SELECT if fresh
        if is_select and cache_key in self._query_cache:
            cached = self._query_cache[cache_key]
            if (datetime.now() - cached["timestamp"]).seconds < 300:
                return cached["rows"], cached["columns"]

        # Prevent data-changing statements
        forbidden = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"}
        if not is_select and any(word in sql.upper() for word in forbidden):
            raise ValueError("Only SELECT queries allowed for analysis")

        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

        # Cache small SELECTs
        if is_select and not params and len(rows) < 1000:
            self._query_cache[cache_key] = {
                "rows": rows,
                "columns": cols,
                "timestamp": datetime.now(),
            }

        return rows, cols

    def get_client_data(self, client_id: int) -> pd.DataFrame:
        """Get all data for a specific client."""
        rows, cols = self.execute_sql_query(
            "SELECT * FROM client_transactions WHERE client_id = ? ORDER BY date DESC",
            (client_id,),
        )
        return pd.DataFrame(rows, columns=cols)

    def get_overall_data(self) -> pd.DataFrame:
        """Get overall market data (limited)."""
        rows, cols = self.execute_sql_query(
            "SELECT * FROM overall_transactions ORDER BY date DESC LIMIT 10000"
        )
        return pd.DataFrame(rows, columns=cols)

    def get_client_summary(self, client_id: int) -> Dict[str, Any]:
        """Get quick client summary with enhanced metrics."""
        sql = """
            SELECT
                COUNT(*) AS transaction_count,
                SUM(amount) AS total_spending,
                AVG(amount) AS avg_transaction,
                MIN(date) AS first_transaction,
                MAX(date) AS last_transaction,
                COUNT(DISTINCT mcc_category) AS unique_categories,
                MAX(current_age) AS current_age,
                MAX(gender) AS gender,
                MAX(yearly_income) AS yearly_income,
                AVG(CASE WHEN is_weekend = 1 THEN amount END) AS avg_weekend_spending,
                COUNT(CASE WHEN is_night_txn = 1 THEN 1 END) AS night_transactions
            FROM client_transactions
            WHERE client_id = ?
        """
        rows, cols = self.execute_sql_query(sql, (client_id,))
        return dict(zip(cols, rows[0])) if rows else {}

    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market overview."""
        sql = """
            SELECT
                COUNT(*) AS total_transactions,
                SUM(amount) AS total_market_volume,
                AVG(amount) AS avg_transaction,
                COUNT(DISTINCT mcc_category) AS unique_categories,
                MIN(current_age) AS min_age,
                MAX(current_age) AS max_age,
                COUNT(DISTINCT gender) AS gender_groups,
                AVG(yearly_income) AS avg_income
            FROM overall_transactions
        """
        rows, cols = self.execute_sql_query(sql)
        return dict(zip(cols, rows[0])) if rows else {}

    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database status and performance info."""
        info: Dict[str, Any] = {
            "database_path": self.db_path,
            "database_size_mb": (
                round(os.path.getsize(self.db_path) / (1024 ** 2), 2)
                if os.path.exists(self.db_path)
                else 0
            ),
            "tables": {},
            "performance": {
                "cache_size": len(self._query_cache),
                "connection_status": "connected" if self.conn else "disconnected",
            },
        }

        try:
            for table in ("client_transactions", "overall_transactions"):
                info["tables"][table] = self._get_table_stats(table)

            # Metadata
            try:
                meta = self.conn.execute(
                    "SELECT key, value FROM _database_metadata"
                ).fetchall()
                info["metadata"] = dict(meta)
            except Exception:
                info["metadata"] = {}

            # Index list
            idxs = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
            info["indexes"] = [row[0] for row in idxs]

        except Exception as e:
            info["error"] = str(e)

        return info

    def clear_cache(self):
        """Clear query cache."""
        self._query_cache.clear()
        logger.info("🧹 Query cache cleared")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("🔒 SQLite connection closed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except Exception:
            pass


# Global DataStore instance for tools
_datastore: Optional[DataStore] = None


def _ensure_datastore() -> DataStore:
    """Ensure global datastore is initialized."""
    global _datastore
    if _datastore is None:
        _datastore = DataStore(
            client_csv_path=Path(__file__).parent
            / "Banking_Data.csv",
            overall_csv_path=Path(__file__).parent
            / "overall_data.csv",
            db_path="banking_data.db",
        )
    return _datastore




@tool
def generate_sql_for_client_analysis(
    user_query: str,
    client_id: int
) -> Dict[str, Any]:
    """
    Generate optimized SQL for client_transactions from a natural language query.
    Always includes client_id filter and uses proper indexes.
    """
    ds = _ensure_datastore()
    schema = ds.get_schema_info()["client_transactions"]

    # Build schema description
    schema_desc = [
        "CLIENT_TRANSACTIONS TABLE:",
        f"Description: {schema['description']}",
        "",
        "KEY COLUMNS:"
    ]
    important_columns = {
        "client_id": "INTEGER - Unique client identifier (ALWAYS required in WHERE clause)",
        "date": "TEXT (YYYY-MM-DD) - Transaction date",
        "amount": "REAL - Transaction amount in dollars",
        "mcc_category": "TEXT - Category (restaurants, grocery, etc.)",
        "merchant_city": "TEXT - City where transaction occurred",
        "is_weekend": "BOOLEAN - Weekend flag",
        "is_night_txn": "BOOLEAN - Night transaction flag",
        "current_age": "INTEGER - Customer age",
        "gender": "TEXT - Customer gender",
        "yearly_income": "REAL - Annual income"
    }
    for col, desc in important_columns.items():
        schema_desc.append(f"- {col}: {desc}")
    schema_desc = "\n".join(schema_desc)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""\
You are an expert SQL generator for banking transaction analysis.
Generate efficient SQLite queries against client_transactions.

{schema_desc}

CRITICAL REQUIREMENTS:
- ALWAYS include WHERE client_id = {client_id}
- Use indexed columns: client_id, date, mcc_category, amount
- Date format YYYY-MM-DD
- Aggregates: SUM, AVG, COUNT
- RETURN ONLY THE SQL QUERY - no explanations or markdown

Generate ONLY the SQL query for:"""),
        ("human", "{user_query}")
    ])

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        sql = resp.content.strip()

        # Strip any fencing or extra text
        lines = [l.strip() for l in sql.splitlines()]
        sql_lines, found = [], False
        for line in lines:
            if line.upper().startswith("SELECT") or found:
                found = True
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip("```sql ").strip("```")

        # Fallback: extract first SELECT
        if not sql.upper().startswith("SELECT"):
            idx = resp.content.upper().find("SELECT")
            sql = resp.content[idx:].split("\n\n")[0].strip()

        # Validate client_id presence
        if f"client_id = {client_id}" not in sql:
            return {"error": f"Missing client_id filter: {sql}"}

        return {
            "sql_query": sql,
            "query_type": "client_analysis",
            "client_id": client_id,
            "original_query": user_query,
            "optimization_used": "client_id index + compound indexes"
        }
    except Exception as e:
        return {"error": f"SQL generation failed: {e}"}


@tool
def generate_sql_for_benchmark_analysis(
    user_query: str,
    demographic_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate optimized SQL for overall_transactions to get market benchmarks.
    Uses demographic filters and proper indexing for performance.
    """
    ds = _ensure_datastore()
    schema = ds.get_schema_info()["overall_transactions"]

    # Build schema description
    schema_desc = [
        "OVERALL_TRANSACTIONS TABLE:",
        f"Description: {schema['description']}",
        "",
        "KEY COLUMNS:"
    ]
    important_columns = {
        "date": "TEXT (YYYY-MM-DD) - Transaction date",
        "amount": "REAL - Transaction amount in dollars",
        "mcc_category": "TEXT - Category",
        "current_age": "INTEGER - Customer age",
        "gender": "TEXT - Customer gender",
        "yearly_income": "REAL - Annual income",
        "is_weekend": "BOOLEAN - Weekend flag",
        "is_night_txn": "BOOLEAN - Night transaction flag"
    }
    for col, desc in important_columns.items():
        schema_desc.append(f"- {col}: {desc}")
    schema_desc = "\n".join(schema_desc)

    # Build filter context
    filter_ctx = ""
    if demographic_filters:
        filter_ctx = "\nAPPLY THESE DEMOGRAPHIC FILTERS:\n"
        for key, val in demographic_filters.items():
            if key == "gender":
                filter_ctx += f"- gender = '{val}'\n"
            elif key == "age_min":
                filter_ctx += f"- current_age >= {val}\n"
            elif key == "age_max":
                filter_ctx += f"- current_age <= {val}\n"
            elif key == "income_min":
                filter_ctx += f"- yearly_income >= {val}\n"
            elif key == "income_max":
                filter_ctx += f"- yearly_income <= {val}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""\
You are an expert SQL generator for market benchmark analysis.
Generate SQLite queries against overall_transactions.

{schema_desc}{filter_ctx}

GUIDELINES:
- Aggregates: AVG(), COUNT(), SUM()
- Group by: mcc_category, current_age, gender
- Indexed columns: date, mcc_category, current_age, gender, amount
- RETURN ONLY THE SQL QUERY - no explanations or markdown

Generate ONLY the SQL query for:"""),
        ("human", "{user_query}")
    ])

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        sql = resp.content.strip()

        # Clean up
        lines = [l.strip() for l in sql.splitlines()]
        sql_lines, found = [], False
        for line in lines:
            if line.upper().startswith("SELECT") or found:
                found = True
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip("```sql ").strip("```")

        if not sql.upper().startswith("SELECT"):
            idx = resp.content.upper().find("SELECT")
            sql = resp.content[idx:].split("\n\n")[0].strip()

        return {
            "sql_query": sql,
            "query_type": "benchmark_analysis",
            "demographic_filters": demographic_filters,
            "original_query": user_query,
            "optimization_used": "demographic indexes + compound indexes"
        }
    except Exception as e:
        return {"error": f"Benchmark SQL generation failed: {e}"}


@tool
def execute_generated_sql(
    sql_query: str,
    query_type: str,
    format_results: bool = True
) -> Dict[str, Any]:
    """
    Execute generated SQL query with performance monitoring and result formatting.
    """
    ds = _ensure_datastore()

    try:
        print(f"[DEBUG] Executing SQL: {sql_query[:100]}...")
        start = datetime.now()
        rows, cols = ds.execute_sql_query(sql_query)
        duration = (datetime.now() - start).total_seconds()

        results = (
            [dict(zip(cols, r)) for r in rows]
            if format_results else
            rows
        )

        print(f"[DEBUG] SQL executed: {len(rows)} rows in {duration:.3f}s")
        return {
            "query_executed": sql_query,
            "query_type": query_type,
            "column_names": cols,
            "results": results,
            "row_count": len(rows),
            "execution_time_seconds": round(duration, 3),
            "execution_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[DEBUG] SQL execution failed: {e}")
        return {
            "error": f"SQL execution failed: {e}",
            "query_executed": sql_query,
            "query_type": query_type,
            "execution_timestamp": datetime.now().isoformat()
        }

