import os
import sqlite3
import json
import logging
from datetime import datetime, timedelta
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
    Enhanced DataStore with budget management capabilities.
    Manages SQLite database with transaction data and budget tables.
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

        # Ensure budget tables exist (even if CSV loading was skipped)
        self._ensure_budget_tables()

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
            # Don't drop budget tables - they contain user data
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

    def _ensure_budget_tables(self):
        """Create budget-related tables if they don't exist."""
        budget_tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS user_budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                monthly_limit REAL NOT NULL,
                budget_type TEXT DEFAULT 'fixed', -- 'percentage', 'fixed', 'goal_based'
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_date TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(client_id, category, is_active)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS budget_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER NOT NULL,
                month TEXT NOT NULL, -- 'YYYY-MM'
                category TEXT NOT NULL,
                budgeted_amount REAL NOT NULL,
                actual_amount REAL NOT NULL,
                variance_amount REAL NOT NULL,
                variance_percentage REAL NOT NULL,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(client_id, month, category)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS financial_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER NOT NULL,
                goal_type TEXT NOT NULL, -- 'savings', 'debt_payoff', 'emergency_fund'
                goal_name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                target_date TEXT,
                monthly_contribution REAL,
                current_progress REAL DEFAULT 0,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_date TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            );
            """,
            # Budget table indexes
            "CREATE INDEX IF NOT EXISTS idx_budget_client ON user_budgets(client_id);",
            "CREATE INDEX IF NOT EXISTS idx_budget_category ON user_budgets(category);",
            "CREATE INDEX IF NOT EXISTS idx_tracking_client_month ON budget_tracking(client_id, month);",
            "CREATE INDEX IF NOT EXISTS idx_goals_client ON financial_goals(client_id);",
        ]

        try:
            for sql in budget_tables_sql:
                self.conn.execute(sql)
            self.conn.commit()
            print("✅ Budget tables ensured")
        except Exception as e:
            logger.error(f"❌ Failed to create budget tables: {e}")
            raise

    def _build_schema_info(self):
        """Build schema metadata for LLM-driven query generation."""
        common_cols = {
            "date": "TEXT (YYYY-MM-DD) – Transaction date",
            "amount": "REAL – Transaction amount in dollars",
            "mcc_category": "TEXT – Spending category (restaurants, grocery, etc.)",
            "card_type": "TEXT – Type of card used",
            "merchant_city": "TEXT – City where transaction occurred",
            "is_weekend": "BOOLEAN – Weekend transaction flag",
            "is_night_txn": "BOOLEAN – Night transaction flag",
            "current_age": "INTEGER – Customer age",
            "gender": "TEXT – Customer gender",
            "yearly_income": "REAL – Annual income",
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
                    "SELECT SUM(amount) FROM client_transactions WHERE client_id = 430 AND date >= '2025-01-01'",
                    "SELECT mcc_category, SUM(amount) FROM client_transactions WHERE client_id = 430 GROUP BY mcc_category",
                ],
            },
            "overall_transactions": {
                "description": "Market benchmark data",
                "columns": common_cols,
                "sample_queries": [
                    "SELECT AVG(amount) FROM overall_transactions WHERE mcc_category = 'restaurants'",
                ],
            },
            "user_budgets": {
                "description": "User-defined budget limits by category",
                "columns": {
                    "client_id": "INTEGER – Client identifier",
                    "category": "TEXT – Budget category (matches mcc_category)",
                    "monthly_limit": "REAL – Monthly budget limit in dollars",
                    "budget_type": "TEXT – Type of budget (fixed, percentage, goal_based)",
                    "is_active": "BOOLEAN – Whether budget is currently active",
                },
                "sample_queries": [
                    "SELECT * FROM user_budgets WHERE client_id = 430 AND is_active = 1",
                ],
            },
            "budget_tracking": {
                "description": "Monthly budget vs actual spending tracking",
                "columns": {
                    "client_id": "INTEGER – Client identifier",
                    "month": "TEXT – Month in YYYY-MM format",
                    "category": "TEXT – Spending category",
                    "budgeted_amount": "REAL – Budgeted amount for the month",
                    "actual_amount": "REAL – Actual spending for the month",
                    "variance_amount": "REAL – Difference (actual - budgeted)",
                    "variance_percentage": "REAL – Percentage variance",
                },
                "sample_queries": [
                    "SELECT * FROM budget_tracking WHERE client_id = 430 AND month = '2025-07'",
                ],
            },
            "financial_goals": {
                "description": "User financial goals and progress tracking",
                "columns": {
                    "client_id": "INTEGER – Client identifier",
                    "goal_type": "TEXT – Type of goal (savings, debt_payoff, emergency_fund)",
                    "goal_name": "TEXT – Name/description of the goal",
                    "target_amount": "REAL – Target amount to achieve",
                    "target_date": "TEXT – Target completion date",
                    "current_progress": "REAL – Current progress toward goal",
                    "monthly_contribution": "REAL – Required monthly contribution",
                },
                "sample_queries": [
                    "SELECT * FROM financial_goals WHERE client_id = 430 AND is_active = 1",
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
                ("version", "1.1"),  # Updated version for budget support
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

        # Date range for transaction tables
        if table in ("client_transactions", "overall_transactions"):
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

        # Prevent data-changing statements on transaction tables (but allow budget table updates)
        forbidden_on_transactions = {"DROP", "DELETE", "UPDATE", "ALTER", "CREATE"}
        if not is_select:
            # Allow INSERT/UPDATE/DELETE on budget tables
            is_budget_operation = any(table in sql.upper() for table in 
                                    ["USER_BUDGETS", "BUDGET_TRACKING", "FINANCIAL_GOALS"])
            is_insert_or_update = any(op in sql.upper() for op in ["INSERT", "UPDATE", "DELETE"])
            
            if not is_budget_operation and any(word in sql.upper() for word in forbidden_on_transactions):
                raise ValueError("Only SELECT queries allowed on transaction tables. Budget table modifications are allowed.")
            elif not is_budget_operation and not is_insert_or_update:
                raise ValueError("Only SELECT queries allowed on transaction tables")

        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

        # Commit for non-SELECT operations
        if not is_select:
            self.conn.commit()

        # Cache small SELECTs
        if is_select and not params and len(rows) < 1000:
            self._query_cache[cache_key] = {
                "rows": rows,
                "columns": cols,
                "timestamp": datetime.now(),
            }

        return rows, cols

    def create_or_update_budget(self, client_id: int, category: str, monthly_limit: float, budget_type: str = "fixed") -> Dict[str, Any]:
        """Create or update a budget for a client with proper handling."""
        try:
            # Check if budget already exists
            existing_budget = self.conn.execute(
                "SELECT id, monthly_limit FROM user_budgets WHERE client_id = ? AND category = ? AND is_active = 1",
                (client_id, category)
            ).fetchone()
            
            if existing_budget:
                # Update existing budget
                budget_id, old_limit = existing_budget
                self.conn.execute(
                    """
                    UPDATE user_budgets 
                    SET monthly_limit = ?, budget_type = ?, updated_date = ?
                    WHERE id = ?
                    """,
                    (monthly_limit, budget_type, datetime.now().isoformat(), budget_id)
                )
                
                action = "updated"
                message = f"Updated {category} budget from ${old_limit:.2f} to ${monthly_limit:.2f}"
            else:
                # Create new budget
                self.conn.execute(
                    """
                    INSERT INTO user_budgets (client_id, category, monthly_limit, budget_type, created_date, updated_date, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (client_id, category, monthly_limit, budget_type, datetime.now().isoformat(), datetime.now().isoformat())
                )
                
                action = "created"
                message = f"Created new {category} budget of ${monthly_limit:.2f}"
            
            self.conn.commit()
            
            # Update budget tracking for current month to reflect changes
            current_month = datetime.now().strftime("%Y-%m")
            self.update_budget_tracking(client_id, current_month)
            
            return {
                "success": True,
                "action": action,
                "message": message,
                "category": category,
                "monthly_limit": monthly_limit,
                "budget_type": budget_type
            }
            
        except Exception as e:
            logger.error(f"Failed to create/update budget: {e}")
            self.conn.rollback()
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to {action if 'action' in locals() else 'create/update'} budget for {category}"
            }

    def delete_budget(self, client_id: int, category: str) -> Dict[str, Any]:
        """Delete (soft-deactivate) a budget for a specific category, handling UNIQUE constraint."""
        try:
            category = category.title().strip()

            # 1) Look up active budget
            existing_budget = self.conn.execute(
                "SELECT id, monthly_limit FROM user_budgets WHERE client_id=? AND category=? AND is_active=1",
                (client_id, category)
            ).fetchone()

            if not existing_budget:
                return {
                    "success": False,
                    "error": "Budget not found",
                    "message": f"No active budget found for {category}"
                }

            budget_id, monthly_limit = existing_budget
            now_iso = datetime.now().isoformat()
            current_month = datetime.now().strftime("%Y-%m")

            # 2) Begin transaction
            self.conn.execute("BEGIN")

            # 3) Remove any prior inactive duplicates for this client/category
            self.conn.execute(
                "DELETE FROM user_budgets WHERE client_id=? AND category=? AND is_active=0",
                (client_id, category)
            )

            # 4) Flip the active row to inactive
            self.conn.execute(
                "UPDATE user_budgets SET is_active=0, updated_date=? WHERE id=?",
                (now_iso, budget_id)
            )

            # 5) Clean future tracking
            self.conn.execute(
                "DELETE FROM budget_tracking WHERE client_id=? AND category=? AND month>=?",
                (client_id, category, current_month)
            )

            # 6) Commit
            self.conn.commit()

            return {
                "success": True,
                "message": f"Successfully deleted {category} budget (${monthly_limit:.2f})",
                "category": category,
                "deleted_limit": monthly_limit
            }

        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete budget for {category}"
            }


    def list_client_budgets_detailed(self, client_id: int) -> Dict[str, Any]:
        """Get detailed budget information including tracking data."""
        try:
            # Get active budgets
            budgets_query = """
            SELECT category, monthly_limit, budget_type, created_date, updated_date
            FROM user_budgets 
            WHERE client_id = ? AND is_active = 1 
            ORDER BY category
            """
            
            budget_rows, budget_cols = self.execute_sql_query(budgets_query, (client_id,))
            budgets_df = pd.DataFrame(budget_rows, columns=budget_cols)
            
            if budgets_df.empty:
                return {
                    "success": True,
                    "message": "No active budgets found",
                    "budgets": [],
                    "total_budgeted": 0.0
                }
            
            # Get current month tracking data
            current_month = datetime.now().strftime("%Y-%m")
            tracking_query = """
            SELECT category, budgeted_amount, actual_amount, variance_amount, variance_percentage
            FROM budget_tracking 
            WHERE client_id = ? AND month = ?
            """
            
            tracking_rows, tracking_cols = self.execute_sql_query(tracking_query, (client_id, current_month))
            tracking_df = pd.DataFrame(tracking_rows, columns=tracking_cols)
            
            # Combine budget and tracking data
            budget_list = []
            total_budgeted = 0.0
            
            for _, budget in budgets_df.iterrows():
                category = budget['category']
                monthly_limit = budget['monthly_limit']
                total_budgeted += monthly_limit
                
                # Find tracking data for this category
                tracking_data = tracking_df[tracking_df['category'] == category]
                
                budget_info = {
                    "category": category,
                    "monthly_limit": monthly_limit,
                    "budget_type": budget['budget_type'],
                    "created_date": budget['created_date'],
                    "updated_date": budget['updated_date']
                }
                
                if not tracking_data.empty:
                    track = tracking_data.iloc[0]
                    budget_info.update({
                        "current_month_actual": track['actual_amount'],
                        "variance_amount": track['variance_amount'],
                        "variance_percentage": track['variance_percentage'],
                        "status": "over_budget" if track['variance_amount'] > 0 else "under_budget"
                    })
                else:
                    budget_info.update({
                        "current_month_actual": 0.0,
                        "variance_amount": 0.0,
                        "variance_percentage": 0.0,
                        "status": "no_tracking_data"
                    })
                
                budget_list.append(budget_info)
            
            return {
                "success": True,
                "budgets": budget_list,
                "total_budgeted": total_budgeted,
                "budget_count": len(budget_list),
                "current_month": current_month
            }
            
        except Exception as e:
            logger.error(f"Failed to get detailed budgets: {e}")
            return {
                "success": False,
                "error": str(e),
                "budgets": [],
                "total_budgeted": 0.0
            }

    def create_budget(self, client_id: int, category: str, monthly_limit: float, budget_type: str = "fixed") -> bool:
        """Legacy method - redirects to create_or_update_budget for backward compatibility."""
        result = self.create_or_update_budget(client_id, category, monthly_limit, budget_type)
        return result["success"]


    def get_client_budgets(self, client_id: int) -> pd.DataFrame:
        """Get all active budgets for a client."""
        rows, cols = self.execute_sql_query(
            "SELECT * FROM user_budgets WHERE client_id = ? AND is_active = 1 ORDER BY category",
            (client_id,)
        )
        return pd.DataFrame(rows, columns=cols)

    def update_budget_tracking(self, client_id: int, month: str) -> bool:
        """Update budget tracking for a specific month."""
        try:
            # Get active budgets
            budgets = self.get_client_budgets(client_id)
            if budgets.empty:
                return False

            # Calculate actual spending by category for the month
            for _, budget in budgets.iterrows():
                category = budget['category']
                budgeted_amount = budget['monthly_limit']
                
                # Get actual spending
                actual_rows, _ = self.execute_sql_query(
                    """
                    SELECT COALESCE(SUM(amount), 0) as actual_amount
                    FROM client_transactions 
                    WHERE client_id = ? AND mcc_category = ? 
                    AND strftime('%Y-%m', date) = ?
                    """,
                    (client_id, category, month)
                )
                
                actual_amount = actual_rows[0][0] if actual_rows else 0
                variance_amount = actual_amount - budgeted_amount
                variance_percentage = (variance_amount / budgeted_amount * 100) if budgeted_amount > 0 else 0
                
                # Insert or update tracking record
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO budget_tracking 
                    (client_id, month, category, budgeted_amount, actual_amount, variance_amount, variance_percentage)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (client_id, month, category, budgeted_amount, actual_amount, variance_amount, variance_percentage)
                )
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update budget tracking: {e}")
            return False

    def get_budget_performance(self, client_id: int, month: str) -> pd.DataFrame:
        """Get budget performance for a specific month."""
        rows, cols = self.execute_sql_query(
            """
            SELECT * FROM budget_tracking 
            WHERE client_id = ? AND month = ?
            ORDER BY ABS(variance_percentage) DESC
            """,
            (client_id, month)
        )
        return pd.DataFrame(rows, columns=cols)

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
            for table in ("client_transactions", "overall_transactions", "user_budgets", "budget_tracking", "financial_goals"):
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
