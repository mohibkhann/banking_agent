#all the neccessary imports
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path
# Import the DataStore 
from banking_agent.data_store.data_store import DataStore
# for current time calculation 
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, Optional


load_dotenv()


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )
)

def get_current_date_context() -> str:
    """Generate current date context for LLM prompts."""
    now = datetime.now()

    # Calculate relevant dates
    last_month_start = (now - relativedelta(months=1)).replace(day=1)
    last_month_end = now.replace(day=1) - timedelta(days=1)
    current_month_start = now.replace(day=1)
    last_year_start = (now.replace(month=1, day=1) - relativedelta(years=1))
    last_year_end = now.replace(month=1, day=1) - timedelta(days=1)

    context = f"""
CURRENT DATE CONTEXT:

- Today's date: {now.strftime("%Y-%m-%d")}
- Current year: {now.year}
- Current month: {now.strftime("%Y-%m")}
- Last month: {last_month_start.strftime("%Y-%m-%d")} to {last_month_end.strftime("%Y-%m-%d")}
- Current month so far: {current_month_start.strftime("%Y-%m-%d")} to {now.strftime("%Y-%m-%d")}
- Last year: {last_year_start.strftime("%Y-%m-%d")} to {last_year_end.strftime("%Y-%m-%d")}

WHEN USER SAYS:

- "last month" → use dates {last_month_start.strftime("%Y-%m-%d")} to {last_month_end.strftime("%Y-%m-%d")}
- "this month" → use dates {current_month_start.strftime("%Y-%m-%d")} to {now.strftime("%Y-%m-%d")}
- "last year" → use dates {last_year_start.strftime("%Y-%m-%d")} to {last_year_end.strftime("%Y-%m-%d")}
- "recent" or "lately" → use last 30 days: {(now - timedelta(days=30)).strftime("%Y-%m-%d")} to {now.strftime("%Y-%m-%d")}
"""
    return context



# Global DataStore instance for tools
_datastore: Optional[DataStore] = None


def _ensure_datastore() -> DataStore:
    """Ensure global datastore is initialized."""
    global _datastore
    if _datastore is None:
        _datastore = DataStore(
            client_csv_path=Path(__file__).parent / "Banking_Data.csv",
            overall_csv_path=Path(__file__).parent / "overall_data.csv",
            db_path="banking_data.db",
        )
    return _datastore


@tool
def generate_sql_for_client_analysis(
    user_query: str,
    client_id: int
) -> Dict[str, Any]:
    """
    Generate optimized SQL for client_transactions with current date context.
    """
    ds = _ensure_datastore()
    schema = ds.get_schema_info()["client_transactions"]

    # Get current date context
    date_context = get_current_date_context()

    # Build schema description
    schema_desc_lines = [
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
        schema_desc_lines.append(f"- {col}: {desc}")
    schema_desc = "\n".join(schema_desc_lines)

    prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
            You are an expert SQL generator for banking transaction analysis.
            Generate efficient SQLite queries against client_transactions. Try making a SQL query which can provide rich, comparative data without deciding the answer inside SQL.

            {date_context}

            {schema_desc}

            CRITICAL REQUIREMENTS:
            - ALWAYS include WHERE client_id = {client_id}
            - Use the CURRENT DATE CONTEXT above for date filtering
            - If the user explicitly mentions specific dates/years (e.g., "2021 or 2023"), you MAY use those exact periods with range filters (YYYY-01-01 to YYYY-12-31). Otherwise, NEVER hardcode years; derive from the provided date context
            - Use indexed columns: client_id, date, mcc_category, amount
            - Date format YYYY-MM-DD
            - Use Aliases in your query
            - Aggregates: SUM, AVG, COUNT
            - NOT SELF-DECIDING: never pick a “winner” or collapse to a single answer in SQL.
            - DO NOT use ORDER BY ... LIMIT 1 to select the max/min period
            - DO NOT use CASE expressions that compare aggregates to choose a label (e.g., return '2021' vs '2023')
            - DO NOT use subqueries that filter to only the max/min aggregate
            - INSTEAD return one row per candidate period/category (e.g., year, month, mcc_category) with the relevant aggregates so the application can decide
            - Prefer sargable range filters over strftime when possible to leverage indexes:
            - e.g., date >= 'YYYY-01-01' AND date < 'YYYY+1-01-01'
            - Try giving a query which provides all the needed comparative data and remains not self-deciding
            - RETURN ONLY THE SQL QUERY - no explanations or markdown

            EXAMPLES:
            - "last month spending" → WHERE date >= 'YYYY-MM-01' AND date <= 'YYYY-MM-DD' (use last-month dates from context); GROUP BY day or category as needed
            - "this year" → WHERE date >= 'YYYY-01-01' AND date <= 'YYYY-MM-DD' (current year to date)

            BAD (self-deciding):
            SELECT CASE
            WHEN SUM(CASE WHEN date >= '2021-01-01' AND date <= '2021-12-31' THEN amount ELSE 0 END) >
                SUM(CASE WHEN date >= '2023-01-01' AND date <= '2023-12-31' THEN amount ELSE 0 END)
            THEN '2021' ELSE '2023' END AS year_with_most_spending
            FROM client_transactions
            WHERE client_id = {client_id};

            BAD (also self-deciding):
            SELECT strftime('%Y',date) AS year,SUM(amount) AS total_spent
            FROM client_transactions
            WHERE client_id = {client_id} AND (date BETWEEN '2021-01-01' AND '2023-12-31')
            GROUP BY year
            ORDER BY total_spent DESC
            LIMIT 1;

            GOOD (comparative, not self-deciding; user named the years):
            SELECT strftime('%Y',date) AS year,SUM(amount) AS total_spent
            FROM client_transactions
            WHERE client_id = {client_id}
            AND (
                (date >= '2021-01-01' AND date < '2022-01-01') OR
                (date >= '2023-01-01' AND date < '2024-01-01')
            )
            GROUP BY year
            ORDER BY year;

            Generate ONLY the SQL query for:
            """),
                ("human", "{user_query}")
            ])
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        
        if not resp or not resp.content:
            return {"error": "No response from LLM"}
            
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
            if idx >= 0:
                sql = resp.content[idx:].split("\n\n")[0].strip()
            else:
                return {"error": f"No valid SQL query generated for: {user_query}"}

        # Validate we have a meaningful SQL query
        if not sql or len(sql.strip()) < 10:
            return {"error": f"Generated SQL too short or empty: {sql}"}

        # Validate client_id presence
        if f"client_id = {client_id}" not in sql:
            return {"error": f"Missing client_id filter: {sql}"}

        # Check for hardcoded 2023 dates and warn
        if "2023" in sql:
            print(f"⚠️ WARNING: Found hardcoded 2023 date in SQL: {sql}")

        return {
            "sql_query": sql,
            "query_type": "client_analysis",
            "client_id": client_id,
            "original_query": user_query,
            "optimization_used": "client_id index + compound indexes",
            "date_context_applied": True
        }

    except Exception as e:
        return {"error": f"SQL generation failed: {e}"}

@tool
def generate_sql_for_benchmark_analysis(
    user_query: str,
    demographic_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate optimized SQL for overall_transactions with current date context.
    """
    ds = _ensure_datastore()
    schema = ds.get_schema_info()["overall_transactions"]

    # Get current date context
    date_context = get_current_date_context()

    # Build schema description
    schema_desc_lines = [
        "OVERALL_TRANSACTIONS TABLE:",
        f"Description: {schema['description']}",
        "",
        "KEY COLUMNS:"
    ]
    important_columns = {
        "date": "TEXT (YYYY-MM-DD) - Transaction date",
        "amount": "REAL - Transaction amount in dollars",
        "mcc_category": "TEXT - Category (use exact matches: 'Groceries', 'Restaurants', 'Transportation', etc.)",
        "current_age": "INTEGER - Customer age",
        "gender": "TEXT - Customer gender",
        "yearly_income": "REAL - Annual income",
        "is_weekend": "BOOLEAN - Weekend flag",
        "is_night_txn": "BOOLEAN - Night transaction flag"
    }
    for col, desc in important_columns.items():
        schema_desc_lines.append(f"- {col}: {desc}")
    schema_desc = "\n".join(schema_desc_lines)

    # Build filter context
    filter_ctx = ""
    if demographic_filters:
        filter_lines = ["\nAPPLY THESE DEMOGRAPHIC FILTERS:"]
        for key, val in demographic_filters.items():
            if key == "gender":
                filter_lines.append(f"- gender = '{val}'")
            elif key == "age_min":
                filter_lines.append(f"- current_age >= {val}")
            elif key == "age_max":
                filter_lines.append(f"- current_age <= {val}")
            elif key == "income_min":
                filter_lines.append(f"- yearly_income >= {val}")
            elif key == "income_max":
                filter_lines.append(f"- yearly_income <= {val}")
        filter_ctx = "\n".join(filter_lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert SQL generator for market benchmark analysis.
Generate SQLite queries against overall_transactions.

{date_context}

{schema_desc}
{filter_ctx}

CRITICAL REQUIREMENTS:
- Use the CURRENT DATE CONTEXT above for date filtering
- NEVER use hardcoded years like 2023 - always use the current dates provided
- For category comparisons, use exact category names like 'Groceries' (not 'grocery')
- Aggregates: AVG(), COUNT(), SUM()
- Group by: mcc_category, current_age, gender
- Use Aliases in your query
- Indexed columns: date, mcc_category, current_age, gender, amount
- ENSURE PROPER SQL SYNTAX - no missing parentheses or incomplete clauses
- RETURN ONLY THE SQL QUERY - no explanations or markdown

EXAMPLE CATEGORY NAMES: 'Groceries', 'Restaurants', 'Transportation', 'Financial Services', 'Entertainment'

Generate ONLY the SQL query for:
"""),
        ("human", "{user_query}")
    ])

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        
        if not resp or not resp.content:
            return {"error": "No response from LLM"}
            
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
            if idx >= 0:
                sql = resp.content[idx:].split("\n\n")[0].strip()
            else:
                return {"error": f"No valid SQL query generated for: {user_query}"}

        # Validate we have a meaningful SQL query
        if not sql or len(sql.strip()) < 10:
            return {"error": f"Generated SQL too short or empty: {sql}"}

        # Check for hardcoded 2023 dates and warn
        if "2023" in sql:
            print(f"⚠️ WARNING: Found hardcoded 2023 date in SQL: {sql}")

        # Basic syntax validation
        if sql.count("(") != sql.count(")"):
            return {"error": f"SQL syntax error - mismatched parentheses: {sql}"}

        return {
            "sql_query": sql,
            "query_type": "benchmark_analysis",
            "demographic_filters": demographic_filters,
            "original_query": user_query,
            "optimization_used": "demographic indexes + compound indexes",
            "date_context_applied": True
        }

    except Exception as e:
        return {"error": f"Benchmark SQL generation failed: {e}"}

@tool
def generate_sql_for_budget_analysis(
    user_query: str,
    client_id: int,
    analysis_type: str = "budget_performance"
) -> Dict[str, Any]:
    """
    Generate optimized SQL for budget-related queries.
    """
    ds = _ensure_datastore()
    
    # Get current date context
    date_context = get_current_date_context()
    
    # Get schema info for budget tables
    budget_schema = ds.get_schema_info().get("user_budgets", {})
    tracking_schema = ds.get_schema_info().get("budget_tracking", {})
    
    # Build comprehensive schema description
    schema_desc = f"""
BUDGET ANALYSIS TABLES:

USER_BUDGETS TABLE:
- client_id: INTEGER - Client identifier (ALWAYS required)
- category: TEXT - Budget category (matches mcc_category from transactions)
- monthly_limit: REAL - Monthly budget limit in dollars
- budget_type: TEXT - Type of budget (fixed, percentage, goal_based)
- is_active: BOOLEAN - Whether budget is currently active (use = 1)

BUDGET_TRACKING TABLE:
- client_id: INTEGER - Client identifier (ALWAYS required)
- month: TEXT - Month in YYYY-MM format
- category: TEXT - Spending category
- budgeted_amount: REAL - Budgeted amount for the month
- actual_amount: REAL - Actual spending for the month
- variance_amount: REAL - Difference (actual - budgeted)
- variance_percentage: REAL - Percentage variance

CLIENT_TRANSACTIONS TABLE (for real-time calculations):
- client_id: INTEGER - Client identifier
- date: TEXT (YYYY-MM-DD) - Transaction date
- amount: REAL - Transaction amount
- mcc_category: TEXT - Spending category
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert SQL generator for budget analysis queries.
Generate efficient SQLite queries for budget management.

{date_context}

{schema_desc}

ANALYSIS TYPES:
- "budget_performance": Compare budgeted vs actual spending
- "budget_status": Current budget limits and categories
- "budget_variance": Identify over/under spending
- "budget_trends": Historical budget performance

CRITICAL REQUIREMENTS:
- ALWAYS include WHERE client_id = {client_id}
- Use CURRENT DATE CONTEXT for date filtering
- For current month analysis, use strftime('%Y-%m', date) for month comparison
- Join tables when needed for comprehensive analysis
- Use meaningful aliases
- RETURN ONLY THE SQL QUERY - no explanations

COMMON PATTERNS:
- Current month spending: WHERE strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
- Budget vs actual: JOIN user_budgets with aggregated client_transactions
- Variance calculation: (actual_amount - budgeted_amount) AS variance

Generate ONLY the SQL query for:
"""),
        ("human", "{user_query}")
    ])

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        resp = llm.invoke(prompt.format_messages(user_query=user_query))
        
        if not resp or not resp.content:
            return {"error": "No response from LLM"}
            
        sql = resp.content.strip()

        # Clean up SQL
        lines = [l.strip() for l in sql.splitlines()]
        sql_lines, found = [], False
        for line in lines:
            if line.upper().startswith("SELECT") or found:
                found = True
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip("```sql ").strip("```")

        # Fallback extraction
        if not sql.upper().startswith("SELECT"):
            idx = resp.content.upper().find("SELECT")
            if idx >= 0:
                sql = resp.content[idx:].split("\n\n")[0].strip()
            else:
                return {"error": f"No valid SQL query generated for: {user_query}"}

        # Validate we have a meaningful SQL query
        if not sql or len(sql.strip()) < 10:
            return {"error": f"Generated SQL too short or empty: {sql}"}

        # Validate client_id presence
        if f"client_id = {client_id}" not in sql:
            return {"error": f"Missing client_id filter in budget query: {sql}"}

        return {
            "sql_query": sql,
            "query_type": "budget_analysis",
            "analysis_type": analysis_type,
            "client_id": client_id,
            "original_query": user_query,
            "optimization_used": "budget table indexes",
            "date_context_applied": True
        }

    except Exception as e:
        return {"error": f"Budget SQL generation failed: {e}"}

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
        if not sql_query or not sql_query.strip():
            return {
                "error": "Empty or null SQL query provided",
                "query_executed": sql_query,
                "query_type": query_type,
                "execution_timestamp": datetime.now().isoformat()
            }
            
        print(f"[DEBUG] Executing SQL: {sql_query[:100]}...")
        start = datetime.now()
        rows, cols = ds.execute_sql_query(sql_query)
        duration = (datetime.now() - start).total_seconds()

        # Safe handling of rows and cols
        if rows is None:
            rows = []
        if cols is None:
            cols = []

        results = (
            [dict(zip(cols, r)) for r in rows]
            if format_results and rows and cols else
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

@tool 
def create_or_update_budget(
    client_id: int,
    category: str,
    monthly_limit: float,
    budget_type: str = "fixed"
) -> Dict[str, Any]:
    """
    Create or update a budget for a specific client and category.
    """
    ds = _ensure_datastore()
    
    try:
        success = ds.create_budget(client_id, category, monthly_limit, budget_type)
        
        if success:
            return {
                "success": True,
                "message": f"Budget created/updated: {category} = ${monthly_limit:.2f}/month",
                "client_id": client_id,
                "category": category,
                "monthly_limit": monthly_limit,
                "budget_type": budget_type
            }
        else:
            return {
                "success": False,
                "error": "Failed to create/update budget",
                "client_id": client_id,
                "category": category
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Budget operation failed: {e}",
            "client_id": client_id,
            "category": category
        }

@tool
def update_budget_tracking_for_month(
    client_id: int,
    month: str
) -> Dict[str, Any]:
    """
    Update budget tracking calculations for a specific client and month.
    Format: month should be 'YYYY-MM' (e.g., '2025-07')
    """
    ds = _ensure_datastore()
    
    try:
        success = ds.update_budget_tracking(client_id, month)
        
        if success:
            # Get the updated tracking data
            performance_df = ds.get_budget_performance(client_id, month)
            
            return {
                "success": True,
                "message": f"Budget tracking updated for {month}",
                "client_id": client_id,
                "month": month,
                "categories_tracked": len(performance_df),
                "tracking_data": performance_df.to_dict('records') if not performance_df.empty else []
            }
        else:
            return {
                "success": False,
                "error": "Failed to update budget tracking - no active budgets found",
                "client_id": client_id,
                "month": month
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Budget tracking update failed: {e}",
            "client_id": client_id,
            "month": month
        }