# spending_agent.py
# Comprehensive Spending Agent tools for LangChain + LangGraph

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from data_store.data_store import DataStore
from langchain_core.tools import tool

# Utility to parse date strings

def _parse_date(date_str: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return None

@tool
def get_spending_summary( client_id: int, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Calculate spending metrics for a client between two dates (inclusive).
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start is None or end is None:
        return {"error": "Invalid date format. Use 'YYYY-MM-DD'."}

    df = DataStore().get_client_data(client_id)
    if df.empty:
        return {"error": f"No data for client {client_id}."}

    period = df[(df['date'] >= start) & (df['date'] <= end)]
    if period.empty:
        return {"error": f"No transactions for client {client_id} in {start_date} to {end_date}."}

    total = float(period['amount'].sum())
    count = int(period.shape[0])
    avg = float(period['amount'].mean())
    med = float(period['amount'].median())
    days = int(period['date'].dt.date.nunique())
    daily_avg = total / days
    mx = float(period['amount'].max())
    mn = float(period['amount'].min())

    return {
       
        "date_range": {"start": start_date, "end": end_date},
        "total_spending": total,
        "transaction_count": count,
        "average_transaction": avg,
        "median_transaction": med,
        "spending_days": days,
        "daily_average": daily_avg,
        "max_transaction": mx,
        "min_transaction": mn
    }

@tool
def analyze_time_patterns(client_id: int) -> Dict[str, Any]:
    """
    Analyze client spending patterns across different time dimensions.
    
    Args:
        client_id: The client identifier
    
    Returns:
        Dictionary with time-based spending pattern analysis
    """
    try:
        data_store = DataStore()
        client_data = data_store.get_client_data(client_id)
        
        if client_data.empty:
            return {"error": f"No data found for client {client_id}"}
        
        # Weekend vs Weekday analysis
        weekend_data = client_data[client_data['is_weekend'] == 1]
        weekday_data = client_data[client_data['is_weekend'] == 0]
        
        # Daily patterns
        daily_spending = client_data.groupby('day_name')['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        # Hourly patterns  
        hourly_spending = client_data.groupby('txn_hour')['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        # Monthly patterns
        monthly_spending = client_data.groupby(client_data['date'].dt.month)['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        # Night vs Day
        night_data = client_data[client_data['is_night_txn'] == 1]
        day_data = client_data[client_data['is_night_txn'] == 0]
        
        return {
            "weekend_vs_weekday": {
                "weekend_total": float(weekend_data['amount'].sum()),
                "weekday_total": float(weekday_data['amount'].sum()),
                "weekend_avg_transaction": float(weekend_data['amount'].mean()) if len(weekend_data) > 0 else 0,
                "weekday_avg_transaction": float(weekday_data['amount'].mean()) if len(weekday_data) > 0 else 0,
                "weekend_percentage": float(weekend_data['amount'].sum() / client_data['amount'].sum() * 100),
                "weekend_preference": weekend_data['amount'].sum() > weekday_data['amount'].sum()
            },
            "daily_patterns": {
                "by_day": daily_spending.to_dict('index'),
                "peak_day": daily_spending['sum'].idxmax(),
                "lowest_day": daily_spending['sum'].idxmin()
            },
            "hourly_patterns": {
                "by_hour": hourly_spending.to_dict('index'),
                "peak_hour": int(hourly_spending['sum'].idxmax()),
                "quiet_hour": int(hourly_spending['sum'].idxmin())
            },
            "monthly_patterns": {
                "by_month": monthly_spending.to_dict('index'),
                "peak_month": int(monthly_spending['sum'].idxmax()),
                "lowest_month": int(monthly_spending['sum'].idxmin())
            },
            "night_vs_day": {
                "night_total": float(night_data['amount'].sum()),
                "day_total": float(day_data['amount'].sum()),
                "night_percentage": float(night_data['amount'].sum() / client_data['amount'].sum() * 100),
                "night_preference": night_data['amount'].sum() > day_data['amount'].sum()
            },
            "behavioral_insights": {
                "is_weekend_spender": weekend_data['amount'].sum() > weekday_data['amount'].sum(),
                "is_night_spender": night_data['amount'].sum() > day_data['amount'].sum(),
                "peak_spending_time": f"{daily_spending['sum'].idxmax()} at {hourly_spending['sum'].idxmax()}:00"
            }
        }
    except Exception as e:
        return {"error": f"Error in time pattern analysis: {str(e)}"}

@tool
def get_spending_by_category(
    client_id: int,
    start_date: str,
    end_date: str,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Breakdown of spending by MCC category over a date range.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start is None or end is None:
        return {"error": "Invalid date format."}

    df = DataStore().get_client_data(client_id)
    period = df[(df['date'] >= start) & (df['date'] <= end)]
    if period.empty:
        return {"error": "No transactions in given period."}

    stats = (
        period.groupby('mcc_category')['amount']
        .agg(total_spent='sum', transaction_count='count', average_spent='mean')
        .round(2)
    )
    stats['percentage'] = (stats['total_spent'] / stats['total_spent'].sum() * 100).round(1)
    top = stats.sort_values('total_spent', ascending=False).head(top_n)

    return {
       
        "date_range": {"start": start_date, "end": end_date},
        "category_breakdown": top.to_dict('index')
    }

@tool
def get_spending_by_category_date(
    client_id: int,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    Time series of category spending: amount per day per MCC category.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start is None or end is None:
        return {"error": "Invalid date format."}

    df = DataStore().get_client_data(client_id)
    period = df[(df['date'] >= start) & (df['date'] <= end)]
    if period.empty:
        return {"error": "No transactions in given period."}

    period['date_only'] = period['date'].dt.strftime('%Y-%m-%d')
    ts = (
        period.groupby(['date_only', 'mcc_category'])['amount']
        .sum()
        .reset_index()
    )
    # Pivot: dates as keys mapping to category:amount dicts
    result: Dict[str, Dict[str, float]] = {}
    for date, group in ts.groupby('date_only'):
        result[date] = {
            row['mcc_category']: float(row['amount']) for _, row in group.iterrows()
        }
    return {
    
        "date_series": result
    }

@tool
def get_spending_by_night( client_id: int, start_date: str, end_date: str, night_start: int = 22, night_end: int = 6 ) -> Dict[str, Any]:
    """
    Analyze spending that occurred during night hours vs day hours.
    night_start and night_end are hours in 24h format.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start is None or end is None:
        return {"error": "Invalid date format."}

    df = DataStore().get_client_data(client_id)
    period = df[(df['date'] >= start) & (df['date'] <= end)]
    if period.empty:
        return {"error": "No transactions in given period."}

    # Extract hour
    period['hour'] = period['date'].dt.hour

    # Night if hour >= night_start or < night_end
    night_mask = (period['hour'] >= night_start) | (period['hour'] < night_end)
    night_data = period[night_mask]
    day_data = period[~night_mask]

    return {
        "client_id": client_id,
        "date_range": {"start": start_date, "end": end_date},
        "night_total": float(night_data['amount'].sum()),
        "day_total": float(day_data['amount'].sum()),
        "night_count": int(night_data.shape[0]),
        "day_count": int(day_data.shape[0]),
        "night_average": float(night_data['amount'].mean()) if not night_data.empty else 0,
        "day_average": float(day_data['amount'].mean()) if not day_data.empty else 0
    }


