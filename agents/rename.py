import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime, timedelta
import json
import operator
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


# GLOBAL DATA STORE - Deals with the retrieval of Banking Data

class DataStore:
    """Global data store for tools to access banking data"""
    _instance = None
    _banking_data = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_data(self, banking_data_path: str):
        """Load banking data"""
        self._banking_data = pd.read_csv("C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv")
        self._banking_data['date'] = pd.to_datetime(self._banking_data['date'])
    
    def get_client_data(self, client_id: str) -> pd.DataFrame:
        """Get filtered data for specific client"""
        if self._banking_data is None:
            raise ValueError("Banking data not loaded")
        return self._banking_data[self._banking_data['client_id'] == client_id].copy()



# STATE DEFINITION

class SpendingAgentState(TypedDict):
    """State for the Spending Agent workflow"""
    client_id: str
    user_query: str
    intent: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]



# PYDANTIC MODELS
# The pydantic models are used to structure data into specific format this enables automatic parsing and serialization 
class QueryIntent(BaseModel):
    """Structured intent classification"""
    query_type: str = Field(description="Type of spending query: summary, category, time_pattern, merchant, comparison, anomaly, custom")
    confidence: float = Field(description="Confidence score 0-1")
    approach: str = Field(description="Processing approach: predefined or dynamic")
    parameters: Dict[str, Any] = Field(description="Extracted parameters from query")
    time_period: Optional[str] = Field(description="Time period if specified")
    specific_category: Optional[str] = Field(description="Specific category if mentioned")

class AnalysisResult(BaseModel):
    """Structured analysis result"""
    summary: str = Field(description="Brief summary of findings")
    key_metrics: Dict[str, Any] = Field(description="Key numerical findings")
    insights: List[str] = Field(description="Generated insights")
    recommendations: List[str] = Field(description="Actionable recommendations")
    data_points: int = Field(description="Number of transactions analyzed")



# TOOLS


@tool
def get_spending_summary(client_id: str, period: str = "all") -> Dict[str, Any]:
    """
    Calculate basic spending metrics for a client over a specified time period.
    
    Args:
        client_id: The client identifier
        period: Time period - 'all', 'last_month', 'current_month', 'last_3_months'
    
    Returns:
        Dictionary with spending summary metrics
    """
    try:
        data_store = DataStore()
        client_data = data_store.get_client_data(client_id)
        
        if client_data.empty:
            return {"error": f"No data found for client {client_id}"}
        
        # Filter by period
        if period == "last_month":
            end_date = datetime.now().replace(day=1) - timedelta(days=1)
            start_date = end_date.replace(day=1)
            period_data = client_data[
                (client_data['date'] >= start_date) & 
                (client_data['date'] <= end_date)
            ]
        elif period == "current_month":
            start_date = datetime.now().replace(day=1)
            period_data = client_data[client_data['date'] >= start_date]
        elif period == "last_3_months":
            start_date = datetime.now() - timedelta(days=90)
            period_data = client_data[client_data['date'] >= start_date]
        else:
            period_data = client_data
        
        if period_data.empty:
            return {"error": f"No data for period: {period}"}
        
        return {
            "total_spending": float(period_data['amount'].sum()),
            "transaction_count": len(period_data),
            "average_transaction": float(period_data['amount'].mean()),
            "median_transaction": float(period_data['amount'].median()),
            "spending_days": period_data['date'].dt.date.nunique(),
            "daily_average": float(period_data['amount'].sum() / period_data['date'].dt.date.nunique()),
            "max_transaction": float(period_data['amount'].max()),
            "min_transaction": float(period_data['amount'].min()),
            "date_range": {
                "start": period_data['date'].min().strftime('%Y-%m-%d'),
                "end": period_data['date'].max().strftime('%Y-%m-%d')
            },
            "period_analyzed": period
        }
    except Exception as e:
        return {"error": f"Error in spending summary: {str(e)}"}

@tool
def analyze_spending_by_category(client_id: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Analyze client spending patterns by merchant categories (MCC codes).
    
    Args:
        client_id: The client identifier
        top_n: Number of top categories to return (default: 10)
    
    Returns:
        Dictionary with category-wise spending analysis
    """
    try:
        data_store = DataStore()
        client_data = data_store.get_client_data(client_id)
        
        if client_data.empty:
            return {"error": f"No data found for client {client_id}"}
        
        # Group by category
        category_spending = client_data.groupby('mcc_category').agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'merchant_id': 'nunique'
        }).round(2)
        
        category_spending.columns = ['total_amount', 'transaction_count', 'avg_amount', 'std_amount', 'unique_merchants']
        category_spending = category_spending.sort_values('total_amount', ascending=False)
        
        # Calculate percentages
        total_spending = client_data['amount'].sum()
        category_spending['percentage'] = (category_spending['total_amount'] / total_spending * 100).round(1)
        
        # Get top categories
        top_categories = category_spending.head(top_n)
        
        return {
            "category_breakdown": top_categories.to_dict('index'),
            "total_categories": len(category_spending),
            "top_category": {
                "name": category_spending.index[0],
                "amount": float(category_spending.iloc[0]['total_amount']),
                "percentage": float(category_spending.iloc[0]['percentage']),
                "transactions": int(category_spending.iloc[0]['transaction_count'])
            },
            "category_diversity_score": len(category_spending),
            "concentration_ratio": float(category_spending.head(3)['percentage'].sum()),  # Top 3 categories
            "spending_distribution": {
                "concentrated": float(category_spending.head(3)['percentage'].sum()) > 60,
                "diversified": len(category_spending) > 10
            }
        }
    except Exception as e:
        return {"error": f"Error in category analysis: {str(e)}"}

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
def analyze_merchant_patterns(client_id: str, top_n: int = 15) -> Dict[str, Any]:
    """
    Analyze client spending patterns by merchants and locations.
    
    Args:
        client_id: The client identifier
        top_n: Number of top merchants to return (default: 15)
    
    Returns:
        Dictionary with merchant-based spending analysis
    """
    try:
        data_store = DataStore()
        client_data = data_store.get_client_data(client_id)
        
        if client_data.empty:
            return {"error": f"No data found for client {client_id}"}
        
        # Merchant analysis
        merchant_analysis = client_data.groupby('merchant_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'merchant_city': 'first',
            'merchant_state': 'first',
            'mcc_category': 'first'
        }).round(2)
        
        merchant_analysis.columns = ['total_spent', 'visit_count', 'avg_transaction', 'city', 'state', 'category']
        merchant_analysis = merchant_analysis.sort_values('total_spent', ascending=False)
        
        # Location analysis
        city_spending = client_data.groupby('merchant_city')['amount'].sum().sort_values(ascending=False)
        state_spending = client_data.groupby('merchant_state')['amount'].sum().sort_values(ascending=False)
        
        # Loyalty metrics
        repeat_merchants = merchant_analysis[merchant_analysis['visit_count'] > 1]
        
        return {
            "top_merchants": merchant_analysis.head(top_n).to_dict('index'),
            "merchant_metrics": {
                "unique_merchants": len(merchant_analysis),
                "average_spend_per_merchant": float(merchant_analysis['total_spent'].mean()),
                "most_visited_merchant": merchant_analysis['visit_count'].idxmax(),
                "highest_spending_merchant": merchant_analysis['total_spent'].idxmax()
            },
            "location_analysis": {
                "top_cities": city_spending.head(5).to_dict(),
                "top_states": state_spending.head(5).to_dict(),
                "geographic_diversity": len(city_spending)
            },
            "loyalty_patterns": {
                "repeat_merchants": len(repeat_merchants),
                "loyalty_ratio": float(len(repeat_merchants) / len(merchant_analysis)),
                "average_visits_per_merchant": float(merchant_analysis['visit_count'].mean()),
                "most_loyal_merchant": {
                    "merchant_id": merchant_analysis['visit_count'].idxmax(),
                    "visits": int(merchant_analysis['visit_count'].max())
                }
            }
        }
    except Exception as e:
        return {"error": f"Error in merchant analysis: {str(e)}"}

@tool
def detect_spending_anomalies(client_id: str, sensitivity: str = "medium") -> Dict[str, Any]:
    """
    Detect unusual spending patterns and anomalous transactions.
    
    Args:
        client_id: The client identifier
        sensitivity: Anomaly detection sensitivity - 'low', 'medium', 'high'
    
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        data_store = DataStore()
        client_data = data_store.get_client_data(client_id)
        
        if client_data.empty:
            return {"error": f"No data found for client {client_id}"}
        
        # Set threshold based on sensitivity
        threshold_factors = {"low": 3.0, "medium": 2.5, "high": 2.0}
        threshold_factor = threshold_factors.get(sensitivity, 2.5)
        
        # Statistical anomaly detection for amounts
        mean_amount = client_data['amount'].mean()
        std_amount = client_data['amount'].std()
        amount_threshold = mean_amount + (threshold_factor * std_amount)
        
        amount_anomalies = client_data[client_data['amount'] > amount_threshold].copy()
        
        # Frequency-based anomalies (unusual merchants)
        merchant_counts = client_data['merchant_id'].value_counts()
        rare_merchants = merchant_counts[merchant_counts == 1].index
        rare_merchant_transactions = client_data[client_data['merchant_id'].isin(rare_merchants)]
        
        # Time-based anomalies (unusual hours)
        hour_counts = client_data['txn_hour'].value_counts()
        unusual_hours = hour_counts[hour_counts < hour_counts.quantile(0.1)].index
        unusual_time_transactions = client_data[client_data['txn_hour'].isin(unusual_hours)]
        
        # Geographic anomalies (new locations)
        location_counts = client_data.groupby(['merchant_city', 'merchant_state']).size()
        rare_locations = location_counts[location_counts == 1].index
        rare_location_transactions = client_data[
            client_data.set_index(['merchant_city', 'merchant_state']).index.isin(rare_locations)
        ]
        
        # Transaction frequency anomalies
        daily_txn_counts = client_data.groupby(client_data['date'].dt.date).size()
        high_frequency_days = daily_txn_counts[daily_txn_counts > daily_txn_counts.mean() + (2 * daily_txn_counts.std())]
        
        return {
            "amount_anomalies": {
                "threshold": float(amount_threshold),
                "count": len(amount_anomalies),
                "total_amount": float(amount_anomalies['amount'].sum()),
                "percentage_of_total": float(len(amount_anomalies) / len(client_data) * 100),
                "largest_transaction": float(amount_anomalies['amount'].max()) if len(amount_anomalies) > 0 else 0,
                "transactions": amount_anomalies[['date', 'amount', 'merchant_id', 'mcc_category']].head(5).to_dict('records')
            },
            "merchant_anomalies": {
                "new_merchants": len(rare_merchant_transactions),
                "new_merchant_spending": float(rare_merchant_transactions['amount'].sum()),
                "exploration_ratio": float(len(rare_merchants) / len(merchant_counts))
            },
            "time_anomalies": {
                "unusual_hour_transactions": len(unusual_time_transactions),
                "unusual_hour_spending": float(unusual_time_transactions['amount'].sum()),
                "unusual_hours": list(unusual_hours)
            },
            "location_anomalies": {
                "new_locations": len(rare_location_transactions),
                "new_location_spending": float(rare_location_transactions['amount'].sum()),
                "geographic_exploration": float(len(rare_locations) / len(location_counts))
            },
            "frequency_anomalies": {
                "high_activity_days": len(high_frequency_days),
                "max_daily_transactions": int(daily_txn_counts.max()),
                "avg_daily_transactions": float(daily_txn_counts.mean())
            },
            "overall_anomaly_score": float((
                len(amount_anomalies) + 
                len(rare_merchant_transactions) + 
                len(unusual_time_transactions)
            ) / len(client_data) * 100),
            "risk_assessment": {
                "low_risk": len(amount_anomalies) < len(client_data) * 0.01,
                "medium_risk": len(client_data) * 0.01 <= len(amount_anomalies) < len(client_data) * 0.05,
                "high_risk": len(amount_anomalies) >= len(client_data) * 0.05
            }
        }
    except Exception as e:
        return {"error": f"Error in anomaly detection: {str(e)}"}

@tool
def compare_spending_periods(client_id: str, period1: str = "current_month", period2: str = "last_month") -> Dict[str, Any]:
    """
    Compare spending between different time periods.
    
    Args:
        client_id: The client identifier
        period1: First period to compare - 'current_month', 'last_month', 'last_3_months'
        period2: Second period to compare - 'current_month', 'last_month', 'last_3_months'
    
    Returns:
        Dictionary with period comparison analysis
    """
    try:
        data_store = DataStore()
        client_data = data_store.get_client_data(client_id)
        
        if client_data.empty:
            return {"error": f"No data found for client {client_id}"}
        
        def get_period_data(period: str) -> pd.DataFrame:
            if period == "current_month":
                start_date = datetime.now().replace(day=1)
                return client_data[client_data['date'] >= start_date]
            elif period == "last_month":
                end_date = datetime.now().replace(day=1) - timedelta(days=1)
                start_date = end_date.replace(day=1)
                return client_data[(client_data['date'] >= start_date) & (client_data['date'] <= end_date)]
            elif period == "last_3_months":
                start_date = datetime.now() - timedelta(days=90)
                return client_data[client_data['date'] >= start_date]
            else:
                return client_data
        
        period1_data = get_period_data(period1)
        period2_data = get_period_data(period2)
        
        if period1_data.empty or period2_data.empty:
            return {"error": f"No data available for comparison periods"}
        
        # Calculate metrics for both periods
        def calculate_metrics(data: pd.DataFrame) -> Dict:
            return {
                "total_spending": float(data['amount'].sum()),
                "transaction_count": len(data),
                "average_transaction": float(data['amount'].mean()),
                "unique_merchants": data['merchant_id'].nunique(),
                "top_category": data.groupby('mcc_category')['amount'].sum().idxmax(),
                "spending_days": data['date'].dt.date.nunique()
            }
        
        metrics1 = calculate_metrics(period1_data)
        metrics2 = calculate_metrics(period2_data)
        
        # Calculate changes
        def calculate_change(val1: float, val2: float) -> Dict:
            if val2 == 0:
                return {"absolute": val1, "percentage": float('inf') if val1 > 0 else 0}
            
            absolute_change = val1 - val2
            percentage_change = (absolute_change / val2) * 100
            
            return {
                "absolute": float(absolute_change),
                "percentage": float(percentage_change)
            }
        
        return {
            "period1": {
                "name": period1,
                "metrics": metrics1
            },
            "period2": {
                "name": period2,
                "metrics": metrics2
            },
            "comparison": {
                "total_spending_change": calculate_change(metrics1["total_spending"], metrics2["total_spending"]),
                "transaction_count_change": calculate_change(metrics1["transaction_count"], metrics2["transaction_count"]),
                "average_transaction_change": calculate_change(metrics1["average_transaction"], metrics2["average_transaction"]),
                "merchant_diversity_change": calculate_change(metrics1["unique_merchants"], metrics2["unique_merchants"])
            },
            "insights": {
                "spending_trend": "increasing" if metrics1["total_spending"] > metrics2["total_spending"] else "decreasing",
                "activity_trend": "more_active" if metrics1["transaction_count"] > metrics2["transaction_count"] else "less_active",
                "spending_behavior": "higher_value_transactions" if metrics1["average_transaction"] > metrics2["average_transaction"] else "lower_value_transactions"
            }
        }
    except Exception as e:
        return {"error": f"Error in period comparison: {str(e)}"}



# LANGGRAPH AGENT IMPLEMENTATION


class SpendingAgent:
    """LangGraph-based Spending Agent using @tool decorators"""
    
    def __init__(self, banking_data_path: str, model_name: str = "gpt-4", memory: bool = True):
        # Initialize data store
        self.data_store = DataStore()
        self.data_store.load_data(banking_data_path)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Define available tools
        self.tools = [
            get_spending_summary,
            analyze_spending_by_category,
            analyze_time_patterns,
            analyze_merchant_patterns,
            detect_spending_anomalies,
            compare_spending_periods
        ]
        
        # Create tool node
        self.tool_node = ToolNode(self.tools)
        
        # Setup memory (optional)
        self.memory = SqliteSaver.from_conn_string(":memory:") if memory else None
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(SpendingAgentState)
        
        # Add nodes
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("tool_executor", self.tool_node)
        workflow.add_node("response_generator", self._response_generator_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the flow
        workflow.set_entry_point("intent_classifier")
        
        # Intent classifier routing
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {
                "execute_tools": "tool_executor",
                "error": "error_handler"
            }
        )
        
        # Tool executor to response generator
        workflow.add_edge("tool_executor", "response_generator")
        
        # End points
        workflow.add_edge("response_generator", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _intent_classifier_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Classify user intent and select appropriate tools"""
        
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing spending queries and selecting appropriate tools.

                Available tools:
                1. get_spending_summary - For basic spending totals, averages, transaction counts
                2. analyze_spending_by_category - For category/MCC analysis, "where do I spend" questions
                3. analyze_time_patterns - For time-based patterns, weekend/weekday, hourly analysis
                4. analyze_merchant_patterns - For merchant-specific analysis, top merchants, locations
                5. detect_spending_anomalies - For unusual transactions, outliers, suspicious activity
                6. compare_spending_periods - For comparing different time periods

                Based on the user query, determine:
                1. Which tools to use (can be multiple)
                2. What parameters to pass to each tool
                3. The order of execution

                Respond with a JSON object containing:
                - tools_to_use: list of tool names
                - tool_parameters: dict mapping tool names to their parameters
                - execution_order: list of tools in execution order
                - query_type: categorization of the query
                - confidence: confidence score 0-1"""),
                            ("human", "User Query: {query}")
        ])
        
        try:
            tool_calls = []
            response = self.llm.invoke(classification_prompt.format_messages(query=state['user_query']))
            
            # Parse the response (in a real implementation, you'd use structured output)
            # For now, create a simple mapping based on keywords
            query_lower = state['user_query'].lower()
            
            tools_to_use = []
            tool_parameters = {}
            
            # Simple keyword-based routing (you could make this more sophisticated)
            if any(word in query_lower for word in ['summary', 'total', 'how much', 'spent']):
                tools_to_use.append('get_spending_summary')
                tool_parameters['get_spending_summary'] = {
                    'client_id': state['client_id'],
                    'period': 'last_month' if 'last month' in query_lower else 'all'
                }
            
            if any(word in query_lower for word in ['category', 'where', 'what', 'breakdown']):
                tools_to_use.append('analyze_spending_by_category')
                tool_parameters['analyze_spending_by_category'] = {
                    'client_id': state['client_id'],
                    'top_n': 10
                }
            
            if any(word in query_lower for word in ['weekend', 'weekday', 'time', 'hour', 'day', 'when']):
                tools_to_use.append('analyze_time_patterns')
                tool_parameters['analyze_time_patterns'] = {
                    'client_id': state['client_id']
                }
            
            if any(word in query_lower for word in ['merchant', 'store', 'shop', 'vendor']):
                tools_to_use.append('analyze_merchant_patterns')
                tool_parameters['analyze_merchant_patterns'] = {
                    'client_id': state['client_id'],
                    'top_n': 15
                }
            
            if any(word in query_lower for word in ['unusual', 'anomal', 'strange', 'suspicious', 'outlier']):
                tools_to_use.append('detect_spending_anomalies')
                tool_parameters['detect_spending_anomalies'] = {
                    'client_id': state['client_id'],
                    'sensitivity': 'medium'
                }
            
            if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'change']):
                tools_to_use.append('compare_spending_periods')
                tool_parameters['compare_spending_periods'] = {
                    'client_id': state['client_id'],
                    'period1': 'current_month',
                    'period2': 'last_month'
                }
            
            # If no specific tools identified, default to summary
            if not tools_to_use:
                tools_to_use = ['get_spending_summary']
                tool_parameters['get_spending_summary'] = {
                    'client_id': state['client_id'],
                    'period': 'all'
                }
            
            state['intent'] = {
                'tools_to_use': tools_to_use,
                'tool_parameters': tool_parameters,
                'execution_order': tools_to_use,
                'query_type': 'multi_tool' if len(tools_to_use) > 1 else tools_to_use[0],
                'confidence': 0.8
            }
            
            state['execution_path'].append("intent_classifier")
            
            # Prepare messages for tool execution
            for tool_name in tools_to_use:
                tool_call = {
                    "name": tool_name,
                    "args": tool_parameters[tool_name],
                    "id": f"call_{tool_name}_{len(state['messages'])}"
                }
                tool_calls.append(tool_call)
            
            # Add AI message with tool calls
            ai_message = AIMessage(
                content="I'll analyze your spending data using the appropriate tools.",
                tool_calls=tool_calls
            )
            state['messages'].append(ai_message)
            
        except Exception as e:
            state['error'] = f"Intent classification error: {str(e)}"
        
        return state
    
    def _response_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate final response based on tool results"""
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial assistant. Based on the tool execution results,
            create a clear, conversational response for the user. 
            
            Guidelines:
            1. Start with a direct answer to their query
            2. Highlight the most important insights
            3. Use specific numbers and percentages
            4. Provide actionable recommendations when appropriate
            5. Keep it conversational but informative
            6. If there are multiple tool results, synthesize them cohesively
            
            Make the response engaging and easy to understand."""),
            ("human", """
            Original Query: {query}
            Tool Results: {results}
            
            Please create a comprehensive response for the user.
            """)
        ])
        
        try:
            # Extract tool results from messages
            tool_results = []
            for message in reversed(state['messages']):
                if hasattr(message, 'content') and isinstance(message.content, list):
                    for content_block in message.content:
                        if hasattr(content_block, 'content'):
                            tool_results.append(content_block.content)
                elif hasattr(message, 'content') and message.content:
                    try:
                        # Try to parse as JSON if it looks like tool output
                        import json
                        parsed_content = json.loads(message.content)
                        if isinstance(parsed_content, dict):
                            tool_results.append(parsed_content)
                    except:
                        pass
            
            # If no tool results found in messages, check if we have analysis_result
            if not tool_results and state.get('analysis_result'):
                tool_results = [state['analysis_result']]
            
            # Generate response
            chain = response_prompt | self.llm
            response = chain.invoke({
                "query": state['user_query'],
                "results": json.dumps(tool_results, indent=2) if tool_results else "No results available"
            })
            
            state['response'] = response.content
            state['execution_path'].append("response_generator")
            
        except Exception as e:
            state['error'] = f"Response generation error: {str(e)}"
            state['response'] = "I apologize, but I encountered an issue generating your response. Please try rephrasing your question."
        
        return state
    
    def _error_handler_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Handle errors gracefully"""
        
        error_message = state.get('error', 'Unknown error occurred')
        
        state['response'] = f"""I apologize, but I encountered an issue while processing your request: {error_message}

Please try:
- Rephrasing your question
- Being more specific about the time period or category
- Asking about a different aspect of your spending

I'm here to help analyze your spending patterns whenever you're ready!"""
        
        state['execution_path'].append("error_handler")
        return state
    
    def _route_after_intent(self, state: SpendingAgentState) -> str:
        """Route after intent classification"""
        if state.get('error'):
            return "error"
        
        intent = state.get('intent')
        if not intent or not intent.get('tools_to_use'):
            return "error"
        
        return "execute_tools"
    
    def process_query(self, client_id: str, user_query: str, config: Dict = None) -> Dict[str, Any]:
        """Process a spending query for a specific client"""
        
        # Initial state
        initial_state = SpendingAgentState(
            client_id=client_id,
            user_query=user_query,
            intent=None,
            analysis_result=None,
            response=None,
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[]
        )
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state, config=config or {})
            
            return {
                "client_id": client_id,
                "query": user_query,
                "response": final_state.get('response'),
                "analysis_result": final_state.get('analysis_result'),
                "intent": final_state.get('intent'),
                "execution_path": final_state.get('execution_path'),
                "error": final_state.get('error'),
                "timestamp": datetime.now().isoformat(),
                "tools_used": final_state.get('intent', {}).get('tools_to_use', [])
            }
        except Exception as e:
            return {
                "client_id": client_id,
                "query": user_query,
                "response": f"I encountered an error processing your request: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema.schema() if hasattr(tool, 'args_schema') else None
            }
            for tool in self.tools
        ]



def demo_spending_agent():
    """Demonstrate the Spending Agent capabilities"""

    
 


if __name__ == "__main__":
    # Run the demo
    demo_spending_agent()
    
    
    print("\n" + "="*60)

