import json
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
# LangGraph imports
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )
)

# Import the updated DataStore and SQL tools
from banking_agent.data_store.data_store import (
DataStore,
generate_sql_for_client_analysis,
generate_sql_for_benchmark_analysis,
execute_generated_sql
)
# from data_store.data_store import (
#     DataStore,
#     execute_generated_sql,
#     generate_sql_for_benchmark_analysis,
#     generate_sql_for_client_analysis,
# )

load_dotenv()


# Pydantic models for structured output
class IntentClassification(BaseModel):
    """Structured intent classification model"""

    analysis_type: str = Field(
        description="Type of analysis: personal, comparative, or hybrid"
    )
    requires_client_data: bool = Field(
        description="Whether client-specific data is needed"
    )
    requires_benchmark_data: bool = Field(
        description="Whether market benchmark data is needed"
    )
    query_focus: str = Field(
        description="Main focus: spending_summary, category_analysis, time_patterns, or comparison"
    )
    time_period: str = Field(
        description="Time scope: last_month, last_quarter, last_year, specific_dates, or all_time"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )


class SpendingAgentState(TypedDict):
    """Enhanced state for SQL-first Spending Agent workflow"""

    client_id: int
    user_query: str
    intent: Optional[Dict[str, Any]]
    sql_queries: Optional[List[Dict[str, Any]]]
    raw_data: Optional[List[Dict[str, Any]]]
    analysis_result: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    analysis_type: Optional[str]


class SpendingAgent:
    """SQL-first LangGraph-based Spending Agent with structured output parsing"""

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
        memory: bool = True,
    ):
        print(f"🚀 Initializing SpendingAgent with SQL-first approach...")
        print(f"📥 Client data: {client_csv_path}")
        print(f"📊 Overall data: {overall_csv_path}")

        # Initialize DataStore with optimized loading
        self.data_store = DataStore(
            client_csv_path=client_csv_path, overall_csv_path=overall_csv_path
        )

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Set up structured output parser
        self.intent_parser = PydanticOutputParser(
            pydantic_object=IntentClassification
        )

        # SQL-powered tools
        self.sql_tools = [
            generate_sql_for_client_analysis,
            generate_sql_for_benchmark_analysis,
            execute_generated_sql,
        ]

        # Setup memory (optional)
        self.memory = SqliteSaver.from_conn_string(":memory:") if memory else None

        # Build the enhanced graph
        self.graph = self._build_graph()
        print("✅ SpendingAgent initialized with SQL-first capabilities!")

    def _build_graph(self) -> StateGraph:
        """Build the SQL-first LangGraph workflow"""

        workflow = StateGraph(SpendingAgentState)

        # Enhanced workflow nodes
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("sql_generator", self._sql_generator_node)
        workflow.add_node("sql_executor", self._sql_executor_node)
        workflow.add_node("data_analyzer", self._data_analyzer_node)
        workflow.add_node("response_generator", self._response_generator_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point
        workflow.set_entry_point("intent_classifier")

        # Enhanced routing with better error handling
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {"generate_sql": "sql_generator", "error": "error_handler"},
        )

        workflow.add_edge("sql_generator", "sql_executor")
        workflow.add_edge("sql_executor", "data_analyzer")
        workflow.add_edge("data_analyzer", "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)

    def _intent_classifier_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Enhanced intent classification with structured output parsing"""

        try:
            print(f"🧠 [DEBUG] Classifying intent for: {state['user_query']}")

            # Create structured prompt with output parser
            classification_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an AI assistant that classifies user queries about spending analytics.

Analyze the user's query and determine:

1. **Analysis Type:**
- "personal": Focus only on client's personal spending (e.g., "How much did I spend?")
- "comparative": Compare to market/benchmarks (e.g., "How do I compare to others?")
- "hybrid": Both personal + comparison (e.g., "My spending vs market average")

2. **Data Requirements:**
- requires_client_data: true if need client's personal transactions
- requires_benchmark_data: true if need market comparison data

3. **Query Focus:**
- "spending_summary": Overall spending amounts, totals
- "category_analysis": Spending by categories (restaurants, shopping, etc.)
- "time_patterns": Time-based analysis (weekend, night, seasonal)
- "comparison": Direct comparisons to market/peers

4. **Time Period:**
- "last_month": Recent month analysis
- "last_quarter": 3-month analysis
- "last_year": Annual analysis
- "all_time": Full historical analysis
- "specific_dates": User specified date range

{format_instructions}

Analyze this query and provide structured classification:""",
                    ),
                    ("human", "{user_query}"),
                ]
            )

            # Format the prompt with parser instructions
            formatted_prompt = classification_prompt.partial(
                format_instructions=self.intent_parser.get_format_instructions()
            )

            # Create chain with structured output
            classification_chain = formatted_prompt | self.llm | self.intent_parser

            # Invoke with error handling
            try:
                intent_result = classification_chain.invoke(
                    {"user_query": state["user_query"]}
                )

                # Convert Pydantic model to dict
                intent_dict = intent_result.model_dump()

                print(f"[DEBUG] Intent classified as: {intent_dict['analysis_type']}")
                print(f"[DEBUG] Confidence: {intent_dict['confidence']}")

                state["intent"] = intent_dict
                state["analysis_type"] = intent_dict["analysis_type"]
                state["execution_path"].append("intent_classifier")

                # Add message for next step
                state["messages"].append(
                    AIMessage(
                        content=f"Query classified as {state['analysis_type']} analysis (confidence: {intent_dict['confidence']:.2f}). Generating SQL queries..."
                    )
                )

            except Exception as parse_error:
                print(f"[DEBUG] Structured parsing failed, trying fallback: {parse_error}")

                # Fallback to simple classification
                fallback_result = self._fallback_intent_classification(
                    state["user_query"]
                )
                state["intent"] = fallback_result
                state["analysis_type"] = fallback_result["analysis_type"]
                state["execution_path"].append("intent_classifier")

                state["messages"].append(
                    AIMessage(
                        content=f"Query classified as {state['analysis_type']} analysis (fallback method). Generating SQL queries..."
                    )
                )

        except Exception as e:
            print(f"[DEBUG] Intent classification failed completely: {e}")
            state["error"] = f"Intent classification error: {str(e)}"

        return state

    def _fallback_intent_classification(self, user_query: str) -> Dict[str, Any]:
        """Fallback intent classification using simple rules"""

        query_lower = user_query.lower()

        # Simple keyword-based classification
        comparative_keywords = [
            "compare",
            "average",
            "others",
            "typical",
            "normal",
            "benchmark",
            "market",
        ]
        personal_keywords = ["my", "i spent", "how much", "total", "summary"]
        category_keywords = [
            "category",
            "categories",
            "restaurant",
            "grocery",
            "shopping",
            "gas",
        ]
        time_keywords = ["last month", "month", "week", "year", "recent"]

        # Determine analysis type
        if any(keyword in query_lower for keyword in comparative_keywords):
            analysis_type = "comparative"
            requires_benchmark_data = True
        elif any(keyword in query_lower for keyword in personal_keywords):
            analysis_type = "personal"
            requires_benchmark_data = False
        else:
            analysis_type = "personal"  # Default
            requires_benchmark_data = False

        # Determine query focus
        if any(keyword in query_lower for keyword in category_keywords):
            query_focus = "category_analysis"
        elif "compare" in query_lower:
            query_focus = "comparison"
        else:
            query_focus = "spending_summary"

        # Determine time period
        if "last month" in query_lower or "month" in query_lower:
            time_period = "last_month"
        elif "year" in query_lower:
            time_period = "last_year"
        else:
            time_period = "last_month"  # Default

        return {
            "analysis_type": analysis_type,
            "requires_client_data": True,  # Always need client data
            "requires_benchmark_data": requires_benchmark_data,
            "query_focus": query_focus,
            "time_period": time_period,
            "confidence": 0.7,  # Lower confidence for fallback
        }

    def _sql_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate appropriate SQL queries based on intent"""

        try:
            print("🔧 [DEBUG] Generating SQL queries...")

            intent = state.get("intent", {})
            sql_queries = []

            # Generate client analysis SQL if needed
            if intent.get("requires_client_data", True):
                try:
                    client_sql_result = generate_sql_for_client_analysis.invoke(
                        {
                            "user_query": state["user_query"],
                            "client_id": state["client_id"],
                        }
                    )

                    if "error" not in client_sql_result:
                        sql_queries.append(client_sql_result)
                        print(
                            f"✅ Generated client SQL: {client_sql_result.get('query_type', 'unknown')}"
                        )
                    else:
                        print(
                            f"⚠️ Client SQL generation failed: {client_sql_result['error']}"
                        )

                except Exception as e:
                    print(f"⚠️ Client SQL generation error: {e}")

            # Generate benchmark analysis SQL if needed
            if intent.get("requires_benchmark_data", False):
                try:
                    demographic_filters = self._get_client_demographics(
                        state["client_id"]
                    )

                    benchmark_sql_result = generate_sql_for_benchmark_analysis.invoke(
                        {
                            "user_query": state["user_query"],
                            "demographic_filters": demographic_filters,
                        }
                    )

                    if "error" not in benchmark_sql_result:
                        sql_queries.append(benchmark_sql_result)
                        print(
                            f"✅ Generated benchmark SQL: {benchmark_sql_result.get('query_type', 'unknown')}"
                        )
                    else:
                        print(
                            f"⚠️ Benchmark SQL generation failed: {benchmark_sql_result['error']}"
                        )

                except Exception as e:
                    print(f"⚠️ Benchmark SQL generation error: {e}")

            if not sql_queries:
                raise ValueError("No SQL queries were successfully generated")

            state["sql_queries"] = sql_queries
            state["execution_path"].append("sql_generator")

            print(f"✅ Generated {len(sql_queries)} SQL queries successfully")
            print(json.dumps([q["sql_query"] for q in sql_queries], indent=2))


        except Exception as e:
            state["error"] = f"SQL generation failed: {e}"
            print(f"❌ SQL generation error: {e}")

        return state

    def _sql_executor_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Execute generated SQL queries with enhanced error handling"""

        try:
            print("⚡ [DEBUG] Executing SQL queries...")

            raw_data = []
            sql_queries = state.get("sql_queries", [])

            for i, query_info in enumerate(sql_queries):
                if "sql_query" not in query_info:
                    print(f"⚠️ Query {i+1}: Missing sql_query field")
                    continue

                try:
                    print(
                        f" Executing query {i+1}: {query_info.get('query_type', 'unknown')}"
                    )

                    execution_result = execute_generated_sql.invoke(
                        {
                            "sql_query": query_info["sql_query"],
                            "query_type": query_info.get("query_type", "unknown"),
                        }
                    )

                    if "error" in execution_result:
                        print(f" ❌ Query {i+1} failed: {execution_result['error']}")
                        continue

                    # Store successful result
                    raw_data.append(
                        {
                            "query_type": query_info.get("query_type"),
                            "original_query": query_info.get("original_query"),
                            "sql_executed": query_info["sql_query"],
                            "results": execution_result.get("results", []),
                            "row_count": execution_result.get("row_count", 0),
                            "execution_time": execution_result.get(
                                "execution_time_seconds", 0
                            ),
                        }
                    )

                    print(
                        f" ✅ Query {i+1} success: {execution_result.get('row_count', 0)} rows in {execution_result.get('execution_time_seconds', 0):.3f}s"
                    )

                except Exception as query_error:
                    print(f" ❌ Query {i+1} execution error: {query_error}")
                    continue

            if not raw_data:
                raise ValueError("No SQL queries executed successfully")

            state["raw_data"] = raw_data
            state["execution_path"].append("sql_executor")

            print(f"✅ Successfully executed {len(raw_data)} queries")

            print(f"🔍 [DEBUG] Results for query {i+1}:")
            print(json.dumps(raw_data[-1], indent=2, default=str))

        except Exception as e:
            state["error"] = f"SQL execution failed: {e}"
            print(f"❌ SQL execution error: {e}")

        return state

    def _data_analyzer_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Analyze raw SQL results using local processing"""

        try:
            print("📊 [DEBUG] Analyzing raw data...")

            raw_data = state.get("raw_data", [])
            analysis_results = []

            for data_chunk in raw_data:
                query_type = data_chunk.get("query_type")
                results = data_chunk.get("results", [])

                if not results:
                    print(f" ⚠️ No results for {query_type}")
                    continue

                try:
                    # Convert to DataFrame for analysis
                    df = pd.DataFrame(results)
                    print(f" 📈 Analyzing {len(df)} rows for {query_type}")

                    if query_type == "client_analysis":
                        personal_analysis = self._analyze_personal_spending(df, state)
                        analysis_results.append(
                            {"type": "personal_analysis", "data": personal_analysis}
                        )

                    elif query_type == "benchmark_analysis":
                        benchmark_analysis = self._analyze_benchmark_data(df, state)
                        analysis_results.append(
                            {"type": "benchmark_analysis", "data": benchmark_analysis}
                        )

                except Exception as analysis_error:
                    print(f" ❌ Analysis error for {query_type}: {analysis_error}")
                    # Continue with other analyses
                    continue

            if not analysis_results:
                # Create minimal analysis from raw data
                analysis_results = [
                    {
                        "type": "basic_analysis",
                        "data": {
                            "raw_data_summary": f"Retrieved {len(raw_data)} data chunks"
                        },
                    }
                ]

            state["analysis_result"] = analysis_results
            state["execution_path"].append("data_analyzer")

            print(f"This is the analysis result {state["analysis_result"]}")

            print(f"✅ Completed analysis: {len(analysis_results)} result sets")

        except Exception as e:
            state["error"] = f"Data analysis failed: {e}"
            print(f"❌ Analysis error: {e}")

        return state

    def _analyze_personal_spending(
        self, df: pd.DataFrame, state: SpendingAgentState
    ) -> Dict[str, Any]:
        """Analyze personal spending data with safe calculations and rich insights"""

        if df.empty:
            return {"error": "No personal spending data available"}

        analysis = {}

        try:
            print(f" [DEBUG] Analyzing {len(df)} rows of client data")
            print(f" [DEBUG] Columns available: {list(df.columns)}")
            print(
                f" [DEBUG] Sample data: {df.head(1).to_dict('records') if not df.empty else 'None'}"
            )

            # Basic spending metrics
            if "amount" in df.columns:
                amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

                analysis["spending_summary"] = {
                    "total_amount": float(amounts.sum()),
                    "transaction_count": len(df),
                    "average_transaction": float(amounts.mean())
                    if len(amounts) > 0
                    else 0,
                    "median_transaction": float(amounts.median())
                    if len(amounts) > 0
                    else 0,
                    "max_transaction": float(amounts.max()) if len(amounts) > 0 else 0,
                    "min_transaction": float(amounts.min()) if len(amounts) > 0 else 0,
                }

                print(
                    f" [DEBUG] Spending summary: ${amounts.sum():.2f} total, {len(df)} transactions"
                )

            # Category breakdown if available
            if "mcc_category" in df.columns and "amount" in df.columns:
                try:
                    amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                    df_with_amounts = df.copy()
                    df_with_amounts["amount"] = amounts

                    category_analysis = (
                        df_with_amounts.groupby("mcc_category")["amount"]
                        .agg(["sum", "count", "mean"])
                        .round(2)
                    )
                    category_analysis = category_analysis.sort_values(
                        "sum", ascending=False
                    )

                    # Convert to dict format that's easy for LLM to understand
                    category_breakdown = {}
                    for category in category_analysis.index:
                        category_breakdown[category] = {
                            "total_spent": float(category_analysis.loc[category, "sum"]),
                            "transaction_count": int(
                                category_analysis.loc[category, "count"]
                            ),
                            "average_per_transaction": float(
                                category_analysis.loc[category, "mean"]
                            ),
                        }

                    analysis["category_breakdown"] = {
                        "categories": category_breakdown,
                        "top_category": category_analysis.index[0]
                        if len(category_analysis) > 0
                        else "Unknown",
                        "total_categories": len(category_analysis),
                    }

                    print(
                        f" [DEBUG] Found {len(category_analysis)} categories, top: {category_analysis.index[0] if len(category_analysis) > 0 else 'None'}"
                    )

                except Exception as cat_error:
                    print(f" ⚠️ Category analysis error: {cat_error}")

            # Add raw data sample for LLM context
            if not df.empty:
                # Get a sample of actual transactions for context
                sample_size = min(5, len(df))
                sample_data = df.head(sample_size)

                analysis["sample_transactions"] = []
                for _, row in sample_data.iterrows():
                    transaction = {}
                    for col in ["date", "amount", "mcc_category", "merchant_city"]:
                        if col in row:
                            transaction[col] = row[col]
                    analysis["sample_transactions"].append(transaction)

        except Exception as e:
            analysis["error"] = f"Personal analysis error: {e}"
            print(f" ❌ Analysis error: {e}")

        return analysis

    def _analyze_benchmark_data(
        self, df: pd.DataFrame, state: SpendingAgentState
    ) -> Dict[str, Any]:
        """Analyze benchmark data with safe calculations"""

        if df.empty:
            return {"error": "No benchmark data available"}

        analysis = {}

        try:
            # Market averages
            if "amount" in df.columns:
                amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                analysis["market_benchmarks"] = {
                    "market_average": float(amounts.mean()) if len(amounts) > 0 else 0,
                    "market_median": float(amounts.median()) if len(amounts) > 0 else 0,
                    "sample_size": len(df),
                }

            # Demographic breakdowns if available
            if "current_age" in df.columns and "amount" in df.columns:
                try:
                    amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                    df_with_amounts = df.copy()
                    df_with_amounts["amount"] = amounts

                    age_analysis = (
                        df_with_amounts.groupby("current_age")["amount"].mean().to_dict()
                    )
                    analysis["demographic_patterns"] = {
                        "spending_by_age": {
                            str(k): float(v) for k, v in age_analysis.items()
                        }
                    }
                except Exception as demo_error:
                    print(f" ⚠️ Demographic analysis error: {demo_error}")

        except Exception as e:
            analysis["error"] = f"Benchmark analysis error: {e}"

        return analysis

    def _response_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate comprehensive response with actual data insights"""

        response_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert financial advisor analyzing real banking transaction data.

The user asked: "{user_query}"

You have successfully retrieved and analyzed their actual spending data. The analysis results contain REAL numbers and transactions from their account.

Your job is to:
1. **Use the specific numbers** from the analysis results
2. **Answer their exact question** with real data
3. **Provide actionable insights** based on the actual spending patterns
4. **Be specific and helpful** - use actual amounts, categories, and transaction counts

NEVER say "the data provided doesn't include" or give generic advice. The analysis contains REAL transaction data that you should analyze and present clearly.

Analysis Type: {analysis_type}""",
                ),
                (
                    "human",
                    """Based on this REAL analysis of the user's actual spending data:

{results}

Please provide a specific, data-driven response to their question: "{user_query}"

Focus on the actual numbers and patterns in their spending data.""",
                ),
            ]
        )

        try:
            results = state.get("analysis_result", [])

            # Ensure we have something to work with
            if not results:
                results = [
                    {
                        "type": "basic",
                        "data": {
                            "message": "Analysis completed but limited data available"
                        },
                    }
                ]

            # Convert to JSON string safely
            try:
                results_json = json.dumps(results, indent=2, default=str)
            except Exception as json_error:
                results_json = str(results)

            print(f" [DEBUG] Sending to LLM: {len(results)} analysis results")
            print(
                f" [DEBUG] Sample result: {str(results[0])[:200]}..."
                if results
                else "No results"
            )

            response = self.llm.invoke(
                response_prompt.format_messages(
                    analysis_type=state.get("analysis_type", "personal"),
                    user_query=state["user_query"],
                    results=results_json,
                )
            )

            state["response"] = response.content
            state["execution_path"].append("response_generator")

            print(
                f" [DEBUG] Generated response length: {len(response.content)} characters"
            )

        except Exception as e:
            print(f"❌ Response generation error: {e}")
            # Provide a basic response based on available data
            state["response"] = self._generate_fallback_response(state)
            state["execution_path"].append("response_generator")

        return state

    def _generate_fallback_response(self, state: SpendingAgentState) -> str:
        """Generate a fallback response when main response generation fails"""

        try:
            analysis_results = state.get("analysis_result", [])

            # Extract basic info from analysis results
            response_parts = [f"I analyzed your query: '{state['user_query']}'"]

            for result in analysis_results:
                if result.get("type") == "personal_analysis":
                    data = result.get("data", {})
                    if "spending_summary" in data:
                        summary = data["spending_summary"]
                        total = summary.get("total_amount", 0)
                        count = summary.get("transaction_count", 0)
                        avg = summary.get("average_transaction", 0)

                        response_parts.append(
                            f"Found {count} transactions totaling ${total:,.2f}"
                        )
                        if avg > 0:
                            response_parts.append(f"Average transaction: ${avg:.2f}")

                    if "category_breakdown" in data:
                        categories = data["category_breakdown"].get("categories", {})
                        if categories:
                            top_category = max(
                                categories.items(), key=lambda x: x[1]["total_spent"]
                            )
                            response_parts.append(
                                f"Top spending category: {top_category[0]} (${top_category[1]['total_spent']:,.2f})"
                            )

            if len(response_parts) == 1:
                response_parts.append(
                    "The analysis completed successfully. You can try asking more specific questions about your spending patterns."
                )

            return "\n\n".join(response_parts)

        except Exception:
            return f"I processed your query '{state['user_query']}' and found some spending data. Please try asking more specific questions about your spending patterns."

    def _error_handler_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Enhanced error handling with helpful suggestions"""

        error_message = state.get("error", "Unknown error occurred")
        print(f"🔧 [DEBUG] Handling error: {error_message}")

        # Provide specific help based on error type
        if "Intent classification" in error_message:
            suggestion = "Try rephrasing your question more clearly. For example:\n- 'How much did I spend last month?'\n- 'Show me my spending by category'\n- 'Compare my spending to others'"
        elif "SQL generation" in error_message:
            suggestion = "There was an issue generating the database query. Try asking about:\n- Total spending amounts\n- Spending by category\n- Recent transactions"
        elif "SQL execution" in error_message:
            suggestion = (
                "There was a database issue. Please check that your data is properly loaded."
            )
        else:
            suggestion = "Try asking more specific questions about your spending patterns."

        state[
            "response"
        ] = f"""I encountered an issue while analyzing your spending: {error_message}

🔧 **Suggestion:** {suggestion}

I have access to your transaction data and can help with various spending analyses. Feel free to try again!"""

        state["execution_path"].append("error_handler")
        return state

    def _route_after_intent(self, state: SpendingAgentState) -> str:
        """Enhanced routing logic with better error checking"""

        if state.get("error"):
            return "error"

        intent = state.get("intent")
        if not intent:
            state["error"] = "No intent was classified"
            return "error"

        # Check if we have valid intent data
        if not intent.get("analysis_type"):
            state["error"] = "Intent classification incomplete"
            return "error"

        return "generate_sql"

    def _get_client_demographics(self, client_id: int) -> Dict[str, Any]:
        """Get client demographics for benchmark filtering with error handling"""

        try:
            client_data = self.data_store.get_client_data(client_id)

            if not client_data.empty:
                first_row = client_data.iloc[0]
                return {
                    "age_min": max(18, int(first_row.get("current_age", 25)) - 5),
                    "age_max": min(80, int(first_row.get("current_age", 35)) + 5),
                    "gender": str(first_row.get("gender", "M")),
                    "income_min": max(
                        0, float(first_row.get("yearly_income", 50000)) * 0.8
                    ),
                }
        except Exception as e:
            print(f"⚠️ Could not get client demographics: {e}")

        # Return default demographics if client data unavailable
        return {"age_min": 25, "age_max": 35, "gender": "M", "income_min": 40000}

    def process_query(
        self, client_id: int, user_query: str, config: Dict = None
    ) -> Dict[str, Any]:
        """Process a spending query with comprehensive error handling"""

        initial_state = SpendingAgentState(
            client_id=client_id,
            user_query=user_query,
            intent=None,
            sql_queries=None,
            raw_data=None,
            analysis_result=None,
            response=None,
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[],
            analysis_type=None,
        )

        try:
            final_state = self.graph.invoke(initial_state, config=config or {})

            return {
                "client_id": client_id,
                "query": user_query,
                "response": final_state.get("response"),
                "analysis_type": final_state.get("analysis_type"),
                "sql_queries": len(final_state.get("sql_queries", [])),
                "execution_path": final_state.get("execution_path"),
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"❌ Graph execution error: {e}")
            return {
                "client_id": client_id,
                "query": user_query,
                "response": "I encountered a system error while processing your request. Please try again with a simpler query.",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
            }


def test_full_spending_agent():
    """Test the complete SpendingAgent workflow"""

    print("🧪 TESTING COMPLETE SPENDING AGENT WORKFLOW")
    print("=" * 60)

    client_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv"
    overall_csv = (
        "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/overall_data.csv"
    )

    try:
        agent = SpendingAgent(
            client_csv_path=client_csv, overall_csv_path=overall_csv, memory=False
        )
        test_queries = [
            "How much did I spend last month?"
        ]

        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            print("-" * 40)

            try:
                result = agent.process_query(client_id=430, user_query=query)

                print(f"✅ Success: {result.get('success', False)}")
                print(f"📊 Analysis Type: {result.get('analysis_type', 'N/A')}")
                print(f"🔧 SQL Queries: {result.get('sql_queries', 0)}")


                # Show the actual response
                response = result.get("response", "No response")
                print("\n💬 Response:")
                print(response)

                if result.get("error"):
                    print(f"❌ Error: {result['error']}")

            except Exception as e:
                print(f"❌ Test Error: {e}")

            print("\n" + "." * 40)

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")



            

if __name__ == "__main__":

    print("\n" + "=" * 60)

    # Then test the full agent
    test_full_spending_agent()
