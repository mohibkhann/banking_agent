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
from langgraph.checkpoint.memory import MemorySaver
# LangGraph imports
from langgraph.graph import StateGraph, END
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
from data_store.data_store import (
DataStore,
generate_sql_for_client_analysis,
generate_sql_for_benchmark_analysis,
execute_generated_sql
)

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
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    analysis_type: Optional[str]


class SpendingAgent:
    """Simplified SQL-first LangGraph-based Spending Agent that passes raw SQL results directly to LLM"""

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
        self.memory = MemorySaver() if memory else None

        # Build the simplified graph
        self.graph = self._build_graph()
        print("✅ SpendingAgent initialized with SQL-first capabilities!")

    def _build_graph(self) -> StateGraph:
        """Build the simplified SQL-first LangGraph workflow"""

        workflow = StateGraph(SpendingAgentState)

        # Simplified workflow nodes - removed data_analyzer
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("sql_generator", self._sql_generator_node)
        workflow.add_node("sql_executor", self._sql_executor_node)
        workflow.add_node("response_generator", self._response_generator_node)  # Now works directly with SQL results
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point - using string instead of START
        workflow.set_entry_point("intent_classifier")

        # Simplified routing
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {"generate_sql": "sql_generator", "error": "error_handler"},
        )

        workflow.add_edge("sql_generator", "sql_executor")
        workflow.add_edge("sql_executor", "response_generator")  # Direct path
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

                    # Store successful result with more context
                    raw_data.append(
                        {
                            "query_type": query_info.get("query_type"),
                            "original_query": query_info.get("original_query"),
                            "sql_executed": query_info["sql_query"],
                            "results": execution_result.get("results", []),
                            "column_names": execution_result.get("column_names", []),
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

            # Debug: Show actual SQL results
            for i, data_chunk in enumerate(raw_data):
                print(f"🔍 [DEBUG] Results for query {i+1}:")
                print(f"  SQL: {data_chunk['sql_executed']}")
                print(f"  Columns: {data_chunk['column_names']}")
                print(f"  Rows: {len(data_chunk['results'])}")
                if data_chunk['results']:
                    print(f"  Sample result: {data_chunk['results'][0]}")

        except Exception as e:
            state["error"] = f"SQL execution failed: {e}"
            print(f"❌ SQL execution error: {e}")

        return state

    def _response_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate response directly from SQL results without intermediate analysis"""

        response_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful personal finance assistant analyzing the user's spending data.

The user asked: "{user_query}"

You have access to their real transaction data and need to provide a natural, conversational response. Your job is to:

1. **Answer directly and naturally** - Don't mention SQL queries, databases, or technical details
2. **Use the actual numbers** from the data provided
3. **Be conversational and helpful** - Sound like a knowledgeable friend, not a robot
4. **Provide insights when possible** - Point out interesting patterns or provide context
5. **Be accurate** - Only use the real data provided, never make up numbers

**Response Style:**
- Start directly with the answer (e.g., "You spent $1,234 last month")
- Use natural language, not technical jargon
- Provide helpful context or insights when relevant
- Be concise but informative
- Never mention "SQL queries", "database results", or technical processes

Analysis Type: {analysis_type}""",
                ),
                (
                    "human",
                    """Here is the user's actual spending data:

{sql_results}

Please answer their question naturally: "{user_query}"

Remember: Give a direct, conversational answer using only the real data shown above.""",
                ),
            ]
        )

        try:
            raw_data = state.get("raw_data", [])

            if not raw_data:
                state["response"] = "I wasn't able to find any spending data for your query. Please try rephrasing your question."
                state["execution_path"].append("response_generator")
                return state

            # Extract just the essential data for the LLM (cleaner format)
            clean_results = []
            
            for data_chunk in raw_data:
                results = data_chunk.get("results", [])
                if results:
                    clean_results.extend(results)

            # Convert to a simple, clean format
            if len(clean_results) == 1 and len(clean_results[0]) == 1:
                # Single value result - just pass the key-value pair
                key, value = list(clean_results[0].items())[0]
                results_text = f"{key}: {value}"
            else:
                # Multiple results - format as clean JSON
                results_text = json.dumps(clean_results, indent=2, default=str)

            print(f" [DEBUG] Sending clean data to LLM:")
            print(f" - Results format: {type(results_text)}")
            print(f" - Sample: {results_text[:200]}...")

            response = self.llm.invoke(
                response_prompt.format_messages(
                    analysis_type=state.get("analysis_type", "personal"),
                    user_query=state["user_query"],
                    sql_results=results_text,
                )
            )

            state["response"] = response.content
            state["execution_path"].append("response_generator")

            print(f" [DEBUG] Generated response length: {len(response.content)} characters")

        except Exception as e:
            print(f"❌ Response generation error: {e}")
            # Provide a basic response based on available data
            state["response"] = self._generate_fallback_response(state)
            state["execution_path"].append("response_generator")

        return state

    def _generate_fallback_response(self, state: SpendingAgentState) -> str:
        """Generate a fallback response when main response generation fails"""

        try:
            raw_data = state.get("raw_data", [])

            if not raw_data:
                return f"I processed your query '{state['user_query']}' but couldn't retrieve any data. Please try a different question."

            response_parts = [f"I analyzed your query: '{state['user_query']}'"]

            for data_chunk in raw_data:
                results = data_chunk.get("results", [])
                if results:
                    # Try to extract basic information from first result
                    first_result = results[0]
                    
                    # Look for common spending-related fields
                    for key, value in first_result.items():
                        if key.lower().startswith('total'):
                            response_parts.append(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
                        elif 'amount' in key.lower():
                            response_parts.append(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
                        elif 'count' in key.lower():
                            response_parts.append(f"{key.replace('_', ' ').title()}: {value}")

            if len(response_parts) == 1:
                response_parts.append("The query executed successfully but returned limited information.")

            return "\n\n".join(response_parts)

        except Exception:
            return f"I processed your query '{state['user_query']}' and retrieved some data. Please try asking more specific questions about your spending patterns."

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
    """Test the complete SpendingAgent workflow with various queries"""

    print("🧪 TESTING SIMPLIFIED SPENDING AGENT WORKFLOW")
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
            "How much did I spend last month?",
            "Show me my spending by category last month",
            "What's my average transaction amount?"
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
    test_full_spending_agent()