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

# Import the updated DataStore and tools
from data_store.data_store import (
    DataStore,
    generate_sql_for_client_analysis,
    generate_sql_for_budget_analysis,
    execute_generated_sql,
    create_or_update_budget,
    update_budget_tracking_for_month
)

load_dotenv()


# Pydantic models for structured output
class BudgetIntentClassification(BaseModel):
    """Structured intent classification for budget queries"""

    analysis_type: str = Field(
        description="Type of analysis: budget_creation, budget_tracking, budget_optimization, or goal_planning"
    )
    requires_budget_data: bool = Field(
        description="Whether existing budget data is needed"
    )
    requires_transaction_data: bool = Field(
        description="Whether transaction data analysis is needed"
    )
    requires_budget_update: bool = Field(
        description="Whether budget creation/modification is needed"
    )
    query_focus: str = Field(
        description="Main focus: performance_review, budget_setup, overspend_analysis, or optimization"
    )
    time_period: str = Field(
        description="Time scope: current_month, last_month, quarterly, or historical"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )


class BudgetAgentState(TypedDict):
    """Enhanced state for Budget Agent workflow"""

    client_id: int
    user_query: str
    intent: Optional[Dict[str, Any]]
    sql_queries: Optional[List[Dict[str, Any]]]
    raw_data: Optional[List[Dict[str, Any]]]
    budget_operations: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    analysis_type: Optional[str]


class BudgetAgent:
    """Budget Management Agent with SQL-first approach for budget analysis and management"""

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
        memory: bool = True,
    ):
        print(f"💰 Initializing BudgetAgent with budget management capabilities...")
        print(f"📥 Client data: {client_csv_path}")
        print(f"📊 Overall data: {overall_csv_path}")

        # Initialize DataStore with budget support
        self.data_store = DataStore(
            client_csv_path=client_csv_path, overall_csv_path=overall_csv_path
        )

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Set up structured output parser
        self.intent_parser = PydanticOutputParser(
            pydantic_object=BudgetIntentClassification
        )

        # Budget-focused tools
        self.budget_tools = [
            generate_sql_for_client_analysis,
            generate_sql_for_budget_analysis,
            execute_generated_sql,
            create_or_update_budget,
            update_budget_tracking_for_month,
        ]

        # Setup memory (optional)
        self.memory = MemorySaver() if memory else None

        # Build the budget-focused graph
        self.graph = self._build_graph()
        print("✅ BudgetAgent initialized with budget management capabilities!")

    def _build_graph(self) -> StateGraph:
        """Build the budget-focused LangGraph workflow"""

        workflow = StateGraph(BudgetAgentState)

        # Budget-specific workflow nodes
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("budget_analyzer", self._budget_analyzer_node)
        workflow.add_node("budget_executor", self._budget_executor_node)
        workflow.add_node("response_generator", self._response_generator_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point - using string
        workflow.set_entry_point("intent_classifier")

        # Routing logic
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {"analyze_budget": "budget_analyzer", "error": "error_handler"},
        )

        workflow.add_edge("budget_analyzer", "budget_executor")
        workflow.add_edge("budget_executor", "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)

    def _intent_classifier_node(self, state: BudgetAgentState) -> BudgetAgentState:
        """Enhanced intent classification for budget queries"""

        try:
            print(f"💰 [DEBUG] Classifying budget intent for: {state['user_query']}")

            classification_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an AI assistant that classifies user queries about budget management.

Analyze the user's query and determine:

1. **Analysis Type:**
- "budget_creation": Creating new budgets or setting budget limits
- "budget_tracking": Monitoring current budget performance vs actual spending
- "budget_optimization": Improving existing budgets or finding savings
- "goal_planning": Integrating budgets with financial goals

2. **Data Requirements:**
- requires_budget_data: true if need existing budget information
- requires_transaction_data: true if need spending transaction analysis
- requires_budget_update: true if need to create/modify budgets

3. **Query Focus:**
- "performance_review": How am I doing against my budget?
- "budget_setup": Create/modify budget allocations
- "overspend_analysis": Where am I overspending?
- "optimization": How can I improve my budget?

4. **Time Period:**
- "current_month": This month's budget performance
- "last_month": Previous month analysis
- "quarterly": 3-month budget review
- "historical": Long-term budget trends

{format_instructions}

Analyze this budget query and provide structured classification:""",
                    ),
                    ("human", "{user_query}"),
                ]
            )

            formatted_prompt = classification_prompt.partial(
                format_instructions=self.intent_parser.get_format_instructions()
            )

            classification_chain = formatted_prompt | self.llm | self.intent_parser

            try:
                intent_result = classification_chain.invoke(
                    {"user_query": state["user_query"]}
                )

                intent_dict = intent_result.model_dump()

                print(f"[DEBUG] Budget intent classified as: {intent_dict['analysis_type']}")
                print(f"[DEBUG] Focus: {intent_dict['query_focus']}")

                state["intent"] = intent_dict
                state["analysis_type"] = intent_dict["analysis_type"]
                state["execution_path"].append("intent_classifier")

                state["messages"].append(
                    AIMessage(
                        content=f"Budget query classified as {state['analysis_type']} analysis. Processing budget request..."
                    )
                )

            except Exception as parse_error:
                print(f"[DEBUG] Structured parsing failed, using fallback: {parse_error}")

                fallback_result = self._fallback_budget_classification(
                    state["user_query"]
                )
                state["intent"] = fallback_result
                state["analysis_type"] = fallback_result["analysis_type"]
                state["execution_path"].append("intent_classifier")

        except Exception as e:
            print(f"[DEBUG] Budget intent classification failed: {e}")
            state["error"] = f"Budget intent classification error: {str(e)}"

        return state

    def _fallback_budget_classification(self, user_query: str) -> Dict[str, Any]:
        """Fallback budget intent classification using keywords"""

        query_lower = user_query.lower()

        # Budget-specific keyword classification
        creation_keywords = ["create", "set up", "new budget", "budget for", "allocate"]
        tracking_keywords = ["how am i doing", "budget performance", "vs budget", "overspend"]
        optimization_keywords = ["save money", "cut spending", "optimize", "improve budget"]
        
        if any(keyword in query_lower for keyword in creation_keywords):
            analysis_type = "budget_creation"
            query_focus = "budget_setup"
            requires_budget_update = True
        elif any(keyword in query_lower for keyword in tracking_keywords):
            analysis_type = "budget_tracking"
            query_focus = "performance_review"
            requires_budget_update = False
        elif any(keyword in query_lower for keyword in optimization_keywords):
            analysis_type = "budget_optimization"
            query_focus = "optimization"
            requires_budget_update = False
        else:
            analysis_type = "budget_tracking"  # Default
            query_focus = "performance_review"
            requires_budget_update = False

        # Time period detection
        if "this month" in query_lower or "current" in query_lower:
            time_period = "current_month"
        elif "last month" in query_lower:
            time_period = "last_month"
        else:
            time_period = "current_month"

        return {
            "analysis_type": analysis_type,
            "requires_budget_data": True,
            "requires_transaction_data": True,
            "requires_budget_update": requires_budget_update,
            "query_focus": query_focus,
            "time_period": time_period,
            "confidence": 0.7,  # Lower confidence for fallback
        }

    def _budget_analyzer_node(self, state: BudgetAgentState) -> BudgetAgentState:
        """Analyze budget requirements and generate appropriate queries/operations"""

        try:
            print("💰 [DEBUG] Analyzing budget requirements...")

            intent = state.get("intent", {})
            sql_queries = []
            budget_operations = []

            # Generate appropriate queries based on intent
            if intent.get("requires_transaction_data", True):
                try:
                    # Get spending analysis for budget context
                    client_sql_result = generate_sql_for_client_analysis.invoke(
                        {
                            "user_query": state["user_query"],
                            "client_id": state["client_id"],
                        }
                    )

                    if "error" not in client_sql_result:
                        sql_queries.append(client_sql_result)
                        print(f"✅ Generated spending analysis SQL")
                    else:
                        print(f"⚠️ Spending analysis failed: {client_sql_result['error']}")

                except Exception as e:
                    print(f"⚠️ Spending analysis error: {e}")

            # Generate budget-specific queries
            if intent.get("requires_budget_data", True):
                try:
                    budget_sql_result = generate_sql_for_budget_analysis.invoke(
                        {
                            "user_query": state["user_query"],
                            "client_id": state["client_id"],
                            "analysis_type": intent.get("analysis_type", "budget_performance"),
                        }
                    )

                    if "error" not in budget_sql_result:
                        sql_queries.append(budget_sql_result)
                        print(f"✅ Generated budget analysis SQL")
                    else:
                        print(f"⚠️ Budget analysis failed: {budget_sql_result['error']}")

                except Exception as e:
                    print(f"⚠️ Budget analysis error: {e}")

            # Handle budget creation/update requests
            if intent.get("requires_budget_update", False):
                budget_operation = self._extract_budget_creation_params(state["user_query"])
                if budget_operation:
                    budget_operations.append(budget_operation)
                    print(f"✅ Identified budget operation: {budget_operation}")

            state["sql_queries"] = sql_queries
            state["budget_operations"] = budget_operations
            state["execution_path"].append("budget_analyzer")

            print(f"✅ Budget analysis complete: {len(sql_queries)} queries, {len(budget_operations)} operations")

        except Exception as e:
            state["error"] = f"Budget analysis failed: {e}"
            print(f"❌ Budget analysis error: {e}")

        return state

    def _extract_budget_creation_params(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Extract budget creation parameters from user query"""
        
        # This is a simple extraction - in production, you'd use more sophisticated NLP
        query_lower = user_query.lower()
        
        # Look for budget amounts
        import re
        amount_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', user_query)
        
        # Look for categories
        categories = ["groceries", "restaurants", "gas", "utilities", "entertainment", 
                     "shopping", "transportation", "healthcare", "bars", "pharmacy"]
        
        found_category = None
        for category in categories:
            if category in query_lower:
                found_category = category.title()
                break
        
        if amount_match and found_category:
            amount = float(amount_match.group(1).replace(",", ""))
            return {
                "operation_type": "create_budget",
                "category": found_category,
                "monthly_limit": amount,
                "budget_type": "fixed"
            }
        
        return None

    def _budget_executor_node(self, state: BudgetAgentState) -> BudgetAgentState:
        """Execute budget queries and operations"""

        try:
            print("⚡ [DEBUG] Executing budget queries and operations...")

            raw_data = []
            
            # Execute SQL queries first
            sql_queries = state.get("sql_queries", [])
            for i, query_info in enumerate(sql_queries):
                if "sql_query" not in query_info:
                    continue

                try:
                    print(f" Executing query {i+1}: {query_info.get('query_type', 'unknown')}")

                    execution_result = execute_generated_sql.invoke(
                        {
                            "sql_query": query_info["sql_query"],
                            "query_type": query_info.get("query_type", "unknown"),
                        }
                    )

                    if "error" in execution_result:
                        print(f" ❌ Query {i+1} failed: {execution_result['error']}")
                        continue

                    raw_data.append(
                        {
                            "query_type": query_info.get("query_type"),
                            "original_query": query_info.get("original_query"),
                            "sql_executed": query_info["sql_query"],
                            "results": execution_result.get("results", []),
                            "column_names": execution_result.get("column_names", []),
                            "row_count": execution_result.get("row_count", 0),
                            "execution_time": execution_result.get("execution_time_seconds", 0),
                        }
                    )

                    print(f" ✅ Query {i+1} success: {execution_result.get('row_count', 0)} rows")

                except Exception as query_error:
                    print(f" ❌ Query {i+1} execution error: {query_error}")

            # Execute budget operations
            budget_operations = state.get("budget_operations", [])
            for i, operation in enumerate(budget_operations):
                try:
                    if operation.get("operation_type") == "create_budget":
                        print(f" Executing budget operation {i+1}: Create budget")
                        
                        result = create_or_update_budget.invoke(
                            {
                                "client_id": state["client_id"],
                                "category": operation["category"],
                                "monthly_limit": operation["monthly_limit"],
                                "budget_type": operation.get("budget_type", "fixed"),
                            }
                        )

                        raw_data.append(
                            {
                                "query_type": "budget_operation",
                                "operation_type": "create_budget",
                                "results": [result],
                                "column_names": ["operation_result"],
                                "row_count": 1,
                            }
                        )

                        print(f" ✅ Budget operation {i+1}: {result.get('message', 'Completed')}")

                except Exception as op_error:
                    print(f" ❌ Budget operation {i+1} error: {op_error}")

            # Update budget tracking for current month if we have budget data
            current_month = datetime.now().strftime("%Y-%m")
            try:
                tracking_result = update_budget_tracking_for_month.invoke(
                    {
                        "client_id": state["client_id"],
                        "month": current_month,
                    }
                )
                
                if tracking_result.get("success"):
                    print(f" ✅ Budget tracking updated for {current_month}")
                    raw_data.append(
                        {
                            "query_type": "budget_tracking_update",
                            "results": [tracking_result],
                            "column_names": ["tracking_result"],
                            "row_count": 1,
                        }
                    )

            except Exception as tracking_error:
                print(f" ⚠️ Budget tracking update failed: {tracking_error}")

            state["raw_data"] = raw_data
            state["execution_path"].append("budget_executor")

            print(f"✅ Budget execution complete: {len(raw_data)} result sets")

        except Exception as e:
            state["error"] = f"Budget execution failed: {e}"
            print(f"❌ Budget execution error: {e}")

        return state

    def _response_generator_node(self, state: BudgetAgentState) -> BudgetAgentState:
        """Generate natural budget management response"""

        response_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful personal budget advisor analyzing the user's financial data.

The user asked: "{user_query}"

You have access to their real budget and spending data. Your job is to:

1. **Answer directly and naturally** - Sound like a knowledgeable financial advisor
2. **Use the actual numbers** from the data provided
3. **Be practical and actionable** - Give specific budget advice
4. **Provide insights** - Point out spending patterns, budget performance, areas for improvement
5. **Be encouraging** - Frame advice positively and help them achieve their goals

**Response Style:**
- Start directly with the answer or insight
- Use natural, conversational language
- Give specific recommendations when possible
- Highlight both successes and areas for improvement
- Never mention technical details like "SQL" or "database"

Analysis Type: {analysis_type}""",
                ),
                (
                    "human",
                    """Here is the user's actual budget and spending data:

{budget_data}

Please answer their budget question naturally: "{user_query}"

Focus on practical budget advice using the real data shown above.""",
                ),
            ]
        )

        try:
            raw_data = state.get("raw_data", [])

            if not raw_data:
                state["response"] = "I wasn't able to find any budget or spending data for your query. Let me help you set up a budget first!"
                state["execution_path"].append("response_generator")
                return state

            # Extract and format data for LLM
            clean_results = []
            
            for data_chunk in raw_data:
                results = data_chunk.get("results", [])
                query_type = data_chunk.get("query_type", "unknown")
                
                if results:
                    clean_results.append({
                        "data_type": query_type,
                        "data": results
                    })

            # Convert to clean format
            budget_data_text = json.dumps(clean_results, indent=2, default=str)

            print(f" [DEBUG] Sending budget data to LLM:")
            print(f" - Number of data sets: {len(clean_results)}")
            print(f" - Sample: {budget_data_text[:300]}...")

            response = self.llm.invoke(
                response_prompt.format_messages(
                    analysis_type=state.get("analysis_type", "budget_tracking"),
                    user_query=state["user_query"],
                    budget_data=budget_data_text,
                )
            )

            state["response"] = response.content
            state["execution_path"].append("response_generator")

            print(f" [DEBUG] Generated budget response length: {len(response.content)} characters")

        except Exception as e:
            print(f"❌ Budget response generation error: {e}")
            state["response"] = self._generate_budget_fallback_response(state)
            state["execution_path"].append("response_generator")

        return state

    def _generate_budget_fallback_response(self, state: BudgetAgentState) -> str:
        """Generate fallback budget response when main generation fails"""

        try:
            raw_data = state.get("raw_data", [])
            
            if not raw_data:
                return f"I processed your budget question '{state['user_query']}' but couldn't find relevant data. Would you like help setting up a budget?"

            response_parts = [f"I analyzed your budget inquiry: '{state['user_query']}'"]

            for data_chunk in raw_data:
                results = data_chunk.get("results", [])
                query_type = data_chunk.get("query_type", "")
                
                if results and query_type == "budget_operation":
                    result = results[0]
                    if result.get("success"):
                        response_parts.append(f"✅ {result.get('message', 'Budget operation completed')}")
                elif results and "amount" in str(results[0]):
                    # Try to extract spending/budget amounts
                    first_result = results[0]
                    for key, value in first_result.items():
                        if 'amount' in key.lower() and isinstance(value, (int, float)):
                            response_parts.append(f"{key.replace('_', ' ').title()}: ${value:,.2f}")

            if len(response_parts) == 1:
                response_parts.append("I found some budget information. Feel free to ask more specific questions about your budget performance or spending patterns.")

            return "\n\n".join(response_parts)

        except Exception:
            return f"I processed your budget question '{state['user_query']}'. Please try asking more specific questions about your budget or spending."

    def _error_handler_node(self, state: BudgetAgentState) -> BudgetAgentState:
        """Handle budget-specific errors with helpful suggestions"""

        error_message = state.get("error", "Unknown budget error occurred")
        print(f"🔧 [DEBUG] Handling budget error: {error_message}")

        if "Budget intent" in error_message:
            suggestion = "Try rephrasing your budget question. For example:\n- 'How am I doing against my budget this month?'\n- 'Create a $500 budget for groceries'\n- 'Where am I overspending?'"
        elif "Budget analysis" in error_message:
            suggestion = "There was an issue analyzing your budget. Try asking about:\n- Current budget performance\n- Setting up new budgets\n- Spending comparisons"
        elif "Budget execution" in error_message:
            suggestion = "There was an issue accessing your budget data. You might need to set up budgets first."
        else:
            suggestion = "Try asking specific budget questions like budget performance, creating budgets, or spending analysis."

        state["response"] = f"""I encountered an issue while processing your budget request: {error_message}

💡 **Suggestion:** {suggestion}

I can help you with budget management, spending analysis, and financial planning. Feel free to try again!"""

        state["execution_path"].append("error_handler")
        return state

    def _route_after_intent(self, state: BudgetAgentState) -> str:
        """Route based on budget intent classification"""

        if state.get("error"):
            return "error"

        intent = state.get("intent")
        if not intent:
            state["error"] = "No budget intent was classified"
            return "error"

        if not intent.get("analysis_type"):
            state["error"] = "Budget intent classification incomplete"
            return "error"

        return "analyze_budget"

    def process_query(
        self, client_id: int, user_query: str, config: Dict = None
    ) -> Dict[str, Any]:
        """Process a budget management query"""

        initial_state = BudgetAgentState(
            client_id=client_id,
            user_query=user_query,
            intent=None,
            sql_queries=None,
            raw_data=None,
            budget_operations=None,
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
                "budget_operations": len(final_state.get("budget_operations", [])),
                "execution_path": final_state.get("execution_path"),
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"❌ Budget graph execution error: {e}")
            return {
                "client_id": client_id,
                "query": user_query,
                "response": "I encountered a system error while processing your budget request. Please try again with a simpler query.",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
            }


def test_budget_agent():
    """Test the complete BudgetAgent workflow"""

    print("💰 TESTING BUDGET MANAGEMENT AGENT")
    print("=" * 60)

    client_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv"
    overall_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/overall_data.csv"

    try:
        agent = BudgetAgent(
            client_csv_path=client_csv, overall_csv_path=overall_csv, memory=False
        )
        
        test_queries = [
            "Create a $800 budget for groceries",
            "How am I doing against my budget this month?",
            "Where am I overspending?",
            "Set up a $300 budget for restaurants"
        ]

        for query in test_queries:
            print(f"\n💰 Testing: '{query}'")
            print("-" * 40)

            try:
                result = agent.process_query(client_id=430, user_query=query)

                print(f"✅ Success: {result.get('success', False)}")
                print(f"📊 Analysis Type: {result.get('analysis_type', 'N/A')}")
                print(f"🔧 SQL Queries: {result.get('sql_queries', 0)}")
                print(f"💰 Budget Operations: {result.get('budget_operations', 0)}")

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
        print(f"❌ Failed to initialize budget agent: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    test_budget_agent()