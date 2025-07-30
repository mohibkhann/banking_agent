import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime, timedelta
import json
import operator
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
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


from banking_agent.data_store.data_store import DataStore
from banking_agent.tools.tools import (
    get_spending_by_category,
    get_spending_by_category_date,
    get_spending_by_night,
    get_spending_summary,
    analyze_time_patterns
    
)

class SpendingAgentState(TypedDict):
    """State for the Spending Agent workflow"""
    client_id: int
    user_query: str
    intent: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
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
            get_spending_by_category,
            get_spending_by_category_date, 
            get_spending_by_night,
            analyze_time_patterns
            
        ]
        
        # Create tool node--
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
        workflow.add_node("collector", self._collect_tool_outputs)
        workflow.add_edge("tool_executor", "collector")
        workflow.add_edge("collector", "response_generator")

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
            ("system", """You are a domain expert at translating natural‑language spending queries into precise tool invocations. Follow this workflow:

        1. **Interpret Intent**  
        - Identify if the user wants a summary of spend, a time‑series, a category breakdown, a night/day comparison, or multiple analyses.

        2. **Select Tools**  
        Choose from the following functions exactly as defined:

        1. **get_spending_summary(client_id: int, start_date: str, end_date: str)**  
            • Returns a dict with:  
                - date_range: {start, end}  
                - total_spending, transaction_count  
                - average_transaction, median_transaction  
                - spending_days, daily_average  
                - max_transaction, min_transaction  

        2. **analyze_time_patterns(client_id: int)**  
            • Returns a dict with:  
                - weekend_vs_weekday (totals, averages, percentages, preference)  
                - daily_patterns (by_day, peak_day, lowest_day)  
                - hourly_patterns (by_hour, peak_hour, quiet_hour)  
                - monthly_patterns (by_month, peak_month, lowest_month)  
                - night_vs_day (totals, percentages, preference)  
                - behavioral_insights (is_weekend_spender, is_night_spender, peak_spending_time)  

        3. **get_spending_by_category(client_id: int, start_date: str, end_date: str, top_n: int = 10)**  
            • Returns a dict with:  
                - date_range: {start, end}  
                - category_breakdown: {mcc_category: {total_spent, transaction_count, average_spent, percentage}}  

        4. **get_spending_by_category_date(client_id: int, start_date: str, end_date: str)**  
            • Returns a dict with:  
                - date_series: { "YYYY-MM-DD": {mcc_category: amount, …}, … }  

        5. **get_spending_by_night(client_id: int, start_date: str, end_date: str, night_start: int = 22, night_end: int = 6)**  
            • Returns a dict with:  
                - client_id, date_range: {start, end}  
                - night_total, day_total  
                - night_count, day_count  
                - night_average, day_average  

        3. **Extract Parameters**  
        - `client_id` (integer)  
        - Dates in ISO `"YYYY-MM-DD"`  
        - Numeric settings: `top_n`, `night_start`, `night_end`

        4. **Determine Order**  
        - If multiple tools, list them in logical execution order.

        5. **Strict JSON Output**  
        Respond **only** with a JSON object matching this schema (no extra text):

        ```json
        {
        "tools_to_use": ["tool_name", …],
        "tool_parameters": {
            "tool_name": { "param": value, … },
            …
        },
        "execution_order": ["tool_name", …],
        "query_type": "<summary|time_series|category_breakdown|night_analysis|multi_tool>",
        "confidence": 0.##       // two‑decimal float
        }
             
        For Example
        User asks: “Show me my total and average spend between 2025-07-01 and 2025-07-15 for client 123.”
        You reply:
            {
            "tools_to_use": ["get_spending_summary"],
            "tool_parameters": {
                "get_spending_summary": {
                "client_id": 123,
                "start_date": "2025-07-01",
                "end_date": "2025-07-15"
                }
            },
            "execution_order": ["get_spending_summary"],
            "query_type": "summary",
            "confidence": 0.95
            }
            ```"""),
                ("human", "User Query: {query}")
        ])

        try:
            # 1) Invoke the classification prompt
            formatted = classification_prompt.format_messages(query=state['user_query'])
            llm_resp = self.llm.invoke(formatted)
            content = llm_resp.content.strip()

            # 2) Parse JSON
            payload = json.loads(content)
            tools_to_use      = payload["tools_to_use"]
            tool_parameters  = payload["tool_parameters"]
            execution_order  = payload["execution_order"]
            query_type       = payload["query_type"]
            confidence       = float(payload.get("confidence", 0))

            # 3) Populate state.intent
            state['intent'] = {
                "tools_to_use":     tools_to_use,
                "tool_parameters":  tool_parameters,
                "execution_order":  execution_order,
                "query_type":       query_type,
                "confidence":       confidence
            }
            state['execution_path'].append("intent_classifier")

            # 4) Build the tool_calls list
            tool_calls = []
            for name in execution_order:
                tool_calls.append({
                    "name": name,
                    "args": tool_parameters[name],
                    "id":   f"call_{name}_{len(state['messages'])}"
                })

            # 5) Add an AIMessage with those tool_calls
            state['messages'].append(AIMessage(
                content="Routing your query to the appropriate analysis tools now.",
                tool_calls=tool_calls
            ))

        except json.JSONDecodeError:
            state['error'] = (
                "Failed to parse JSON from the intent classifier. "
                "Please ensure the LLM response is valid JSON."
            )
            return state
        
    def _collect_tool_outputs(self, state: SpendingAgentState) -> SpendingAgentState:
        outputs = []
        for msg in state['messages']:
            if isinstance(msg, AIMessage) and msg.tool_calls is None:
                try:
                    obj = json.loads(msg.content)
                    outputs.append(obj)
                except:
                    pass
        state['analysis_result'] = outputs
        return state


      
    def _response_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate final, conversational reply from structured tool outputs."""
        # 1) Build a cleaner prompt
        response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly, expert financial assistant. 
        You’ve already run the following analyses (as JSON). 
        Now turn that into a clear, conversational answer:

        - Begin with a direct summary.
        - Highlight top insights (numbers, percentages).
        - Offer one actionable tip if relevant.
        - Keep it concise and easy to read.
        """),
                ("human", """
        Original Query: {query}

        Tool Results (JSON):
        {results}

        Please write the user‑facing reply now.
        """)
            ])

        try:
            # 2) Grab the structured results
            results = state.get('analysis_result') or []

            # 3) Serialize just once
            results_json = json.dumps(results, indent=2)

            
            response = self.llm.invoke(
                response_prompt.format_messages(
                    query=state['user_query'],
                    results=results_json
                )
            )

            # 5) Save and mark progression
            state['response'] = response.content
            state['execution_path'].append("response_generator")

        except Exception as e:
            state['error'] = f"Response generation error: {e}"
            state['response'] = (
                "Sorry, I ran into an issue putting together your answer. "
                "Could you try rephrasing?"
            )

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
    
    def process_query(self, client_id: int, user_query: str, config: Dict = None) -> Dict[str, Any]:
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
    

    def demo_spending_agent(self):
        """Demonstrate the Spending Agent capabilities with example calls"""
        # Update this path to point to your banking CSV file
        data_path = 'path/to/banking_data.csv'
        print(f"Loading data from: {data_path}")
        try:
            agent = SpendingAgent(data_path)
        except Exception as e:
            print(f"Error initializing agent: {e}")
            return

        # List available tools
        print("\nAvailable tools:")
        for tool in agent.get_available_tools():
            print(f"- {tool['name']}: {tool.get('description', 'No description')} (args: {tool.get('args_schema')})")

        # Example queries
        examples = [
            ('430', 'How much did I spend last month?'),
            ('1566', 'Show me my spending by category.'),
            ('430', 'What time of day do I spend the most?')
        ]

        for client_id, query in examples:
            print(f"\n=== Query: '{query}' for client {client_id} ===")
            try:
                result = agent.process_query(client_id=client_id, user_query=query)
                print("Response:")
                print(result.get('response'))
                print("Full result object:")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error processing query '{query}': {e}")

if __name__ == "__main__":

    SpendingAgent('C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv').demo_spending_agent()
    print("\n" + "="*60)


    
 



