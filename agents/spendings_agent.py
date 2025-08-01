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
    analysis_result: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]

class SpendingAgent:
    """LangGraph-based Spending Agent using @tool decorators"""
    
    def __init__(self, banking_data_path: str, model_name: str = "gpt-4o", memory: bool = True):
        # Initialize data store
        self.data_store = DataStore()
        print(f"The Banking Data path provided {banking_data_path}")
        self.data_store.load_data(banking_data_path)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # the tools we have defined under tools package
        self.tools = [
            get_spending_summary,
            get_spending_by_category,
            get_spending_by_category_date, 
            get_spending_by_night,
            analyze_time_patterns
            
        ]

        # Setup memory (optional)
        self.memory = SqliteSaver.from_conn_string(":memory:") if memory else None
        
        # Build the graph
        self.graph = self._build_graph()
        print(self.graph.get_graph().draw_mermaid())
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        workflow = StateGraph(SpendingAgentState)

        # the nodes in the graph
        workflow.add_node("intent_classifier",    self._intent_classifier_node)
        workflow.add_node("tool_executor",        self._custom_tool_executor)
        workflow.add_node("collector",            self._collect_tool_outputs)
        workflow.add_node("response_generator",   self._response_generator_node)
        workflow.add_node("error_handler",        self._error_handler_node)

        #the entry or start to our graph
        workflow.set_entry_point("intent_classifier")

        #After intent, branch on success or error
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {
                "execute_tools": "tool_executor",
                "error":          "error_handler"
            }
        )

        workflow.add_edge("tool_executor",      "collector")
        workflow.add_edge("collector",          "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("error_handler",     END)

        print("Let Print the Graph now")




        # Compile with optional checkpointing
        return workflow.compile(checkpointer=self.memory)
    def _intent_classifier_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Classify user intent and select appropriate tools via structured JSON from the LLM."""
        # 1) Build the classification prompt

        classification_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                "You are an AI assistant that helps classify user queries about spending analytics. "
                "Based on the user's query, select the appropriate tool(s) to use and provide the necessary parameters in a JSON format.\n\n"
                "Available tools:\n"
                "1. get_spending_summary(client_id: int, start_date: str, end_date: str)\n"
                "2. analyze_time_patterns(client_id: int)\n"
                "3. get_spending_by_category(client_id: int, start_date: str, end_date: str, top_n: int = 10)\n"
                "4. get_spending_by_category_date(client_id: int, start_date: str, end_date: str)\n"
                "5. get_spending_by_night(client_id: int, start_date: str, end_date: str, night_start: int = 22, night_end: int = 6)\n\n"
                "Always respond in this exact JSON structure:\n\n"
                "{{\n"
                "  \"tools_to_use\": [\"tool_name\"],\n"
                "  \"tool_parameters\": {{\n"
                "    \"tool_name\": {{\n"
                "      \"client_id\": 123,\n"
                "      \"start_date\": \"YYYY-MM-DD\",\n"
                "      \"end_date\": \"YYYY-MM-DD\"\n"
                "    }}\n"
                "  }},\n"
                "  \"execution_order\": [\"tool_name\"],\n"
                "  \"query_type\": \"summary|time_patterns|category_breakdown|category_date|night_analysis\"\n"
                "}}"
                "Use only valid JSON syntax. Do not explain. Do not include markdown or code blocks. "
                "Wrap everything inside a valid JSON object with double quotes."
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        messages = [
            HumanMessage(content=state["user_query"])
        ]

        prompt_inputs = classification_prompt.invoke({"messages": messages})




        try:
            # Debug: entering classifier
            print("⏺️ [DEBUG] Running intent classifier for:", state['user_query'])
            
            print("[DEBUG] About to call llm.invoke()")
            llm_resp = self.llm.invoke(prompt_inputs)
            print("[DEBUG] llm.invoke() succeeded")
            print("[DEBUG] LLM response:\n", llm_resp.content)
            
            #  Parse JSON payload
            payload = json.loads(llm_resp.content.strip())
            tool_params = payload["tool_parameters"]

            #  Normalize any "start"/"end" keys to "start_date"/"end_date"
            for name, args in tool_params.items():
                if "start" in args and "end" in args:
                    args["start_date"] = args.pop("start")
                    args["end_date"]   = args.pop("end")
            
            tools_to_use    = payload["tools_to_use"]
            execution_order = payload["execution_order"]
            query_type      = payload["query_type"]
            confidence      = float(payload.get("confidence", 0))

            # 3) Populate state.intent
            state['intent'] = {
                "tools_to_use":     tools_to_use,
                "tool_parameters":  tool_params,
                "execution_order":  execution_order,
                "query_type":       query_type,
                "confidence":       confidence
            }
            state['execution_path'].append("intent_classifier")

            # 4) Build tool_calls
            tool_calls = []
            for name in execution_order:
                tool_calls.append({
                    "name": name,
                    "args": tool_params[name],
                    "id":   f"call_{name}_{len(state['messages'])}"
                })

            # 5) Enqueue AIMessage that routes to tools
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
        except Exception as e:
            state['error'] = f"Intent classification error: {e}"
            return state

        return state
    

    def _custom_tool_executor(self, state: SpendingAgentState) -> SpendingAgentState:
        """Execute tools and store outputs as messages."""
        try:
            tool_outputs = []
            for tool_name in state['intent']['execution_order']:
                # Find the tool
                tool_fn = next(t for t in self.tools if t.name == tool_name)
                params = state['intent']['tool_parameters'][tool_name]
        
                result = tool_fn.invoke(params)
                print(f"These are the results {result}")
                
                tool_outputs.append(result)
                state['messages'].append(AIMessage(content=json.dumps(result),tool_calls=None))  # Store result
            state['execution_path'].append("tool_executor")
        except Exception as e:
            state['error'] = f"Tool execution failed: {e}"
        return state

    def _collect_tool_outputs(self, state: SpendingAgentState) -> SpendingAgentState:
        outputs = []
        print("≈ current messages:", state['messages'])
        for msg in state['messages']:
            if isinstance(msg, AIMessage):
                tc = getattr(msg, "tool_calls", None)
                # accept messages with no tool_calls or an empty list
                if tc is None or tc == []:
                    try:
                        outputs.append(json.loads(msg.content))
                    except json.JSONDecodeError:
                        pass
        state['analysis_result'] = outputs
        state['execution_path'].append("collector")
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
        # # Update this path to point to your banking CSV file
        # data_path = 'C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv'
        # print(f"Loading data from: {data_path}")
        # try:
        #     agent = SpendingAgent(data_path, memory=False)
        # except Exception as e:
        #     print(f"Error initializing agent: {e}")
        #     return

        # # List available tools
        # print("\nAvailable tools:")
        # for tool in agent.get_available_tools():
        #     print(f"- {tool['name']}: {tool.get('description', 'No description')} (args: {tool.get('args_schema')})")

        # # Example queries
        # examples = [
        #     ('430', 'How much did I spend last month?')
        # ]

        # for client_id, query in examples:
        #     print(f"\n=== Query: '{query}' for client {client_id} ===")
        #     try:
        #         result = agent.process_query(client_id=client_id, user_query=query)
        #         print("Response:")
        #         print(result.get('response'))
        #         print("Full result object:")
        #         print(json.dumps(result, indent=2))
        #     except Exception as e:
        #         print(f"Error processing query '{query}': {e}")
        # 1) Create your agent





if __name__ == "__main__":
    print("This is our 9010 try")
    # SpendingAgent('C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv').demo_spending_agent()
    agent = SpendingAgent("C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv", memory=False)

    # 2) Build a fake state with intent already populated
    state = {
    "client_id":       430,
    "user_query":      "ignored",
    "intent": {
        "execution_order":     ["get_spending_summary"],
        "tool_parameters": {
        "get_spending_summary": {
            "client_id": 430,
            "start_date": "2023-09-01",
            "end_date":   "2023-09-30"
        }
        }
    },
    "messages":        [],      # executor will append directly here
    "analysis_result": None,
    "response":        None,
    "error":           None,
    "execution_path":  []
    }

    # 3) Run only your executor + collector
    state = agent._custom_tool_executor(state)
    state = agent._collect_tool_outputs(state)

    # 4) Inspect
    print("Execution path:", state["execution_path"])
    print("Raw outputs:", state["analysis_result"])
    print("\n" + "="*60)


    
 



