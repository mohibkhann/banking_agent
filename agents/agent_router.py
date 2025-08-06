import json
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
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

# Import our agents
from agents.spendings_agent import SpendingAgent
from agents.budget_agent import BudgetAgent

load_dotenv()


# Pydantic models for routing
class AgentRouting(BaseModel):
    """Structured agent routing decision"""
    
    primary_agent: Literal["spending", "budget"] = Field(
        description="Primary agent to handle the query: 'spending' for spending analysis, 'budget' for budget management"
    )
    secondary_agent: Optional[Literal["spending", "budget"]] = Field(
        default=None,
        description="Secondary agent if cross-agent context is needed"
    )
    query_type: str = Field(
        description="Type of query: spending_analysis, budget_management, comparative, or hybrid"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Query urgency for prioritization"
    )
    confidence: float = Field(
        description="Routing confidence score between 0 and 1", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision"
    )


@dataclass
class ConversationContext:
    """Conversation context for memory management"""
    client_id: int
    session_start: datetime
    last_interaction: datetime
    message_count: int
    recent_topics: List[str]
    last_agent_used: Optional[str]
    conversation_summary: str
    key_insights: List[str]


class MultiAgentState(TypedDict):
    """State for multi-agent routing system"""
    
    client_id: int
    user_query: str
    conversation_context: Optional[ConversationContext]
    routing_decision: Optional[Dict[str, Any]]
    primary_response: Optional[str]
    secondary_response: Optional[str]
    final_response: str
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    session_id: str


class PersonalFinanceRouter:
    """
    Intelligent router that manages multiple financial agents with memory and context.
    Routes queries to appropriate agents and maintains conversation context.
    """

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
        enable_memory: bool = True,
        memory_db_path: str = "conversation_memory.db"
    ):
        print(f"🤖 Initializing PersonalFinanceRouter...")
        print(f"📊 Data sources: Client CSV, Overall CSV")
        print(f"🧠 Memory enabled: {enable_memory}")

        # Initialize agents
        print("🔄 Loading Spending Agent...")
        self.spending_agent = SpendingAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False  # We'll handle memory at router level
        )

        print("💰 Loading Budget Agent...")
        self.budget_agent = BudgetAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False  # We'll handle memory at router level
        )

        # Initialize LLM for routing
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Set up routing parser
        self.routing_parser = PydanticOutputParser(pydantic_object=AgentRouting)

        # Memory management
        self.enable_memory = enable_memory
        if enable_memory:
            self.memory = SqliteSaver.from_conn_string(memory_db_path)
            print(f"💾 Memory database: {memory_db_path}")
        else:
            self.memory = None

        # Conversation contexts (in-memory cache)
        self.contexts: Dict[str, ConversationContext] = {}

        # Build router graph
        self.graph = self._build_router_graph()
        
        print("✅ PersonalFinanceRouter initialized successfully!")

    def _build_router_graph(self) -> StateGraph:
        """Build the multi-agent routing workflow"""

        workflow = StateGraph(MultiAgentState)

        # Router workflow nodes
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("query_router", self._query_router_node)
        workflow.add_node("spending_agent_node", self._spending_agent_node)
        workflow.add_node("budget_agent_node", self._budget_agent_node)
        workflow.add_node("response_synthesizer", self._response_synthesizer_node)
        workflow.add_node("memory_updater", self._memory_updater_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point
        workflow.set_entry_point("context_manager")

        # Routing logic
        workflow.add_edge("context_manager", "query_router")
        workflow.add_conditional_edges(
            "query_router",
            self._route_to_agents,
            {
                "spending": "spending_agent_node",
                "budget": "budget_agent_node",
                "error": "error_handler"
            }
        )

        workflow.add_edge("spending_agent_node", "response_synthesizer")
        workflow.add_edge("budget_agent_node", "response_synthesizer")
        workflow.add_edge("response_synthesizer", "memory_updater")
        workflow.add_edge("memory_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)

    def _context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """Manage conversation context and memory"""

        try:
            print("🧠 [DEBUG] Managing conversation context...")

            client_id = state["client_id"]
            session_id = state.get("session_id", f"session_{client_id}_{datetime.now().strftime('%Y%m%d')}")
            
            # Get or create conversation context
            if session_id in self.contexts:
                context = self.contexts[session_id]
                context.last_interaction = datetime.now()
                context.message_count += 1
            else:
                context = ConversationContext(
                    client_id=client_id,
                    session_start=datetime.now(),
                    last_interaction=datetime.now(),
                    message_count=1,
                    recent_topics=[],
                    last_agent_used=None,
                    conversation_summary="New conversation started",
                    key_insights=[]
                )
                self.contexts[session_id] = context

            # Add current query topic
            query_lower = state["user_query"].lower()
            if "budget" in query_lower:
                if "budget" not in context.recent_topics:
                    context.recent_topics.append("budget")
            if any(word in query_lower for word in ["spend", "spent", "spending", "transaction"]):
                if "spending" not in context.recent_topics:
                    context.recent_topics.append("spending")

            # Keep only recent topics (last 5)
            context.recent_topics = context.recent_topics[-5:]

            state["conversation_context"] = context
            state["session_id"] = session_id
            state["execution_path"].append("context_manager")

            print(f"[DEBUG] Context updated: {context.message_count} messages, topics: {context.recent_topics}")

        except Exception as e:
            print(f"❌ Context management error: {e}")
            state["error"] = f"Context management failed: {e}"

        return state

    def _query_router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Intelligent query routing with context awareness"""

        try:
            print("🎯 [DEBUG] Routing query to appropriate agent...")

            context = state.get("conversation_context")
            recent_context = ""
            if context:
                recent_context = f"""
CONVERSATION CONTEXT:
- Message count: {context.message_count}
- Recent topics: {', '.join(context.recent_topics)}
- Last agent used: {context.last_agent_used or 'None'}
- Key insights: {', '.join(context.key_insights[-3:]) if context.key_insights else 'None'}
"""

            routing_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are an intelligent query router for a personal finance system.

You have access to two specialized agents:

**SPENDING AGENT** - Best for:
- Spending analysis and transaction queries
- "How much did I spend on X?"
- "Show me my spending by category"
- "What's my average transaction?"
- "Compare my spending to others"
- Historical spending analysis

**BUDGET AGENT** - Best for:
- Budget management and planning
- "Create a budget for X"
- "How am I doing against my budget?"
- "Where am I overspending?"
- "Set up a budget"
- Budget performance and optimization

{recent_context}

{self.routing_parser.get_format_instructions()}

Analyze the query and route intelligently:"""
                ),
                (
                    "human",
                    "Route this query: {user_query}"
                )
            ])

            try:
                chain = routing_prompt | self.llm | self.routing_parser
                routing_result = chain.invoke({"user_query": state["user_query"]})
                
                routing_dict = routing_result.model_dump()
                state["routing_decision"] = routing_dict
                
                print(f"[DEBUG] Routed to: {routing_dict['primary_agent']} (confidence: {routing_dict['confidence']:.2f})")
                print(f"[DEBUG] Reasoning: {routing_dict['reasoning']}")

            except Exception as parse_error:
                print(f"[DEBUG] Structured routing failed, using fallback: {parse_error}")
                
                # Fallback routing logic
                query_lower = state["user_query"].lower()
                
                budget_keywords = ["budget", "create", "set up", "overspend", "allocate", "limit"]
                spending_keywords = ["spend", "spent", "spending", "transaction", "category", "total", "average"]
                
                if any(word in query_lower for word in budget_keywords):
                    primary_agent = "budget"
                    reasoning = "Contains budget-related keywords"
                elif any(word in query_lower for word in spending_keywords):
                    primary_agent = "spending"
                    reasoning = "Contains spending-related keywords"
                else:
                    # Default based on context
                    if context and context.last_agent_used:
                        primary_agent = context.last_agent_used
                        reasoning = "Following conversation context"
                    else:
                        primary_agent = "spending"
                        reasoning = "Default to spending analysis"

                state["routing_decision"] = {
                    "primary_agent": primary_agent,
                    "secondary_agent": None,
                    "query_type": "analysis",
                    "urgency": "medium",
                    "confidence": 0.7,
                    "reasoning": reasoning
                }

            state["execution_path"].append("query_router")

        except Exception as e:
            print(f"❌ Query routing error: {e}")
            state["error"] = f"Query routing failed: {e}"

        return state

    def _spending_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute spending agent query"""

        try:
            print("📊 [DEBUG] Executing spending agent query...")

            result = self.spending_agent.process_query(
                client_id=state["client_id"],
                user_query=state["user_query"]
            )

            state["primary_response"] = result.get("response")
            
            # Update context
            if state.get("conversation_context"):
                state["conversation_context"].last_agent_used = "spending"

            state["execution_path"].append("spending_agent")
            print(f"✅ Spending agent completed: {result.get('success', False)}")

        except Exception as e:
            print(f"❌ Spending agent error: {e}")
            state["error"] = f"Spending agent failed: {e}"

        return state

    def _budget_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute budget agent query"""

        try:
            print("💰 [DEBUG] Executing budget agent query...")

            result = self.budget_agent.process_query(
                client_id=state["client_id"],
                user_query=state["user_query"]
            )

            state["primary_response"] = result.get("response")
            
            # Update context
            if state.get("conversation_context"):
                state["conversation_context"].last_agent_used = "budget"

            state["execution_path"].append("budget_agent")
            print(f"✅ Budget agent completed: {result.get('success', False)}")

        except Exception as e:
            print(f"❌ Budget agent error: {e}")
            state["error"] = f"Budget agent failed: {e}"

        return state

    def _response_synthesizer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Synthesize and enhance the final response"""

        try:
            print("🔧 [DEBUG] Synthesizing final response...")

            primary_response = state.get("primary_response", "")
            context = state.get("conversation_context")
            routing = state.get("routing_decision", {})

            # For now, use primary response directly but add context
            if context and context.message_count > 1:
                # Add conversational continuity for multi-turn conversations
                if context.message_count == 2:
                    prefix = "Thanks for the follow-up! "
                elif context.message_count > 5:
                    prefix = "Continuing our conversation... "
                else:
                    prefix = ""
                
                state["final_response"] = prefix + primary_response
            else:
                state["final_response"] = primary_response

            # Add helpful suggestions based on agent used
            agent_used = routing.get("primary_agent")
            if agent_used == "spending" and "budget" not in state["user_query"].lower():
                state["final_response"] += "\n\n💡 *Would you like help creating a budget based on this spending analysis?*"
            elif agent_used == "budget" and context and "spending" in context.recent_topics:
                state["final_response"] += "\n\n📊 *I can also provide detailed spending breakdowns if you'd like more analysis.*"

            state["execution_path"].append("response_synthesizer")
            print("✅ Response synthesis complete")

        except Exception as e:
            print(f"❌ Response synthesis error: {e}")
            state["final_response"] = state.get("primary_response", "I apologize, but I encountered an issue processing your request.")

        return state

    def _memory_updater_node(self, state: MultiAgentState) -> MultiAgentState:
        """Update conversation memory and context"""

        try:
            print("💾 [DEBUG] Updating conversation memory...")

            context = state.get("conversation_context")
            if context:
                # Update conversation summary
                if state.get("final_response"):
                    # Extract key insights from the response
                    response = state["final_response"]
                    if "$" in response and any(word in response.lower() for word in ["spent", "budget", "total"]):
                        # Extract monetary insights
                        import re
                        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', response)
                        if amounts:
                            insight = f"Recent discussion: {amounts[0]} mentioned"
                            if insight not in context.key_insights:
                                context.key_insights.append(insight)

                # Keep only recent insights
                context.key_insights = context.key_insights[-5:]

                # Update conversation summary
                query_type = state.get("routing_decision", {}).get("query_type", "general")
                context.conversation_summary = f"Recent {query_type} discussion, {context.message_count} messages"

            state["execution_path"].append("memory_updater")
            print("✅ Memory updated")

        except Exception as e:
            print(f"❌ Memory update error: {e}")

        return state

    def _error_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle routing and execution errors"""

        error_message = state.get("error", "Unknown routing error")
        print(f"🔧 [DEBUG] Handling router error: {error_message}")

        state["final_response"] = f"""I encountered an issue processing your request: {error_message}

🤖 **I can help you with:**
- 📊 **Spending Analysis**: "How much did I spend last month?"
- 💰 **Budget Management**: "Create a $500 budget for groceries"
- 📈 **Comparisons**: "How do I compare to others?"
- 🎯 **Budget Tracking**: "Am I over budget this month?"

Please try rephrasing your question, and I'll route it to the right specialist!"""

        state["execution_path"].append("error_handler")
        return state

    def _route_to_agents(self, state: MultiAgentState) -> str:
        """Determine which agent to route to"""

        if state.get("error"):
            return "error"

        routing_decision = state.get("routing_decision")
        if not routing_decision:
            state["error"] = "No routing decision made"
            return "error"

        primary_agent = routing_decision.get("primary_agent")
        if primary_agent == "spending":
            return "spending"
        elif primary_agent == "budget":
            return "budget"
        else:
            state["error"] = f"Unknown agent: {primary_agent}"
            return "error"

    def chat(
        self,
        client_id: int,
        user_query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main chat interface for user interactions"""

        if not session_id:
            session_id = f"session_{client_id}_{datetime.now().strftime('%Y%m%d_%H')}"

        initial_state = MultiAgentState(
            client_id=client_id,
            user_query=user_query,
            conversation_context=None,
            routing_decision=None,
            primary_response=None,
            secondary_response=None,
            final_response="",
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[],
            session_id=session_id
        )

        try:
            config = {"configurable": {"thread_id": session_id}} if self.memory else {}
            final_state = self.graph.invoke(initial_state, config=config)

            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": final_state.get("final_response", "No response generated"),
                "agent_used": final_state.get("routing_decision", {}).get("primary_agent", "unknown"),
                "execution_path": final_state.get("execution_path", []),
                "message_count": final_state.get("conversation_context", ConversationContext(0, datetime.now(), datetime.now(), 0, [], None, "", [])).message_count,
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"❌ Router execution error: {e}")
            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": "I encountered a system error. Please try again with a simpler question.",
                "agent_used": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
            }

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for a session"""
        
        if session_id in self.contexts:
            context = self.contexts[session_id]
            return {
                "session_id": session_id,
                "client_id": context.client_id,
                "session_start": context.session_start.isoformat(),
                "message_count": context.message_count,
                "recent_topics": context.recent_topics,
                "last_agent_used": context.last_agent_used,
                "key_insights": context.key_insights,
                "conversation_summary": context.conversation_summary
            }
        else:
            return {"error": "Session not found"}


def interactive_chat_demo():
    """Interactive chat demo as user 430"""
    
    print("🤖 PERSONAL FINANCE CHAT DEMO")
    print("=" * 60)
    print("💬 Chatting as User ID: 430")
    print("🔧 Type 'quit' to exit, 'summary' for conversation summary")
    print("=" * 60)

    # Initialize router
    client_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv"
    overall_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/overall_data.csv"

    try:
        router = PersonalFinanceRouter(
            client_csv_path=client_csv,
            overall_csv_path=overall_csv,
            enable_memory=True
        )

        client_id = 430
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Demo conversation flow
        demo_queries = [
            "How much did I spend last month?",
            "That seems like a lot, can you break it down by category?",
            "I want to create a budget to control my spending better",
            "Set up a $1000 budget for groceries",
            "How am I doing against my new budget this month?"
        ]

        print(f"\n🎬 **DEMO CONVERSATION FLOW**")
        print(f"Session ID: {session_id}")
        print("-" * 40)

        for i, query in enumerate(demo_queries, 1):
            print(f"\n👤 **User**: {query}")
            
            result = router.chat(
                client_id=client_id,
                user_query=query,
                session_id=session_id
            )

            print(f"🤖 **Agent** ({result['agent_used'].upper()}): {result['response']}")
            print(f"⚙️  *Execution: {' → '.join(result['execution_path'])}*")
            
            if result.get('error'):
                print(f"❌ *Error: {result['error']}*")

            # Small delay for demo effect
            import time
            time.sleep(1)

        print(f"\n📊 **CONVERSATION SUMMARY**")
        print("-" * 40)
        summary = router.get_conversation_summary(session_id)
        print(f"• Messages: {summary.get('message_count', 0)}")
        print(f"• Topics: {', '.join(summary.get('recent_topics', []))}")
        print(f"• Key Insights: {summary.get('key_insights', [])}")

        print(f"\n🎯 **INTERACTIVE MODE**")
        print("Now you can continue the conversation...")
        print("-" * 40)

        # Interactive mode
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for using Personal Finance Assistant!")
                    break
                elif user_input.lower() == 'summary':
                    summary = router.get_conversation_summary(session_id)
                    print(f"\n📊 **Session Summary:**")
                    for key, value in summary.items():
                        if key != 'error':
                            print(f"  • {key}: {value}")
                    continue
                elif not user_input:
                    continue

                result = router.chat(
                    client_id=client_id,
                    user_query=user_input,
                    session_id=session_id
                )

                print(f"\n🤖 **{result['agent_used'].upper()} Agent**: {result['response']}")
                
                if result.get('error'):
                    print(f"❌ *Error: {result['error']}*")

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize router: {e}")


if __name__ == "__main__":
    interactive_chat_demo()