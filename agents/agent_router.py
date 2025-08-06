import json
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
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
            # Use in-memory checkpointer for latest LangGraph
            self.memory = MemorySaver()
            print(f"💾 Memory: In-memory checkpointer enabled")
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

        # Entry point - using string
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
        """Enhanced conversation context and memory management with better follow-up handling"""

        try:
            print("🧠 [DEBUG] Managing conversation context...")

            # Basic query validation with better follow-up detection
            user_query = state["user_query"].strip()
            
            # Handle very short or invalid queries, but allow common follow-ups
            common_followups = ['ok', 'yes', 'no', 'show', 'tell', 'more', 'continue', 'please']
            query_words = user_query.lower().split()
            
            if len(user_query) <= 2 and user_query.lower() not in ['no', 'yes', 'ok']:
                print(f"[DEBUG] ⚠️ Invalid query detected: '{user_query}'")
                state["error"] = "Please provide a more complete question about your finances."
                return state
            elif len(query_words) <= 3 and all(word in common_followups for word in query_words):
                # This is a follow-up like "ok please show", "yes tell me", etc.
                print(f"[DEBUG] 🔄 Follow-up command detected: '{user_query}'")

            client_id = state["client_id"]
            session_id = state.get("session_id", f"session_{client_id}_{datetime.now().strftime('%Y%m%d')}")
            
            # Get or create conversation context
            if session_id in self.contexts:
                context = self.contexts[session_id]
                context.last_interaction = datetime.now()
                context.message_count += 1
                print(f"[DEBUG] 📝 Continuing conversation: message #{context.message_count}")
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
                print(f"[DEBUG] 🆕 New conversation started")

            # Enhanced topic detection with context awareness
            query_lower = user_query.lower()
            
            # For follow-up queries, enhance with context
            is_followup = False
            followup_patterns = ["ok", "please show", "show me", "tell me", "yes", "continue", "more details"]
            
            if any(pattern in query_lower for pattern in followup_patterns) and context.message_count > 1:
                is_followup = True
                print(f"[DEBUG] 🔄 Follow-up detected, will use context from previous interaction")
                
                # Add context-based intent to the query for better processing
                if "comparison" in context.recent_topics and any(word in query_lower for word in ["show", "tell", "please"]):
                    # User likely wants to see the comparison mentioned earlier
                    state["enhanced_query"] = f"Show me how my spending compares to similar customers (follow-up to previous conversation)"
                    print(f"[DEBUG] 💡 Enhanced query with context: comparison analysis requested")
                elif context.last_agent_used == "spending" and any(word in query_lower for word in ["show", "more"]):
                    state["enhanced_query"] = f"Show me more details about my spending breakdown (follow-up)"
                    print(f"[DEBUG] 💡 Enhanced query with context: spending details requested")
                elif context.last_agent_used == "budget" and any(word in query_lower for word in ["show", "tell"]):
                    state["enhanced_query"] = f"Show me my budget details (follow-up)"
                    print(f"[DEBUG] 💡 Enhanced query with context: budget details requested")
            
            # Detect topics from current query (enhanced or original)
            enhanced_query = state.get("enhanced_query", user_query)
            enhanced_lower = enhanced_query.lower()
            
            detected_topics = []
            
            if any(word in enhanced_lower for word in ["budget", "create", "set up", "overspend", "allocate"]):
                detected_topics.append("budget")
            if any(word in enhanced_lower for word in ["spend", "spent", "spending", "transaction", "category", "total", "where did i"]):
                detected_topics.append("spending")
            if any(word in enhanced_lower for word in ["save", "savings", "saved"]):
                detected_topics.append("savings")
            if any(word in enhanced_lower for word in ["compare", "comparison", "vs", "versus", "against", "average", "similar"]):
                detected_topics.append("comparison")
            
            # Add detected topics to recent topics
            for topic in detected_topics:
                if topic not in context.recent_topics:
                    context.recent_topics.append(topic)
                    print(f"[DEBUG] 🏷️ Added topic: {topic}")

            # Keep only recent topics (last 5)
            context.recent_topics = context.recent_topics[-5:]
            
            if is_followup:
                print(f"[DEBUG] 🔄 Follow-up context: {context.last_agent_used or 'previous'} agent, topics: {context.recent_topics}")

            state["conversation_context"] = context
            state["session_id"] = session_id
            state["execution_path"].append("context_manager")

            print(f"[DEBUG] Context: {context.message_count} messages, topics: {context.recent_topics}, last agent: {context.last_agent_used or 'None'}")

        except Exception as e:
            print(f"❌ Context management error: {e}")
            state["error"] = f"Context management failed: {e}"

        return state

    def _query_router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Intelligent query routing with enhanced follow-up handling"""

        try:
            print("🎯 [DEBUG] Routing query to appropriate agent...")

            context = state.get("conversation_context")
            enhanced_query = state.get("enhanced_query")  # May have context-enhanced query
            original_query = state["user_query"]
            
            # Use enhanced query for routing if available, original otherwise
            query_for_routing = enhanced_query if enhanced_query else original_query
            
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
- Follow-up questions about spending data

**BUDGET AGENT** - Best for:
- Budget management and planning
- "Create a budget for X"
- "How am I doing against my budget?"
- "Where am I overspending?"
- "Set up a budget"
- Budget performance and optimization

{recent_context}

ROUTING RULES:
1. If user asks about spending/transactions → SPENDING AGENT
2. If user asks about budgets/budget creation → BUDGET AGENT  
3. For follow-ups or vague queries → use CONTEXT (last agent or topic)
4. Simple responses like "ok please show", "yes", "tell me more" → use LAST AGENT + TOPIC CONTEXT

Query to analyze: "{query_for_routing}"
Original user input: "{original_query}"

Respond with just the agent name: "spending" or "budget"
"""
                ),
                (
                    "human",
                    "Route this query based on the context provided above."
                )
            ])

            try:
                # Try simple routing without structured parsing
                chain = routing_prompt | self.llm
                routing_response = chain.invoke({
                    "query_for_routing": query_for_routing,
                    "original_query": original_query
                })
                
                # Extract agent from response
                response_text = routing_response.content.lower().strip()
                
                if "spending" in response_text:
                    primary_agent = "spending"
                    reasoning = "Query contains spending-related content or context"
                elif "budget" in response_text:
                    primary_agent = "budget"
                    reasoning = "Query contains budget-related content or context"
                else:
                    # Enhanced fallback logic
                    primary_agent, reasoning = self._enhanced_fallback_routing(
                        original_query, query_for_routing, context
                    )

                state["routing_decision"] = {
                    "primary_agent": primary_agent,
                    "secondary_agent": None,
                    "query_type": "analysis",
                    "urgency": "medium",
                    "confidence": 0.8,
                    "reasoning": reasoning,
                    "routing_method": "context_aware",
                    "enhanced_query_used": enhanced_query is not None
                }
                
                print(f"[DEBUG] ✅ Routed to: {primary_agent} ({reasoning})")
                if enhanced_query:
                    print(f"[DEBUG] 💡 Used enhanced query for routing")

            except Exception as parse_error:
                print(f"[DEBUG] Context-aware routing failed, using enhanced fallback: {parse_error}")
                
                primary_agent, reasoning = self._enhanced_fallback_routing(
                    original_query, query_for_routing, context
                )

                state["routing_decision"] = {
                    "primary_agent": primary_agent,
                    "secondary_agent": None,
                    "query_type": "analysis", 
                    "urgency": "medium",
                    "confidence": 0.7,
                    "reasoning": reasoning,
                    "routing_method": "enhanced_fallback"
                }
                
                print(f"[DEBUG] ✅ Enhanced fallback routed to: {primary_agent} ({reasoning})")

            state["execution_path"].append("query_router")

        except Exception as e:
            print(f"❌ Query routing error: {e}")
            state["error"] = f"Query routing failed: {e}"

        return state
    
    def _enhanced_fallback_routing(self, original_query: str, enhanced_query: str, context) -> tuple:
        """Enhanced fallback routing with better context awareness"""
        
        query_to_analyze = enhanced_query if enhanced_query else original_query
        query_lower = query_to_analyze.lower()
        original_lower = original_query.lower()
        
        # Strong budget keywords
        budget_keywords = ["budget", "create", "set up", "overspend", "allocate", "limit", "spending plan"]
        # Strong spending keywords  
        spending_keywords = ["spend", "spent", "spending", "transaction", "category", "total", "average", "where did i", "how much", "breakdown", "compare", "comparison"]
        # Follow-up keywords that need context
        followup_keywords = ["please show", "show me", "yes", "continue", "more", "details", "tell me", "ok"]
        
        if any(word in query_lower for word in budget_keywords):
            return "budget", "Contains budget-related keywords"
        elif any(word in query_lower for word in spending_keywords):
            return "spending", "Contains spending-related keywords"
        elif any(word in original_lower for word in followup_keywords):
            # Enhanced follow-up handling
            if context and context.last_agent_used:
                if "comparison" in context.recent_topics or "compare" in query_lower:
                    return "spending", f"Follow-up comparison request, using spending agent"
                else:
                    return context.last_agent_used, f"Follow-up question, continuing with {context.last_agent_used} agent"
            else:
                return "spending", "Follow-up with no context, defaulting to spending"
        else:
            # Default based on conversation history
            if context and context.last_agent_used:
                return context.last_agent_used, "Using conversation context"
            else:
                return "spending", "Default routing to spending agent"

    def _spending_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute spending agent query with context-aware processing"""

        try:
            print("📊 [DEBUG] Executing spending agent query...")

            # Use enhanced query if available, otherwise use original
            query_to_process = state.get("enhanced_query", state["user_query"])
            
            if state.get("enhanced_query"):
                print(f"[DEBUG] 💡 Processing enhanced query: {query_to_process}")

            result = self.spending_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process  # Use context-enhanced query
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
        """Execute budget agent query with context-aware processing"""

        try:
            print("💰 [DEBUG] Executing budget agent query...")

            # Use enhanced query if available, otherwise use original
            query_to_process = state.get("enhanced_query", state["user_query"])
            
            if state.get("enhanced_query"):
                print(f"[DEBUG] 💡 Processing enhanced query: {query_to_process}")

            result = self.budget_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process  # Use context-enhanced query
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
        """Enhanced response synthesis with conversation continuity"""

        try:
            print("🔧 [DEBUG] Synthesizing final response...")

            primary_response = state.get("primary_response", "")
            context = state.get("conversation_context")
            routing = state.get("routing_decision", {})

            if not primary_response:
                state["final_response"] = "I apologize, but I couldn't generate a response to your query."
                return state

            # Enhanced conversational continuity
            prefix = ""
            if context and context.message_count > 1:
                # Different prefixes based on conversation state
                if context.message_count == 2:
                    if routing.get("routing_method") == "fallback" and context.last_agent_used:
                        prefix = ""  # No prefix for natural follow-up
                elif context.message_count > 2:
                    # Check if this is a follow-up that should be connected
                    query_lower = state["user_query"].lower()
                    if any(word in query_lower for word in ["please provide", "show me", "tell me more", "continue", "yes"]):
                        prefix = ""  # Natural continuation
                    elif context.last_agent_used == routing.get("primary_agent"):
                        prefix = ""  # Same agent, natural flow
                    else:
                        prefix = f"Switching to {routing.get('primary_agent')} analysis... "
            
            # Combine response
            state["final_response"] = prefix + primary_response

            # Enhanced cross-agent suggestions based on conversation flow
            agent_used = routing.get("primary_agent")
            recent_topics = context.recent_topics if context else []
            
            suggestion = ""
            if agent_used == "spending":
                if "budget" not in recent_topics and context and context.message_count >= 2:
                    suggestion = "\n\n💡 *Based on this spending analysis, would you like help creating or reviewing a budget?*"
                elif "comparison" not in recent_topics:
                    suggestion = "\n\n📊 *I can also show you how your spending compares to similar customers if you're interested.*"
            elif agent_used == "budget":
                if "spending" in recent_topics:
                    suggestion = "\n\n📈 *I can provide more detailed spending breakdowns to help fine-tune your budget.*"
                elif context and context.message_count <= 2:
                    suggestion = "\n\n🎯 *I can also analyze your historical spending patterns to suggest better budget allocations.*"

            state["final_response"] += suggestion

            state["execution_path"].append("response_synthesizer")
            print("✅ Response synthesis complete with conversation continuity")

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
    print("🔧 Type 'quit' to exit, 'test' for simple test")
    print("=" * 60)

    # Initialize router
    client_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv"
    overall_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/overall_data.csv"

    try:
        router = PersonalFinanceRouter(
            client_csv_path=client_csv,
            overall_csv_path=overall_csv,
            enable_memory=False  # Disable memory for now to avoid compatibility issues
        )

        client_id = 430
        session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        print(f"\n🎬 **SIMPLE TEST MODE**")
        print(f"Session ID: {session_id}")
        print("-" * 40)

        # Simple test queries first
        test_queries = [
            "How much did I spend last month?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n👤 **User**: {query}")
            
            result = router.chat(
                client_id=client_id,
                user_query=query,
                session_id=session_id
            )

            print(f"🤖 **Agent** ({result['agent_used'].upper()}): {result['response']}")
            print(f"⚙️  *Success: {result['success']}*")
            
            if result.get('error'):
                print(f"❌ *Error: {result['error']}*")
            else:
                print("✅ *Test passed!*")

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
                elif user_input.lower() == 'test':
                    # Quick test of individual agents
                    print("\n🧪 **Testing Individual Agents**")
                    
                    # Test spending agent directly
                    print("📊 Testing Spending Agent...")
                    try:
                        spending_result = router.spending_agent.process_query(
                            client_id=client_id,
                            user_query="How much did I spend last month?"
                        )
                        print(f"✅ Spending Agent: {spending_result['success']}")
                        if spending_result['success']:
                            print(f"📝 Response: {spending_result['response'][:100]}...")
                        else:
                            print(f"❌ Error: {spending_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"❌ Spending Agent failed: {e}")
                    
                    # Test budget agent directly  
                    print("\n💰 Testing Budget Agent...")
                    try:
                        budget_result = router.budget_agent.process_query(
                            client_id=client_id,
                            user_query="Create a $800 budget for groceries"
                        )
                        print(f"✅ Budget Agent: {budget_result['success']}")
                        if budget_result['success']:
                            print(f"📝 Response: {budget_result['response'][:100]}...")
                        else:
                            print(f"❌ Error: {budget_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"❌ Budget Agent failed: {e}")
                        
                    continue
                elif not user_input:
                    continue

                result = router.chat(
                    client_id=client_id,
                    user_query=user_input,
                    session_id=session_id
                )

                # Enhanced response display with full routing information
                agent_emoji = {
                    'spending': '📊',
                    'budget': '💰',
                    'error': '❌',
                    'unknown': '🤖'
                }.get(result['agent_used'], '🤖')
                
                agent_name = result['agent_used'].title() + " Agent" if result['agent_used'] not in ['error', 'unknown'] else 'System'
                
                print(f"\n{agent_emoji} **{agent_name}**: {result['response']}")
                
                # Show detailed execution path
                if result.get('execution_path'):
                    path_display = ' → '.join(result['execution_path'])
                    print(f"🛤️  *Execution Path: {path_display}*")
                
                # Show routing decision details
                print(f"🎯 *Routing: {result.get('message_count', 0)} messages in conversation*")
                
                # Handle errors gracefully
                if result.get('error'):
                    print(f"\n⚠️ *Technical note: {result['error']}*")
                    print("💡 Try rephrasing your question or ask for help with 'help'")

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize router: {e}")
        print("💡 Trying individual agent tests...")
        
        # Try testing agents individually
        try:
            from agents.spendings_agent import SpendingAgent
            print("🧪 Testing Spending Agent directly...")
            spending_agent = SpendingAgent(
                client_csv_path=client_csv,
                overall_csv_path=overall_csv,
                memory=False
            )
            result = spending_agent.process_query(430, "How much did I spend last month?")
            print(f"✅ Spending Agent works: {result['success']}")
            print(f"📝 Response: {result['response'][:100]}...")
            
        except Exception as e2:
            print(f"❌ Spending Agent also failed: {e2}")


if __name__ == "__main__":
    interactive_chat_demo()