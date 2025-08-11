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

# Pydantic models for routing with better domain detection
class AgentRouting(BaseModel):
    """Structured agent routing decision with domain relevance"""
    
    is_relevant: bool = Field(
        description="True if query is related to personal finance, spending, budgeting, or banking; False otherwise"
    )
    primary_agent: Literal["spending", "budget", "irrelevant"] = Field(
        description="Primary agent: 'spending' for transaction analysis, 'budget' for budget management, 'irrelevant' for off-topic"
    )
    query_category: str = Field(
        description="Category: finance_spending, finance_budget, finance_general, greeting, off_topic, unclear"
    )
    confidence: float = Field(
        description="Routing confidence score between 0 and 1", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision"
    )
    suggested_response_tone: Optional[str] = Field(
        default=None,
        description="For irrelevant queries: friendly_redirect, polite_decline, or clarification_needed"
    )


@dataclass
class ConversationContext:
    client_id: int
    session_start: datetime
    last_interaction: datetime
    message_count: int
    recent_topics: List[str]
    last_agent_used: Optional[str]
    conversation_summary: str
    key_insights: List[str]
    recent_conversations: List[Dict[str, Any]]  
    older_summary: str  


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
    enhanced_query: Optional[str]  # Add this field


class PersonalFinanceRouter:
    """
    Intelligent router that manages multiple financial agents with memory and context.
    Routes queries to appropriate agents and handles irrelevant queries gracefully.
    """

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
        enable_memory: bool = True,
        memory_db_path: str = "conversation_memory.db"
    ):
        print(f"Initializing PersonalFinanceRouter...")
        print(f"Data sources: Client CSV, Overall CSV")
        print(f"Memory enabled: {enable_memory}")

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
        workflow.add_node("irrelevant_handler", self._irrelevant_handler_node)
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
                "irrelevant": "irrelevant_handler",
                "error": "error_handler"
            }
        )

        workflow.add_edge("spending_agent_node", "response_synthesizer")
        workflow.add_edge("budget_agent_node", "response_synthesizer")
        workflow.add_edge("irrelevant_handler", "response_synthesizer")
        workflow.add_edge("response_synthesizer", "memory_updater")
        workflow.add_edge("memory_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)


    def _context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced conversation context and memory management"""

        try:
            print("🧠 [DEBUG] Managing conversation context...")

            # Basic query validation
            user_query = state["user_query"].strip()
            
            if len(user_query) <= 2 and user_query.lower() not in ['no', 'yes', 'ok']:
                print(f"[DEBUG] ⚠️ Invalid query detected: '{user_query}'")
                state["error"] = "Please provide a more complete question about your finances."
                return state

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
                    key_insights=[],
                    recent_conversations=[],  # NEW: Empty list for recent conversations
                    older_summary=""  # NEW: Empty summary for older conversations
                )
                self.contexts[session_id] = context
                print(f"[DEBUG] 🆕 New conversation started")

            # Enhanced follow-up detection using recent conversations
            query_lower = user_query.lower()
            state["enhanced_query"] = None
            
            # Check for follow-up patterns
            followup_patterns = ["show me", "tell me more", "what about", "more details", "continue"]
            is_followup = any(pattern in query_lower for pattern in followup_patterns)
            
            if is_followup and context.recent_conversations:
                print(f"[DEBUG] 🔄 Follow-up detected, using recent conversation context")
                
                # Get the most recent conversation
                last_conversation = context.recent_conversations[-1]
                last_query = last_conversation.get("user_query", "").lower()
                last_response = last_conversation.get("agent_response", "").lower()
                
                # Enhance query based on last conversation
                if "spending" in last_query or "spent" in last_response:
                    if any(word in query_lower for word in ["show", "more", "details"]):
                        state["enhanced_query"] = f"Show me more details about my spending patterns (follow-up to: {last_conversation['user_query']})"
                        print(f"[DEBUG] 💡 Enhanced query for spending follow-up")
                
                elif "budget" in last_query or "budget" in last_response:
                    if any(word in query_lower for word in ["show", "tell", "more"]):
                        state["enhanced_query"] = f"Show me my budget details (follow-up to: {last_conversation['user_query']})"
                        print(f"[DEBUG] 💡 Enhanced query for budget follow-up")
                
                elif "compare" in last_query or "comparison" in last_response:
                    if any(word in query_lower for word in ["show", "tell"]):
                        state["enhanced_query"] = f"Show me spending comparisons (follow-up to previous conversation)"
                        print(f"[DEBUG] 💡 Enhanced query for comparison follow-up")

            # Detect topics from current query
            enhanced_query = state.get("enhanced_query")
            query_to_analyze = enhanced_query if enhanced_query else user_query
            analyzed_lower = query_to_analyze.lower()
            
            detected_topics = []
            
            if any(word in analyzed_lower for word in ["budget", "create", "set up", "overspend", "allocate"]):
                detected_topics.append("budget")
            if any(word in analyzed_lower for word in ["spend", "spent", "spending", "transaction", "category", "total"]):
                detected_topics.append("spending")
            if any(word in analyzed_lower for word in ["compare", "comparison", "vs", "versus", "average", "similar"]):
                detected_topics.append("comparison")
            if any(word in analyzed_lower for word in ["save", "savings", "saved"]):
                detected_topics.append("savings")
            
            # Add detected topics to recent topics
            for topic in detected_topics:
                if topic not in context.recent_topics:
                    context.recent_topics.append(topic)
                    print(f"[DEBUG] 🏷️ Added topic: {topic}")

            # Keep only recent topics (last 5)
            context.recent_topics = context.recent_topics[-5:]

            state["conversation_context"] = context
            state["session_id"] = session_id
            state["execution_path"].append("context_manager")

            print(f"[DEBUG] Context: {context.message_count} messages, topics: {context.recent_topics}, last agent: {context.last_agent_used or 'None'}")

        except Exception as e:
            print(f"❌ Context management error: {e}")
            state["error"] = f"Context management failed: {e}"

        return state


    def _query_router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Intelligent query routing with improved domain detection"""

        try:
            print("🎯 [DEBUG] Routing query to appropriate agent...")

            context = state.get("conversation_context")
            enhanced_query = state.get("enhanced_query")
            original_query = state["user_query"]
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

            # First try structured parsing with Pydantic
            try:
                routing_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        f"""You are an intelligent query router for a personal finance assistant system.

                Your job is to determine if queries are relevant to personal finance/banking and route them appropriately.

                DOMAIN SCOPE:
                Our system handles:
                - Personal spending analysis and transactions
                - Budget creation, management, and tracking  
                - Financial comparisons and benchmarks
                - Banking transactions and account information
                - Savings goals and financial planning
                - Spending patterns and insights

                ROUTING RULES:

                1. **RELEVANT FINANCE QUERIES** (is_relevant: true):
                - Spending/transactions → primary_agent: "spending"
                - Budget management → primary_agent: "budget"
                - General finance questions → choose most appropriate agent
                - **Questions about capabilities in finance context** → primary_agent: "spending"

                2. **EXAMPLES OF RELEVANT QUERIES:**
                - "How much did I spend last month?" → spending
                - "Create a budget for groceries" → budget
                - "What can you help me with?" → spending (finance capabilities)
                - "What else can you do?" → spending (finance capabilities)
                - "How can you assist me?" → spending (finance capabilities)
                - "What services do you provide?" → spending (finance capabilities)

                3. **IRRELEVANT QUERIES** (is_relevant: false):
                Examples of irrelevant:
                - General knowledge ("What is the capital of France?")
                - Weather, news, sports
                - Programming/coding help (unless finance-related)
                - Health, recipes, travel planning
                - Entertainment recommendations
                - Academic subjects unrelated to finance

                For these → primary_agent: "irrelevant"

                4. **AMBIGUOUS QUERIES**:
                - If unclear but possibly finance-related → mark as relevant and route to spending agent
                - If clearly off-topic → mark as irrelevant

                {recent_context}

                Provide a JSON response with these fields:
                - is_relevant: boolean (true if finance-related, false otherwise)
                - primary_agent: string ("spending", "budget", or "irrelevant")
                - query_category: string (describe the type of query)
                - confidence: number between 0 and 1
                - reasoning: string (brief explanation)
                - suggested_response_tone: string or null (for irrelevant queries: "friendly_redirect", "polite_decline", or null)

                Analyze this query:
                Query: "{query_for_routing}"
                Original input: "{original_query}"
                """
                    ),
                    (
                        "human",
                        "Route this query and respond with a JSON object containing the routing decision."
                    )
                ])

                response = self.llm.invoke(
                    routing_prompt.format_messages(
                        query_for_routing=query_for_routing,
                        original_query=original_query,
                        recent_context=recent_context
                    )
                )
                
                # Parse JSON response
                import json
                response_text = response.content.strip()
                
                # Try to extract JSON from the response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "{" in response_text and "}" in response_text:
                    # Find the JSON object in the response
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    response_text = response_text[json_start:json_end]
                
                try:
                    routing_dict = json.loads(response_text)
                    
                    # Validate required fields
                    required_fields = ["is_relevant", "primary_agent", "confidence", "reasoning"]
                    for field in required_fields:
                        if field not in routing_dict:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Set defaults for optional fields
                    routing_dict.setdefault("query_category", "unknown")
                    routing_dict.setdefault("suggested_response_tone", None)
                    
                    print(f"[DEBUG] ✅ Query relevance: {routing_dict['is_relevant']}")
                    print(f"[DEBUG] ✅ Routed to: {routing_dict['primary_agent']} ({routing_dict['reasoning']})")
                    print(f"[DEBUG] ✅ Confidence: {routing_dict['confidence']:.2f}")
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[DEBUG] JSON parsing failed: {e}, using keyword-based routing")
                    raise e
                
                state["routing_decision"] = routing_dict
                
                if enhanced_query:
                    print(f"[DEBUG] 💡 Used enhanced query for routing")

            except Exception as parse_error:
                print(f"[DEBUG] Structured routing failed ({parse_error}), using simplified fallback")
                
                # Simplified fallback without structured parsing
                fallback_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a router for a personal finance assistant.

    Determine if this query is about:
    1. Personal spending/transactions → respond: "spending"
    2. Budget management → respond: "budget"  
    3. Neither (off-topic) → respond: "irrelevant"

    Respond with ONLY ONE WORD: spending, budget, or irrelevant"""),
                    ("human", "Query: {query}")
                ])
                
                fallback_response = self.llm.invoke(
                    fallback_prompt.format_messages(query=query_for_routing)
                )
                
                response_text = fallback_response.content.lower().strip()
                
                # Determine agent from response
                if "spending" in response_text:
                    primary_agent = "spending"
                    is_relevant = True
                elif "budget" in response_text:
                    primary_agent = "budget"
                    is_relevant = True
                else:
                    primary_agent = "irrelevant"
                    is_relevant = False
                
                state["routing_decision"] = {
                    "is_relevant": is_relevant,
                    "primary_agent": primary_agent,
                    "query_category": "unknown",
                    "confidence": 0.7,
                    "reasoning": "Fallback routing based on keyword detection",
                    "suggested_response_tone": "friendly_redirect" if not is_relevant else None
                }
                
                print(f"[DEBUG] ✅ Fallback routed to: {primary_agent}")

            state["execution_path"].append("query_router")

        except Exception as e:
            print(f"❌ Query routing error: {e}")
            state["error"] = f"Query routing failed: {e}"
            state["routing_decision"] = None  # Ensure it's set even on error

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
        """Execute spending agent query with conversation context"""

        try:
            print("📊 [DEBUG] Executing spending agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process")
                state["error"] = "No valid query provided to spending agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your spending again."
                return state
            
            # Always pass conversation context
            context = state.get("conversation_context")
            
            if enhanced_query:
                print(f"[DEBUG] 💡 Processing enhanced query with context")
            else:
                print(f"[DEBUG] 📝 Processing original query with context")

            # Always pass context to spending agent
            result = self.spending_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process,
                conversation_context=context  # Always pass context
            )

            state["primary_response"] = result.get("response")
            
            # Update context
            if context:
                context.last_agent_used = "spending"

            state["execution_path"].append("spending_agent")
            print(f"✅ Spending agent completed with context: {result.get('success', False)}")

        except Exception as e:
            print(f"❌ Spending agent error: {e}")
            state["error"] = f"Spending agent failed: {e}"
            state["primary_response"] = None

        return state

    def _budget_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute budget agent query with conversation context"""

        try:
            print("💰 [DEBUG] Executing budget agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process")
                state["error"] = "No valid query provided to budget agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your budget again."
                return state
            
            # Always pass conversation context
            context = state.get("conversation_context")
            
            if enhanced_query:
                print(f"[DEBUG] 💡 Processing enhanced query with context")
            else:
                print(f"[DEBUG] 📝 Processing original query with context")

            # Always pass context to budget agent
            result = self.budget_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process,
                conversation_context=context  # Always pass context
            )

            state["primary_response"] = result.get("response")
            
            # Update context
            if context:
                context.last_agent_used = "budget"

            state["execution_path"].append("budget_agent")
            print(f"✅ Budget agent completed with context: {result.get('success', False)}")

        except Exception as e:
            print(f"❌ Budget agent error: {e}")
            state["error"] = f"Budget agent failed: {e}"
            state["primary_response"] = None

        return state


    def _response_synthesizer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced response synthesis with better handling of irrelevant queries"""

        try:
            print("🔧 [DEBUG] Synthesizing final response...")

            primary_response = state.get("primary_response", "")
            context = state.get("conversation_context")
            routing = state.get("routing_decision", {})

            if not primary_response:
                # Check if we have an error that needs to be communicated
                if state.get("error"):
                    state["final_response"] = f"I encountered an issue: {state['error']}\n\nPlease try rephrasing your question about spending or budgeting."
                else:
                    state["final_response"] = "I apologize, but I couldn't generate a response to your query. Please try asking about your spending patterns or budget management."
                return state

            # Handle case when routing_decision is None or empty
            if not routing:
                state["final_response"] = primary_response
                state["execution_path"].append("response_synthesizer")
                return state

            # Different handling for irrelevant queries
            if routing.get("primary_agent") == "irrelevant":
                # For irrelevant queries, use the response as-is without adding finance suggestions
                state["final_response"] = primary_response
            else:
                # For relevant queries, add contextual suggestions
                prefix = ""
                if context and context.message_count > 1:
                    # Add context-aware prefixes for conversation continuity
                    if context.last_agent_used and context.last_agent_used != routing.get("primary_agent"):
                        prefix = f"Switching to {routing.get('primary_agent')} analysis... "
                
                state["final_response"] = prefix + primary_response
                
                # Add cross-agent suggestions for relevant queries
                agent_used = routing.get("primary_agent")
                recent_topics = context.recent_topics if context else []
                
                suggestion = ""
                if agent_used == "spending" and "budget" not in recent_topics:
                    suggestion = "\n\n💡 *Based on this spending analysis, would you like help creating or reviewing a budget?*"
                elif agent_used == "budget" and context and context.message_count <= 2:
                    suggestion = "\n\n🎯 *I can also analyze your historical spending patterns to suggest better budget allocations.*"
                
                state["final_response"] += suggestion

            state["execution_path"].append("response_synthesizer")
            print("✅ Response synthesis complete")

        except Exception as e:
            print(f"❌ Response synthesis error: {e}")
            state["final_response"] = state.get("primary_response", "I apologize, but I encountered an issue processing your request. Please try asking about your spending or budget.")

        return state
    


    def _create_conversation_summary(self, conversation: Dict[str, Any]) -> str:
        """Create a concise summary of a conversation"""
        
        query = conversation.get("user_query", "")
        response = conversation.get("agent_response", "")
        agent = conversation.get("agent_used", "")
        topics = conversation.get("topics", [])
        
        # Extract key information
        summary_parts = []
        
        # Add main topic
        if topics:
            summary_parts.append(f"{topics[0]}")
        
        # Add monetary amounts if present
        import re
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', response)
        if amounts:
            summary_parts.append(f"{amounts[0]}")
        
        # Add action if identifiable
        if "create" in query.lower() or "budget" in query.lower():
            summary_parts.append("budget creation")
        elif "compare" in query.lower():
            summary_parts.append("comparison")
        elif "spend" in query.lower():
            summary_parts.append("spending analysis")
        
        return f"{agent}: {', '.join(summary_parts)}" if summary_parts else f"{agent}: general query"

    def _extract_insights(self, context: ConversationContext, response: str) -> None:
        """Extract key insights from response"""
        
        # Extract monetary insights
        import re
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', response)
        if amounts:
            insight = f"Discussed amount: {amounts[0]}"
            if insight not in context.key_insights:
                context.key_insights.append(insight)
        
        # Extract category insights
        categories = ["groceries", "restaurants", "clothing", "transportation", "entertainment", "budget"]
        for category in categories:
            if category in response.lower():
                insight = f"Discussed {category}"
                if insight not in context.key_insights:
                    context.key_insights.append(insight)
                    break
        
        # Keep only last 5 insights
        context.key_insights = context.key_insights[-5:]

    def _update_conversation_summary(self, context: ConversationContext) -> None:
        """Update overall conversation summary"""
        
        if context.message_count <= 3:
            # Early conversation
            topics = ", ".join(context.recent_topics) if context.recent_topics else "general finance"
            context.conversation_summary = f"Early conversation about {topics}"
        else:
            # Ongoing conversation
            recent_topics = ", ".join(context.recent_topics[-3:]) if context.recent_topics else "various topics"
            agents_used = set()
            for conv in context.recent_conversations:
                if conv.get("agent_used"):
                    agents_used.add(conv["agent_used"])
            
            context.conversation_summary = f"Ongoing discussion: {recent_topics}. Agents: {', '.join(agents_used)}"

    # Enhanced get_conversation_summary method

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for a session"""
        
        if session_id in self.contexts:
            context = self.contexts[session_id]
            
            # Format recent conversations for display
            recent_display = []
            for conv in context.recent_conversations:
                recent_display.append({
                    "query": conv["user_query"][:100] + "..." if len(conv["user_query"]) > 100 else conv["user_query"],
                    "response_preview": conv["agent_response"][:100] + "..." if len(conv["agent_response"]) > 100 else conv["agent_response"],
                    "agent": conv["agent_used"],
                    "timestamp": conv["timestamp"]
                })
            
            return {
                "session_id": session_id,
                "client_id": context.client_id,
                "session_start": context.session_start.isoformat(),
                "message_count": context.message_count,
                "recent_topics": context.recent_topics,
                "last_agent_used": context.last_agent_used,
                "key_insights": context.key_insights,
                "conversation_summary": context.conversation_summary,
                "recent_conversations": recent_display,
                "older_summary": context.older_summary,
                "total_conversations": len(context.recent_conversations)
            }
        else:
            return {"error": "Session not found"}

    # Method to get context for LLM prompts (when needed)

    def get_conversation_context_for_llm(self, session_id: str) -> str:
        """Get conversation context for LLM prompts (when follow-up enhancement is needed)"""
        
        if session_id not in self.contexts:
            return ""
        
        context = self.contexts[session_id]
        
        if not context.recent_conversations:
            return ""
        
        # Build context string from recent conversations
        context_parts = []
        
        # Add recent topics
        if context.recent_topics:
            context_parts.append(f"Recent topics: {', '.join(context.recent_topics[-3:])}")
        
        # Add last conversation for context
        if context.recent_conversations:
            last_conv = context.recent_conversations[-1]
            context_parts.append(f"Last Q: {last_conv['user_query'][:80]}...")
            context_parts.append(f"Last A: {last_conv['agent_response'][:80]}...")
        
        # Add older summary if exists
        if context.older_summary:
            context_parts.append(f"Earlier: {context.older_summary}")
        
        return " | ".join(context_parts)


    def _memory_updater_node(self, state: MultiAgentState) -> MultiAgentState:
        """Update conversation memory: Keep last 3 conversations + summarize older ones"""

        try:
            print("💾 [DEBUG] Updating conversation memory...")

            context = state.get("conversation_context")
            if context:
                # Create current conversation record
                current_conversation = {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": context.recent_conversations[-1]["user_query"] if context.recent_conversations else state.get("user_query", ""),
                    "agent_response": state.get("final_response", ""),
                    "agent_used": state.get("routing_decision", {}).get("primary_agent"),
                    "topics": context.recent_topics.copy(),
                    "success": state.get("error") is None
                }
                
                # Add to recent conversations
                context.recent_conversations.append(current_conversation)
                
                # If we now have more than 3 conversations, move the oldest to summary
                if len(context.recent_conversations) > 3:
                    # Take the oldest conversation for summarization
                    oldest_conversation = context.recent_conversations.pop(0)
                    
                    # Add to older summary
                    summary_entry = self._create_conversation_summary(oldest_conversation)
                    
                    if context.older_summary:
                        context.older_summary += f" | {summary_entry}"
                    else:
                        context.older_summary = summary_entry
                    
                    # Keep summary concise (last 200 chars to avoid token bloat)
                    if len(context.older_summary) > 200:
                        context.older_summary = "..." + context.older_summary[-197:]
                    
                    print(f"[DEBUG] 💾 Moved oldest conversation to summary")
                
                # Update agent tracking
                routing_decision = state.get("routing_decision", {})
                if routing_decision.get("primary_agent"):
                    context.last_agent_used = routing_decision["primary_agent"]
                
                # Extract key insights from current response
                final_response = state.get("final_response", "")
                if final_response:
                    self._extract_insights(context, final_response)
                
                # Update conversation summary
                self._update_conversation_summary(context)
                
                print(f"[DEBUG] 💾 Memory updated: {len(context.recent_conversations)} recent conversations")
                if context.older_summary:
                    print(f"[DEBUG] 📚 Older summary: {context.older_summary[:100]}...")

            state["execution_path"].append("memory_updater")
            print("✅ Memory updated with conversation history")

        except Exception as e:
            print(f"❌ Memory update error: {e}")

        return state

    def _irrelevant_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle irrelevant queries with natural, helpful responses"""
        
        try:
            print("🚫 [DEBUG] Handling irrelevant query...")
            
            user_query = state["user_query"]
            context = state.get("conversation_context")
            
            # Generate natural, helpful response
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly personal banking assistant. The user has asked a question that's outside your expertise area of personal finance, spending, and budgeting.

    Your response should be:
    1. Warm and understanding
    2. Briefly acknowledge their question
    3. Naturally redirect to what you can help with
    4. Offer specific examples relevant to banking/finance

    Keep it conversational - like a bank employee politely redirecting the conversation.

    AVOID:
    - Technical explanations about your limitations
    - Formal language like "I'm designed to" or "My domain is"
    - Long lists of capabilities
    - Apologetic tone

    BE NATURAL AND HELPFUL:
    - "That's outside my area, but I'm here to help with your finances!"
    - "I focus on helping with your money matters"
    - Give 2-3 specific examples of what you can do
    """),
                ("human", """The user asked: "{query}"

    Provide a brief, natural response that redirects them to finance topics.""")
            ])
            
            response = self.llm.invoke(
                response_prompt.format_messages(query=user_query)
            )
            
            base_response = response.content
            
            suggestions = ""
            if context and context.recent_topics:
                if "spending" in context.recent_topics:
                    suggestions = "\n\nSince we were talking about your spending, would you like to dive deeper into any particular category or time period?"
                elif "budget" in context.recent_topics:
                    suggestions = "\n\nWe were discussing budgets - would you like help setting up budgets for specific categories?"
                elif "comparison" in context.recent_topics:
                    suggestions = "\n\nI can show you how your spending compares to similar customers in different areas like dining, shopping, or entertainment."
            else:
                # Default suggestions for new conversations
                suggestions = "\n\nI can help you understand where your money goes each month, set up budgets, or see how your spending compares to others. What interests you most?"
            
            state["primary_response"] = base_response + suggestions
            state["execution_path"].append("irrelevant_handler")
            
            print(f"✅ Handled irrelevant query naturally")
            
        except Exception as e:
            print(f"❌ Irrelevant handler error: {e}")
            # Simple fallback response
            state["primary_response"] = f"That's outside my area of expertise, but I'm here to help you with your finances! I can analyze your spending patterns, help you create budgets, or show you how you compare to similar customers. What would you like to explore?"
            
        return state

    def _error_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle routing and execution errors with natural responses"""

        error_message = state.get("error", "Unknown routing error")
        print(f"🔧 [DEBUG] Handling router error: {error_message}")
        
        user_query = state["user_query"]

        # Provide natural error responses without technical details
        if "Invalid query" in error_message:
            state["final_response"] = "I'd be happy to help! Could you tell me a bit more about what you'd like to know about your finances? I can help with spending analysis, budgeting, or comparing your habits to others."
        
        elif "Intent classification" in error_message or "routing" in error_message.lower():
            state["final_response"] = "I want to make sure I understand what you're looking for. Are you interested in seeing your spending patterns, setting up a budget, or something else financial? Just let me know!"
            
        elif "No valid query" in error_message:
            state["final_response"] = "I'm here to help with your financial questions! You can ask me about your spending, budgets, or how you compare to other customers. What would you like to know?"
            
        else:
            # Generic friendly error
            state["final_response"] = f"I ran into a small hiccup while processing your request. No worries though! Try asking me about your spending patterns, budget management, or financial comparisons. I'm here to help!"

        state["execution_path"].append("error_handler")
        return state

    def _route_to_agents(self, state: MultiAgentState) -> str:
        """Determine which agent/handler to route to"""

        if state.get("error"):
            return "error"

        routing_decision = state.get("routing_decision")
        if not routing_decision:
            state["error"] = "No routing decision made"
            return "error"

        primary_agent = routing_decision.get("primary_agent")
        
        # Route based on agent decision
        if primary_agent == "spending":
            return "spending"
        elif primary_agent == "budget":
            return "budget"
        elif primary_agent == "irrelevant":
            return "irrelevant"
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
            session_id=session_id,
            enhanced_query=None  # Initialize this field
        )

        try:
            config = {"configurable": {"thread_id": session_id}} if self.memory else {}
            final_state = self.graph.invoke(initial_state, config=config)

            # Handle case where routing_decision might be None
            agent_used = "unknown"
            if final_state.get("routing_decision"):
                agent_used = final_state["routing_decision"].get("primary_agent", "unknown")

            # Handle case where conversation_context might be None
            message_count = 0
            if final_state.get("conversation_context"):
                message_count = final_state["conversation_context"].message_count
            else:
                # Create a default context if it doesn't exist
                default_context = ConversationContext(
                    client_id=client_id,
                    session_start=datetime.now(),
                    last_interaction=datetime.now(),
                    message_count=1,
                    recent_topics=[],
                    last_agent_used=None,
                    conversation_summary="",
                    key_insights=[]
                )
                message_count = default_context.message_count

            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": final_state.get("final_response", "No response generated"),
                "agent_used": agent_used,
                "execution_path": final_state.get("execution_path", []),
                "message_count": message_count,
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"❌ Router execution error: {e}")
            import traceback
            traceback.print_exc()  # This will help debug the exact line causing issues
            
            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": "I encountered a system error. Please try again with a simpler question about your spending or budget.",
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