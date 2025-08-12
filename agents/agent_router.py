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

# Enhanced Pydantic models for routing
class AgentRouting(BaseModel):
    """Structured agent routing decision with direct response capability"""
    
    is_relevant: bool = Field(
        description="True if query is related to personal finance, spending, budgeting, or banking; False otherwise"
    )
    primary_agent: Literal["spending", "budget", "direct_response", "irrelevant"] = Field(
        description="Primary agent: 'spending' for transaction analysis, 'budget' for budget management, 'direct_response' for clarifications using context, 'irrelevant' for off-topic"
    )
    query_category: str = Field(
        description="Category: finance_spending, finance_budget, clarification_question, greeting, off_topic, unclear"
    )
    confidence: float = Field(
        description="Routing confidence score between 0 and 1", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision"
    )
    can_answer_directly: bool = Field(
        description="True if router can answer using conversation context without agent help"
    )
    suggested_response_tone: Optional[str] = Field(
        default=None,
        description="For irrelevant queries: friendly_redirect, polite_decline, or clarification_needed"
    )


@dataclass
class ConversationContext:
    """Enhanced conversation context for memory management"""
    client_id: int
    session_start: datetime
    last_interaction: datetime
    message_count: int
    recent_topics: List[str]
    last_agent_used: Optional[str]
    conversation_summary: str
    key_insights: List[str]

    # Enhanced context fields
    conversation_history: List[Dict[str, Any]]  
    last_user_query: Optional[str]  
    last_agent_response: Optional[str]
    last_sql_executed: Optional[str]  # NEW: Store last SQL for context
    last_query_results: Optional[Dict[str, Any]]  # NEW: Store last results


class MultiAgentState(TypedDict):
    """Enhanced state for multi-agent routing system"""
    
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
    enhanced_query: Optional[str]


class PersonalFinanceRouter:
    """
    Enhanced router that can handle clarification questions directly using conversation context.
    Routes complex queries to appropriate agents and handles simple follow-ups independently.
    """

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
        enable_memory: bool = True,
        memory_db_path: str = "conversation_memory.db"
    ):
        print(f"Initializing Enhanced PersonalFinanceRouter...")
        print(f"Data sources: Client CSV, Overall CSV")
        print(f"Memory enabled: {enable_memory}")

        # Initialize agents
        print("🔄 Loading Spending Agent...")
        self.spending_agent = SpendingAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False
        )

        print("💰 Loading Budget Agent...")
        self.budget_agent = BudgetAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False
        )

        # Initialize LLM for routing and direct responses
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

        # Build enhanced router graph
        self.graph = self._build_router_graph()
        
        print("✅ Enhanced PersonalFinanceRouter initialized successfully!")

    def _build_router_graph(self) -> StateGraph:
        """Build the enhanced multi-agent routing workflow with direct response capability"""

        workflow = StateGraph(MultiAgentState)

        # Enhanced router workflow nodes
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("query_router", self._query_router_node)
        workflow.add_node("direct_responder", self._direct_responder_node)  # NEW NODE
        workflow.add_node("spending_agent_node", self._spending_agent_node)
        workflow.add_node("budget_agent_node", self._budget_agent_node)
        workflow.add_node("irrelevant_handler", self._irrelevant_handler_node)
        workflow.add_node("response_synthesizer", self._response_synthesizer_node)
        workflow.add_node("memory_updater", self._memory_updater_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point
        workflow.set_entry_point("context_manager")

        # Enhanced routing logic
        workflow.add_edge("context_manager", "query_router")
        workflow.add_conditional_edges(
            "query_router",
            self._route_to_agents,
            {
                "spending": "spending_agent_node",
                "budget": "budget_agent_node",
                "direct_response": "direct_responder",  # NEW ROUTE
                "irrelevant": "irrelevant_handler",
                "error": "error_handler"
            }
        )

        # All agent paths go to response synthesizer
        workflow.add_edge("spending_agent_node", "response_synthesizer")
        workflow.add_edge("budget_agent_node", "response_synthesizer")
        workflow.add_edge("direct_responder", "response_synthesizer")  # NEW EDGE
        workflow.add_edge("irrelevant_handler", "response_synthesizer")
        
        # Final flow
        workflow.add_edge("response_synthesizer", "memory_updater")
        workflow.add_edge("memory_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)

    def _context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced conversation context management with SQL and results tracking"""

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
            
            if session_id in self.contexts:
                context = self.contexts[session_id]
                context.last_interaction = datetime.now()
                context.message_count += 1
                context.last_user_query = user_query
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
                    conversation_history=[],
                    last_user_query=user_query,
                    last_agent_response=None,
                    last_sql_executed=None,  # NEW
                    last_query_results=None  # NEW
                )
                self.contexts[session_id] = context
                print(f"[DEBUG] 🆕 New conversation started")

            # Enhanced topic detection
            query_lower = user_query.lower()
            detected_topics = []
            
            if any(word in query_lower for word in ["budget", "create", "set up", "overspend", "allocate"]):
                detected_topics.append("budget")
            if any(word in query_lower for word in ["spend", "spent", "spending", "transaction", "category", "total", "where did i"]):
                detected_topics.append("spending")
            if any(word in query_lower for word in ["save", "savings", "saved"]):
                detected_topics.append("savings")
            if any(word in query_lower for word in ["compare", "comparison", "vs", "versus", "against", "average", "similar"]):
                detected_topics.append("comparison")

            # Add detected topics
            for topic in detected_topics:
                if topic not in context.recent_topics:
                    context.recent_topics.append(topic)
                    print(f"[DEBUG] 🏷️ Added topic: {topic}")

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
        """Enhanced intelligent query routing with direct response detection"""

        try:
            print("🎯 [DEBUG] Routing query to appropriate agent...")

            context = state.get("conversation_context")
            original_query = state["user_query"]
            
            # Build conversation context for routing decision
            recent_context = ""
            if context:
                recent_context = f"""
CONVERSATION CONTEXT:
- Message count: {context.message_count}
- Recent topics: {', '.join(context.recent_topics)}
- Last agent used: {context.last_agent_used or 'None'}
- Last SQL executed: {context.last_sql_executed or 'None'}
- Last query results available: {'Yes' if context.last_query_results else 'No'}
"""

            # Enhanced routing prompt with direct response capability
            routing_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are an intelligent query router for a personal finance assistant system.

Your job is to determine if queries should go to agents OR be answered directly using conversation context.

ROUTING OPTIONS:

1. **DIRECT_RESPONSE** (can_answer_directly: true):
   Use this when the router can answer using conversation context without agent help:
   - Clarification questions about previous queries ("which week?", "what period?", "what dates?")
   - Questions about methodology ("how did you calculate that?", "what data did you use?")
   - Follow-up questions that reference previous results ("based on last query...")
   - Simple confirmations or explanations about previous interactions

2. **SPENDING AGENT** (primary_agent: "spending"):
   - New spending analysis requests
   - Transaction queries requiring database access
   - Complex spending patterns analysis

3. **BUDGET AGENT** (primary_agent: "budget"):
   - Budget creation, management, tracking
   - Budget vs actual analysis

4. **IRRELEVANT** (primary_agent: "irrelevant"):
   - Non-finance topics

EXAMPLES OF DIRECT_RESPONSE QUERIES:
- "Based on last query which week are you referring to?"
- "What period did you analyze?"
- "How did you calculate that amount?"
- "What data source did you use?"
- "Can you clarify the timeframe?"
- "What does 'last month' mean exactly?"

{recent_context}

Provide a JSON response with:
- is_relevant: boolean
- primary_agent: string ("spending", "budget", "direct_response", or "irrelevant")
- query_category: string
- confidence: number 0-1
- reasoning: string
- can_answer_directly: boolean (true if router can handle this)

Query: "{original_query}"
"""
                ),
                (
                    "human",
                    "Route this query and respond with a JSON object containing the routing decision."
                )
            ])

            response = self.llm.invoke(
                routing_prompt.format_messages(
                    original_query=original_query,
                    recent_context=recent_context
                )
            )
            
            # Parse JSON response with improved error handling
            try:
                response_text = response.content.strip()
                
                # Extract JSON from response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "{" in response_text and "}" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    response_text = response_text[json_start:json_end]
                
                routing_dict = json.loads(response_text)
                
                # Validate and set defaults
                required_fields = ["is_relevant", "primary_agent", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in routing_dict:
                        routing_dict[field] = self._get_default_routing_value(field)
                
                routing_dict.setdefault("query_category", "unknown")
                routing_dict.setdefault("can_answer_directly", False)
                routing_dict.setdefault("suggested_response_tone", None)
                
                print(f"[DEBUG] ✅ Query relevance: {routing_dict['is_relevant']}")
                print(f"[DEBUG] ✅ Routed to: {routing_dict['primary_agent']} ({routing_dict['reasoning']})")
                print(f"[DEBUG] ✅ Confidence: {routing_dict['confidence']:.2f}")
                
            except (json.JSONDecodeError, ValueError) as e:
                routing_dict = self._enhanced_fallback_routing(original_query, context)
                
            state["routing_decision"] = routing_dict
            state["execution_path"].append("query_router")

        except Exception as e:
            print(f"❌ Query routing error: {e}")
            state["error"] = f"Query routing failed: {e}"
            state["routing_decision"] = None

        return state

    def _enhanced_fallback_routing(self, query: str, context) -> Dict[str, Any]:
        """Enhanced fallback routing with direct response detection"""
        
        query_lower = query.lower()
        
        # Detect clarification questions
        clarification_patterns = [
            "which", "what period", "what timeframe", "what dates", "based on last",
            "how did you", "what data", "can you clarify", "what does", "mean exactly"
        ]
        
        if any(pattern in query_lower for pattern in clarification_patterns):
            if context and (context.last_sql_executed or context.last_query_results):
                return {
                    "is_relevant": True,
                    "primary_agent": "direct_response",
                    "query_category": "clarification_question",
                    "confidence": 0.8,
                    "reasoning": "Clarification question with available context",
                    "can_answer_directly": True
                }
        
        # Budget keywords
        if any(word in query_lower for word in ["budget", "create", "set up", "overspend"]):
            return {
                "is_relevant": True,
                "primary_agent": "budget",
                "query_category": "finance_budget",
                "confidence": 0.7,
                "reasoning": "Budget-related keywords detected",
                "can_answer_directly": False
            }
        
        # Spending keywords
        if any(word in query_lower for word in ["spend", "spending", "transaction", "total"]):
            return {
                "is_relevant": True,
                "primary_agent": "spending",
                "query_category": "finance_spending", 
                "confidence": 0.7,
                "reasoning": "Spending-related keywords detected",
                "can_answer_directly": False
            }
        
        # Default to irrelevant
        return {
            "is_relevant": False,
            "primary_agent": "irrelevant",
            "query_category": "off_topic",
            "confidence": 0.6,
            "reasoning": "No clear finance topic detected",
            "can_answer_directly": False
        }

    def _get_default_routing_value(self, field: str):
        """Get default values for missing routing fields"""
        defaults = {
            "is_relevant": False,
            "primary_agent": "irrelevant", 
            "confidence": 0.5,
            "reasoning": "Default routing due to parsing error"
        }
        return defaults.get(field, None)

    def _direct_responder_node(self, state: MultiAgentState) -> MultiAgentState:
        """NEW NODE: Handle queries directly using conversation context"""
        
        try:
            print("🎯 [DEBUG] Handling query directly using conversation context...")
            
            context = state.get("conversation_context")
            user_query = state["user_query"]
            
            if not context:
                state["primary_response"] = "I don't have enough conversation context to answer that question directly. Please ask me about your spending or budget!"
                state["execution_path"].append("direct_responder")
                return state
                
            # Build comprehensive context for LLM
            context_info = self._build_detailed_context(context)
            
            direct_response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful personal banking assistant with access to conversation context.

The user is asking a clarification or follow-up question that you can answer using the conversation history and context provided below.

Your response should be:
1. Direct and informative
2. Reference the specific context (dates, amounts, methods) from previous interactions
3. Be helpful and clear about what was done previously
4. Offer to provide more specific information if needed

DO NOT make up information. Only use what's provided in the context below.

CONVERSATION CONTEXT:
{context_info}

Answer the user's question directly using this context."""),
                ("human", "User question: {user_query}")
            ])
            
            response = self.llm.invoke(
                direct_response_prompt.format_messages(
                    context_info=context_info,
                    user_query=user_query
                )
            )
            
            state["primary_response"] = response.content
            state["execution_path"].append("direct_responder")
            
            print(f"✅ Direct response generated using conversation context")
            
        except Exception as e:
            print(f"❌ Direct responder error: {e}")
            state["primary_response"] = f"I couldn't process your clarification question due to an error. Please rephrase or ask a new question about your spending or budget."
            state["execution_path"].append("direct_responder")
            
        return state

    def _build_detailed_context(self, context: ConversationContext) -> str:
        """Build detailed context string for direct responses"""
        
        context_parts = [
            f"Session info: Message #{context.message_count}, started {context.session_start.strftime('%Y-%m-%d %H:%M')}"
        ]
        
        # Add recent topics
        if context.recent_topics:
            context_parts.append(f"Recent topics discussed: {', '.join(context.recent_topics)}")
        
        # Add last SQL executed (very important for methodology questions)
        if context.last_sql_executed:
            context_parts.append(f"Last SQL query executed: {context.last_sql_executed}")
        
        # Add last query results
        if context.last_query_results:
            context_parts.append(f"Last query results: {json.dumps(context.last_query_results, indent=2, default=str)}")
        
        # Add conversation history (last 3 interactions)
        if context.conversation_history:
            context_parts.append("\nRecent conversation:")
            recent_convs = context.conversation_history[-3:]
            
            for i, conv in enumerate(recent_convs, 1):
                if conv.get("user_query") and conv.get("agent_response"):
                    context_parts.append(f"  {i}. User: {conv['user_query']}")
                    context_parts.append(f"     Assistant: {conv['agent_response'][:200]}...")
                    if conv.get("sql_executed"):
                        context_parts.append(f"     SQL used: {conv['sql_executed']}")
        
        # Add key insights
        if context.key_insights:
            context_parts.append(f"\nKey insights from our discussion: {', '.join(context.key_insights)}")
        
        return "\n".join(context_parts)

    def _spending_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced spending agent execution - SIMPLIFIED VERSION"""

        try:
            print("📊 [DEBUG] Executing spending agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                state["error"] = "No valid query provided to spending agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your spending again."
                return state

            # Pass conversation context to spending agent
            result = self.spending_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process,
                conversation_context=state.get("conversation_context")
            )



            # SIMPLIFIED: Just set the response directly
            if result and result.get("response"):
                state["primary_response"] = result["response"]

            else:
                state["primary_response"] = "I processed your spending query but couldn't generate a response."
            
            # Enhanced context update with SQL and results tracking
            if state.get("conversation_context") and result and result.get("success"):
                context = state["conversation_context"]
                context.last_agent_used = "spending"
                
                # Store SQL and results for future reference
                if result.get('sql_executed'):
                    context.last_sql_executed = result['sql_executed'][-1] if result['sql_executed'] else None
                    print(f"[DEBUG] 💾 Stored SQL: {context.last_sql_executed[:100] if context.last_sql_executed else 'None'}...")
                
                if result.get('raw_data'):
                    context.last_query_results = result['raw_data']
                    print(f"[DEBUG] 💾 Stored query results: {len(result['raw_data'])} data chunks")
            
            # CRITICAL DEBUG: Check state before returning
            print(f"[DEBUG] 🔍 FINAL STATE CHECK:")
            print(f"[DEBUG] 🔍 state['primary_response'] exists: {bool(state.get('primary_response'))}")
            print(f"[DEBUG] 🔍 state['primary_response'] length: {len(state.get('primary_response', ''))}")
            print(f"[DEBUG] 🔍 state['primary_response'] content: {state.get('primary_response', '')[:100]}...")
                    
            state["execution_path"].append("spending_agent")
            print(f"✅ Spending agent completed: {result.get('success', False) if result else False}")

        except Exception as e:
            print(f"❌ Spending agent error: {e}")
            state["error"] = f"Spending agent failed: {e}"
            state["primary_response"] = "I encountered an error processing your spending query."

        return state

    def _budget_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced budget agent execution with context storage"""

        try:
            print("💰 [DEBUG] Executing budget agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                state["error"] = "No valid query provided to budget agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your budget again."
                return state

            result = self.budget_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process
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
            state["primary_response"] = None

        return state

    def _irrelevant_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle irrelevant queries with natural, helpful responses"""
        
        try:
            print("🚫 [DEBUG] Handling irrelevant query...")
            
            user_query = state["user_query"]
            context = state.get("conversation_context")
            
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly personal banking assistant. The user has asked a question that's outside your expertise area of personal finance, spending, and budgeting.

Your response should be:
1. Warm and understanding
2. Briefly acknowledge their question
3. Naturally redirect to what you can help with
4. Offer specific examples relevant to banking/finance

Keep it conversational - like a bank employee politely redirecting the conversation."""),
                ("human", """The user asked: "{query}"

Provide a brief, natural response that redirects them to finance topics.""")
            ])
            
            response = self.llm.invoke(
                response_prompt.format_messages(query=user_query)
            )
            
            base_response = response.content
            
            # Add context-aware suggestions
            suggestions = ""
            if context and context.recent_topics:
                if "spending" in context.recent_topics:
                    suggestions = "\n\nSince we were talking about your spending, would you like to dive deeper into any particular category or time period?"
                elif "budget" in context.recent_topics:
                    suggestions = "\n\nWe were discussing budgets - would you like help setting up budgets for specific categories?"
            else:
                suggestions = "\n\nI can help you understand where your money goes each month, set up budgets, or see how your spending compares to others. What interests you most?"
            
            state["primary_response"] = base_response + suggestions
            state["execution_path"].append("irrelevant_handler")
            
        except Exception as e:
            print(f"❌ Irrelevant handler error: {e}")
            state["primary_response"] = f"That's outside my area of expertise, but I'm here to help you with your finances! I can analyze your spending patterns, help you create budgets, or show you how you compare to similar customers. What would you like to explore?"
            
        return state

    def _response_synthesizer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced response synthesis with better debugging"""

        try:
            print("🔧 [DEBUG] Synthesizing final response...")

            primary_response = state.get("primary_response", "")
            context = state.get("conversation_context")
            routing = state.get("routing_decision", {})
            
            # ENHANCED DEBUG: Check what we actually have
            print(f"[DEBUG] 🔍 Primary response exists: {bool(primary_response)}")
            print(f"[DEBUG] 🔍 Primary response length: {len(primary_response) if primary_response else 0}")
            print(f"[DEBUG] 🔍 Primary response sample: {primary_response[:100] if primary_response else 'None'}...")
            print(f"[DEBUG] 🔍 Routing decision: {routing.get('primary_agent', 'None')}")

            # FIXED: Better validation of primary_response
            if not primary_response or primary_response.strip() == "":
                print(f"[DEBUG] ❌ No valid primary response found")
                if state.get("error"):
                    state["final_response"] = f"I encountered an issue: {state['error']}\n\nPlease try rephrasing your question about spending or budgeting."
                else:
                    state["final_response"] = "I apologize, but I couldn't generate a response to your query. Please try asking about your spending patterns or budget management."
                state["execution_path"].append("response_synthesizer")
                return state

            # Handle different agent types
            if not routing:
                print(f"[DEBUG] ⚠️ No routing decision, using primary response as-is")
                state["final_response"] = primary_response
                state["execution_path"].append("response_synthesizer")
                return state

            agent_used = routing.get("primary_agent")
            print(f"[DEBUG] 🎯 Processing response from: {agent_used}")
            
            if agent_used == "irrelevant":
                state["final_response"] = primary_response
            elif agent_used == "direct_response":
                # For direct responses, use as-is
                state["final_response"] = primary_response
                print(f"[DEBUG] ✅ Direct response provided using conversation context")
            else:
                # For spending/budget agents, add cross-agent suggestions
                state["final_response"] = primary_response
                
                recent_topics = context.recent_topics if context else []
                suggestion = ""
                if agent_used == "spending" and "budget" not in recent_topics:
                    suggestion = "\n\n💡 *Based on this spending analysis, would you like help creating or reviewing a budget?*"
                elif agent_used == "budget" and context and context.message_count <= 2:
                    suggestion = "\n\n🎯 *I can also analyze your historical spending patterns to suggest better budget allocations.*"
                
                state["final_response"] += suggestion

            print(f"[DEBUG] ✅ Final response length: {len(state['final_response'])}")
            print(f"[DEBUG] ✅ Final response sample: {state['final_response'][:100]}...")
            
            state["execution_path"].append("response_synthesizer")
            print("✅ Response synthesis complete")

        except Exception as e:
            print(f"❌ Response synthesis error: {e}")
            # FIXED: Ensure we don't lose the original response due to synthesis errors
            primary_response = state.get("primary_response", "")
            if primary_response and primary_response.strip():
                print(f"[DEBUG] 🛟 Using primary response despite synthesis error")
                state["final_response"] = primary_response
            else:
                state["final_response"] = "I apologize, but I encountered an issue processing your request. Please try asking about your spending or budget."

        return state

    def _memory_updater_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced memory update with SQL and results storage"""

        try:
            print("💾 [DEBUG] Updating conversation memory...")

            context = state.get("conversation_context")
            if context:
                # Store the current interaction with enhanced information
                current_interaction = {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": context.last_user_query,
                    "agent_response": state.get("final_response"),
                    "agent_used": state.get("routing_decision", {}).get("primary_agent"),
                    "can_answer_directly": state.get("routing_decision", {}).get("can_answer_directly", False),
                    "success": state.get("error") is None
                }
                
                # Add to conversation history
                context.conversation_history.append(current_interaction)
                
                # Store the response for potential follow-up queries
                context.last_agent_response = state.get("final_response")
                
                # Keep only last 10 interactions to prevent memory bloat
                context.conversation_history = context.conversation_history[-10:]
                
                # Update conversation summary based on recent interactions
                if len(context.conversation_history) >= 2:
                    recent_queries = [h["user_query"] for h in context.conversation_history[-3:]]
                    context.conversation_summary = f"Recent topics: {', '.join(context.recent_topics)}. Last queries: {'; '.join(recent_queries[-2:])}"
                
                # Extract key insights from the response
                if state.get("final_response"):
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

                print(f"[DEBUG] 💾 Stored interaction: {context.last_user_query[:50]}... -> {len(state.get('final_response', ''))} chars")
                print(f"[DEBUG] 📚 Total conversation history: {len(context.conversation_history)} interactions")

            state["execution_path"].append("memory_updater")
            print("✅ Memory updated with enhanced query-response history")

        except Exception as e:
            print(f"❌ Memory update error: {e}")

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
        """Enhanced routing logic with direct response capability"""

        # Check for errors first
        if state.get("error"):
            return "error"

        routing_decision = state.get("routing_decision")
        if not routing_decision:
            state["error"] = "No routing decision made"
            return "error"

        primary_agent = routing_decision.get("primary_agent")
        
        # Enhanced routing based on agent decision
        if primary_agent == "spending":
            return "spending"
        elif primary_agent == "budget":
            return "budget"
        elif primary_agent == "direct_response":
            return "direct_response"  # NEW ROUTE
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
        """Enhanced chat interface with direct response capability"""

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
            enhanced_query=None
        )

        try:
            config = {"configurable": {"thread_id": session_id}} if self.memory else {}
            final_state = self.graph.invoke(initial_state, config=config)

            # Handle case where routing_decision might be None
            agent_used = "unknown"
            can_answer_directly = False
            if final_state.get("routing_decision"):
                agent_used = final_state["routing_decision"].get("primary_agent", "unknown")
                can_answer_directly = final_state["routing_decision"].get("can_answer_directly", False)

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
                    key_insights=[],
                    conversation_history=[],
                    last_user_query=user_query,
                    last_agent_response=None,
                    last_sql_executed=None,
                    last_query_results=None
                )
                message_count = default_context.message_count

            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": final_state.get("final_response", "No response generated"),
                "agent_used": agent_used,
                "can_answer_directly": can_answer_directly,  # NEW FIELD
                "execution_path": final_state.get("execution_path", []),
                "message_count": message_count,
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"❌ Router execution error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": "I encountered a system error. Please try again with a simpler question about your spending or budget.",
                "agent_used": "error",
                "can_answer_directly": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
            }

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get enhanced conversation summary for a session"""
        
        if session_id in self.contexts:
            context = self.contexts[session_id]
            return {
                "session_id": session_id,
                "client_id": context.client_id,
                "session_start": context.session_start.isoformat(),
                "message_count": context.message_count,
                "recent_topics": context.recent_topics,
                "last_agent_used": context.last_agent_used,
                "last_sql_executed": context.last_sql_executed,
                "key_insights": context.key_insights,
                "conversation_summary": context.conversation_summary,
                "has_query_results": context.last_query_results is not None
            }
        else:
            return {"error": "Session not found"}

    def get_conversation_history(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get conversation history with enhanced details"""
        if session_id in self.contexts:
            context = self.contexts[session_id]
            history = context.conversation_history[-limit:] if context.conversation_history else []
            return history
        else:
            return []


def enhanced_interactive_chat_demo():
    """Enhanced interactive chat demo with direct response testing"""
    
    print("=" * 60)
    print("💬 Chatting as User ID: 430")
    print("🔧 Type 'quit' to exit, 'test' for agent tests")


    # Initialize enhanced router
    client_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv"
    overall_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/overall_data.csv"

    try:
        router = PersonalFinanceRouter(
            client_csv_path=client_csv,
            overall_csv_path=overall_csv,
            enable_memory=False
        )

        client_id = 430
        session_id = f"enhanced_demo_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        print(f"Session ID: {session_id}")
        print("-" * 40)

        # Test sequence that demonstrates direct response capability
        test_sequence = [
            "How much did I spend last month?"
        ]

        for i, query in enumerate(test_sequence, 1):
            print(f"\n👤 **Test {i}**: {query}")
            
            result = router.chat(
                client_id=client_id,
                user_query=query,
                session_id=session_id
            )

            # Enhanced result display
            agent_emoji = {
                'spending': '📊',
                'budget': '💰',
                'direct_response': '🎯',
                'irrelevant': '🚫',
                'error': '❌',
                'unknown': '🤖'
            }.get(result['agent_used'], '🤖')
            
            agent_name = result['agent_used'].replace('_', ' ').title()
            direct_indicator = " (DIRECT)" if result.get('can_answer_directly') else ""
            
            print(f"🤖 **{agent_name}{direct_indicator}**: {result['response']}")
            print(f"⚙️  *Success: {result['success']}, Path: {' → '.join(result['execution_path'])}*")
            
            if result.get('error'):
                print(f"❌ *Error: {result['error']}*")
            else:
                print("✅ *Test passed!*")

        print(f"\n🎯 **INTERACTIVE MODE**")
        print("Try asking clarification questions like:")
        print("- 'What timeframe did you use?'")
        print("- 'How did you calculate that?'") 
        print("- 'Which dates exactly?'")
        print("-" * 40)

        # Interactive mode
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for using GX Bank Personal Finance Assistant!")
                    break
                elif user_input.lower() == 'test':
                    # Show conversation summary
                    summary = router.get_conversation_summary(session_id)
                    print(f"\n📊 **Conversation Summary**")
                    print(f"Messages: {summary.get('message_count', 0)}")
                    print(f"Topics: {summary.get('recent_topics', [])}")
                    print(f"Last Agent: {summary.get('last_agent_used', 'None')}")
                    print(f"Has SQL History: {summary.get('last_sql_executed') is not None}")
                    continue
                elif user_input.lower() == 'history':
                    # Show conversation history
                    history = router.get_conversation_history(session_id)
                    print(f"\n📚 **Recent Conversation History**")
                    for i, item in enumerate(history, 1):
                        print(f"{i}. User: {item['user_query']}")
                        print(f"   Agent: {item['agent_response'][:100]}...")
                        print(f"   Used: {item['agent_used']}")
                    continue
                elif not user_input:
                    continue

                result = router.chat(
                    client_id=client_id,
                    user_query=user_input,
                    session_id=session_id
                )

                # Enhanced response display
                agent_emoji = {
                    'spending': '📊',
                    'budget': '💰',
                    'direct_response': '🎯',
                    'irrelevant': '🚫',
                    'error': '❌',
                    'unknown': '🤖'
                }.get(result['agent_used'], '🤖')
                
                agent_name = result['agent_used'].replace('_', ' ').title()
                direct_indicator = " (DIRECT)" if result.get('can_answer_directly') else ""
                
                print(f"\n{agent_emoji} **{agent_name}{direct_indicator}**: {result['response']}")
                
                # Show execution path
                if result.get('execution_path'):
                    path_display = ' → '.join(result['execution_path'])
                    print(f"🛤️  *Path: {path_display}*")
                
                # Show routing info
                print(f"🎯 *Message #{result.get('message_count', 0)} | Direct Response: {result.get('can_answer_directly', False)}*")
                
                # Handle errors gracefully
                if result.get('error'):
                    print(f"\n⚠️ *Technical note: {result['error']}*")

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize enhanced router: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    enhanced_interactive_chat_demo()