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
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import os
import sys
import traceback

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
# Import our enhanced RAG agent
from agents.rag_agent import RAGAgent

load_dotenv()

# Pydantic models for routing with better domain detection
class AgentRouting(BaseModel):
    """Structured agent routing decision with domain relevance"""
    
    is_relevant: bool = Field(
        description="True if query is related to personal finance, spending, budgeting, or banking; False otherwise"
    )
    primary_agent: Literal["spending", "budget", "rag", "irrelevant"] = Field(
        description="Primary agent: 'spending' for transaction analysis, 'budget' for budget management, 'rag' for external banking info, 'irrelevant' for off-topic"
    )
    query_category: str = Field(
        description="Category: finance_spending, finance_budget, finance_external, finance_general, greeting, off_topic, unclear"
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
    """Conversation context for memory management"""
    client_id: int
    session_start: datetime
    last_interaction: datetime
    message_count: int
    recent_topics: List[str]
    last_agent_used: Optional[str]
    conversation_summary: str
    key_insights: List[str]
    conversation_history: List[Dict[str, Any]]  
    last_user_query: Optional[str]  
    last_agent_response: Optional[str]


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
    enhanced_query: Optional[str]


class EnhancedPersonalFinanceRouter:
    """
    Enhanced router with real RAG agent collaboration.
    The RAG agent can now actually collaborate with spending and budget agents.
    """

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
    ):
        print(f"Initializing Enhanced PersonalFinanceRouter...")
        print(f"Data sources: Client CSV, Overall CSV")

        # Initialize agents in the right order for collaboration
        print("📊 Loading Spending Agent...")
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

        # ENHANCED: Initialize RAG agent with references to other agents
        print("🔍 Loading Enhanced RAG Agent with collaboration...")
        self.rag_agent = RAGAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False,  # We'll handle memory at router level
            # CRITICAL: Pass agent references for real collaboration
            spending_agent=self.spending_agent,
            budget_agent=self.budget_agent
        )

        self.llm = AzureChatOpenAI(
                        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),            
                        temperature=0,
                    )
        
        # Set up routing parser
        self.routing_parser = PydanticOutputParser(pydantic_object=AgentRouting)

        # Conversation contexts (in-memory cache)
        self.contexts: Dict[str, ConversationContext] = {}

        # Build router graph
        self.graph = self._build_router_graph()

    def _build_router_graph(self) -> StateGraph:
        """Build the multi-agent routing workflow"""

        workflow = StateGraph(MultiAgentState)

        # Router workflow nodes
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("query_router", self._query_router_node)
        workflow.add_node("spending_agent_node", self._spending_agent_node)
        workflow.add_node("budget_agent_node", self._budget_agent_node)
        workflow.add_node("rag_agent_node", self._rag_agent_node)  # NEW: RAG agent node
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
                "rag": "rag_agent_node",  # NEW: RAG routing
                "irrelevant": "irrelevant_handler",
                "error": "error_handler"
            }
        )

        workflow.add_edge("spending_agent_node", "response_synthesizer")
        workflow.add_edge("budget_agent_node", "response_synthesizer")
        workflow.add_edge("rag_agent_node", "response_synthesizer")  # NEW: RAG to synthesizer
        workflow.add_edge("irrelevant_handler", "response_synthesizer")
        workflow.add_edge("response_synthesizer", "memory_updater")
        workflow.add_edge("memory_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=None)  # Using in-memory contexts

    def _context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced conversation context and memory management"""

        try:
            print("🧠 [DEBUG] Managing conversation context...")

            user_query = state["user_query"].strip()
            common_followups = ['ok', 'yes', 'no', 'show', 'tell', 'more', 'continue', 'please']
            query_words = user_query.lower().split()
            
            if len(user_query) <= 2 and user_query.lower() not in ['no', 'yes', 'ok']:
                print(f"[DEBUG] ⚠️ Invalid query detected: '{user_query}'")
                state["error"] = "Please provide a more complete question about your finances."
                return state
            elif len(query_words) <= 3 and all(word in common_followups for word in query_words):
                print(f"[DEBUG] 🔄 Follow-up command detected: '{user_query}'")

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
                    last_agent_response=None  
                )
                self.contexts[session_id] = context

            query_lower = user_query.lower()
            last_interaction = context.conversation_history[-1] if context.conversation_history else None
            
            state["enhanced_query"] = None
            
            # Enhanced topic detection for RAG queries
            detected_topics = []
            
            # Existing topics
            if any(word in query_lower for word in ["budget", "create", "set up", "overspend", "allocate"]):
                detected_topics.append("budget")
            if any(word in query_lower for word in ["spend", "spent", "spending", "transaction", "category", "total", "where did i"]):
                detected_topics.append("spending")
            if any(word in query_lower for word in ["save", "savings", "saved"]):
                detected_topics.append("savings")
            if any(word in query_lower for word in ["compare", "comparison", "vs", "versus", "against", "average", "similar"]):
                detected_topics.append("comparison")
            
            # NEW: Banking/external topics
            if any(word in query_lower for word in ["credit card", "loan", "mortgage", "account", "investment", "offer", "bank", "rate"]):
                detected_topics.append("banking")
            if any(word in query_lower for word in ["policy", "policies", "service", "services", "terms", "fees"]):
                detected_topics.append("policy")
            if any(word in query_lower for word in ["afford", "financing", "qualify"]):
                detected_topics.append("affordability")

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
        """Enhanced query routing with RAG agent support"""

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

            # Enhanced routing prompt with RAG support
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
- EXTERNAL banking products and services (credit cards, loans, policies)

ROUTING RULES:

1. **RELEVANT FINANCE QUERIES** (is_relevant: true):
- Personal spending/transactions → primary_agent: "spending"
- Budget management → primary_agent: "budget"
- External banking products/services → primary_agent: "rag"
- General finance questions → choose most appropriate agent
- **Questions about capabilities in finance context** → primary_agent: "spending"

2. **EXAMPLES OF RELEVANT QUERIES:**
- "How much did I spend last month?" → spending
- "Create a budget for groceries" → budget
- "What credit cards do you offer?" → rag
- "Based on my spending, which credit card suits me?" → rag (needs collaboration)
- "Can I afford a Ford Escape with your loan rates?" → rag (needs collaboration)
- "What are your auto loan policies?" → rag
- "How am I doing against my budget?" → budget
- "What can you help me with?" → spending (finance capabilities)

3. **RAG AGENT CRITERIA:**
- Questions about bank products (credit cards, loans, accounts)
- Questions about bank policies, rates, services
- Questions combining external banking info with personal data
- Affordability questions that need both loan info and personal data

4. **IRRELEVANT QUERIES** (is_relevant: false):
Examples of irrelevant:
- General knowledge ("What is the capital of France?")
- Weather, news, sports
- Programming/coding help (unless finance-related)
- Health, recipes, travel planning
- Entertainment recommendations

For these → primary_agent: "irrelevant"

{recent_context}

Provide a JSON response with these fields:
- is_relevant: boolean (true if finance-related, false otherwise)
- primary_agent: string ("spending", "budget", "rag", or "irrelevant")
- query_category: string (describe the type of query)
- confidence: number between 0 and 1
- reasoning: string (brief explanation)
- suggested_response_tone: string or null

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
                response_text = response.content.strip()
                
                # Try to extract JSON from the response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "{" in response_text and "}" in response_text:
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
                    print(f"[DEBUG] JSON parsing failed: {e}, using enhanced keyword-based routing")
                    routing_dict = self._enhanced_fallback_routing(original_query, enhanced_query, context)
                
                state["routing_decision"] = routing_dict
                
                if enhanced_query:
                    print(f"[DEBUG] 💡 Used enhanced query for routing")

            except Exception as parse_error:
                print(f"[DEBUG] Structured routing failed ({parse_error}), using enhanced fallback")
                routing_dict = self._enhanced_fallback_routing(original_query, enhanced_query, context)
                state["routing_decision"] = routing_dict

            state["execution_path"].append("query_router")

        except Exception as e:
            print(f"❌ Query routing error: {e}")
            state["error"] = f"Query routing failed: {e}"
            state["routing_decision"] = None

        return state

    def _enhanced_fallback_routing(self, original_query: str, enhanced_query: str, context) -> Dict[str, Any]:
        """Enhanced fallback routing with RAG agent support"""
        
        query_to_analyze = enhanced_query if enhanced_query else original_query
        query_lower = query_to_analyze.lower()
        original_lower = original_query.lower()
        
        # Enhanced keyword sets
        budget_keywords = ["budget", "create", "set up", "overspend", "allocate", "limit", "spending plan"]
        spending_keywords = ["spend", "spent", "spending", "transaction", "category", "total", "average", "where did i", "how much", "breakdown"]
        
        # NEW: RAG keywords for external banking info
        rag_keywords = [
            "credit card", "loan", "mortgage", "account", "investment", "offer", "rate", "policy", "service",
            "what do you offer", "what cards", "what loans", "afford", "financing", "qualify", "terms", "fees"
        ]
        
        followup_keywords = ["please show", "show me", "yes", "continue", "more", "details", "tell me", "ok"]
        
        # Check for RAG queries first
        if any(keyword in query_lower for keyword in rag_keywords):
            return {
                "is_relevant": True,
                "primary_agent": "rag",
                "query_category": "finance_external",
                "confidence": 0.8,
                "reasoning": "Contains external banking product/service keywords",
                "suggested_response_tone": None
            }
        elif any(word in query_lower for word in budget_keywords):
            return {
                "is_relevant": True,
                "primary_agent": "budget",
                "query_category": "finance_budget",
                "confidence": 0.8,
                "reasoning": "Contains budget-related keywords",
                "suggested_response_tone": None
            }
        elif any(word in query_lower for word in spending_keywords):
            return {
                "is_relevant": True,
                "primary_agent": "spending",
                "query_category": "finance_spending",
                "confidence": 0.8,
                "reasoning": "Contains spending-related keywords",
                "suggested_response_tone": None
            }
        elif any(word in original_lower for word in followup_keywords):
            # Enhanced follow-up handling
            if context and context.last_agent_used:
                return {
                    "is_relevant": True,
                    "primary_agent": context.last_agent_used,
                    "query_category": "finance_followup",
                    "confidence": 0.7,
                    "reasoning": f"Follow-up question, continuing with {context.last_agent_used} agent",
                    "suggested_response_tone": None
                }
            else:
                return {
                    "is_relevant": True,
                    "primary_agent": "spending",
                    "query_category": "finance_general",
                    "confidence": 0.6,
                    "reasoning": "Follow-up with no context, defaulting to spending",
                    "suggested_response_tone": None
                }
        else:
            # Check if it's finance-related but unclear
            finance_indicators = ["money", "financial", "bank", "payment", "purchase", "cost"]
            if any(indicator in query_lower for indicator in finance_indicators):
                return {
                    "is_relevant": True,
                    "primary_agent": "spending",
                    "query_category": "finance_general",
                    "confidence": 0.6,
                    "reasoning": "General finance-related query, defaulting to spending",
                    "suggested_response_tone": None
                }
            else:
                return {
                    "is_relevant": False,
                    "primary_agent": "irrelevant",
                    "query_category": "off_topic",
                    "confidence": 0.7,
                    "reasoning": "No finance-related keywords detected",
                    "suggested_response_tone": "friendly_redirect"
                }

    def _spending_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute spending agent query"""

        try:
            print("📊 [DEBUG] Executing spending agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process.")
                state["error"] = "No valid query provided to spending agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your spending again."
                return state
            
            result = self.spending_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process,
                conversation_context=state.get("conversation_context")
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
            state["primary_response"] = None

        return state

    def _budget_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute budget agent query"""

        try:
            print("💰 [DEBUG] Executing budget agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process.")
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

    def _rag_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """NEW: Execute RAG agent query with collaboration capabilities"""

        try:
            print("🔍 [DEBUG] Executing RAG agent query...")

            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process.")
                state["error"] = "No valid query provided to RAG agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about banking products or services again."
                return state

            # Execute RAG agent with collaboration capabilities
            result = self.rag_agent.process_query(
                client_id=state["client_id"],
                user_query=query_to_process
            )

            state["primary_response"] = result.get("response")
            
            # Update context
            if state.get("conversation_context"):
                state["conversation_context"].last_agent_used = "rag"

            state["execution_path"].append("rag_agent")
            
            # Enhanced logging for RAG results
            print(f"✅ RAG agent completed: {result.get('success', False)}")
            if result.get('collaboration_summary'):
                collab_summary = result['collaboration_summary']
                print(f"[DEBUG] 🤝 Collaboration summary: {collab_summary}")

        except Exception as e:
            print(f"❌ RAG agent error: {e}")
            state["error"] = f"RAG agent failed: {e}"
            state["primary_response"] = None

        return state

    def _response_synthesizer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced response synthesis"""

        try:
            print("🔧 [DEBUG] Synthesizing final response...")

            primary_response = state.get("primary_response", "")
            context = state.get("conversation_context")
            routing = state.get("routing_decision", {})

            if not primary_response:
                if state.get("error"):
                    state["final_response"] = f"I encountered an issue: {state['error']}\n\nPlease try rephrasing your question about spending, budgeting, or banking services."
                else:
                    state["final_response"] = "I apologize, but I couldn't generate a response to your query. Please try asking about your spending patterns, budget management, or banking products."
                return state

            if not routing:
                state["final_response"] = primary_response
                state["execution_path"].append("response_synthesizer")
                return state

            # Different handling for different agent types
            agent_used = routing.get("primary_agent")
            
            if agent_used == "irrelevant":
                state["final_response"] = primary_response
            elif agent_used == "rag":
                # RAG responses are already comprehensive, minimal processing needed
                state["final_response"] = primary_response
            else:
                # For spending/budget agents, add cross-agent suggestions
                prefix = ""
                if context and context.message_count > 1:
                    if context.last_agent_used and context.last_agent_used != agent_used:
                        prefix = f"Switching to {agent_used} analysis... "
                
                state["final_response"] = prefix + primary_response
                
                # Add intelligent cross-agent suggestions
                recent_topics = context.recent_topics if context else []
                suggestion = ""
                
                if agent_used == "spending" and "banking" in recent_topics:
                    suggestion = "\n\n💡 *I can also help you find banking products that match your spending patterns.*"
                elif agent_used == "budget" and "banking" in recent_topics:
                    suggestion = "\n\n💡 *Based on your budget, I can help you find suitable banking products.*"
                elif agent_used == "spending" and "budget" not in recent_topics:
                    suggestion = "\n\n💡 *Based on this spending analysis, would you like help creating or reviewing a budget?*"
                elif agent_used == "budget" and context and context.message_count <= 2:
                    suggestion = "\n\n🎯 *I can also analyze your historical spending patterns or help you explore banking products.*"
                
                state["final_response"] += suggestion

            state["execution_path"].append("response_synthesizer")
            print("✅ Response synthesis complete")

        except Exception as e:
            print(f"❌ Response synthesis error: {e}")
            state["final_response"] = state.get("primary_response", "I apologize, but I encountered an issue processing your request. Please try asking about your spending, budget, or banking services.")

        return state

    def _memory_updater_node(self, state: MultiAgentState) -> MultiAgentState:
        """Update conversation memory"""

        try:
            print("💾 [DEBUG] Updating conversation memory...")

            context = state.get("conversation_context")
            if context:
                current_interaction = {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": context.last_user_query,
                    "agent_response": state.get("final_response"),
                    "agent_used": state.get("routing_decision", {}).get("primary_agent"),
                    "success": state.get("error") is None
                }
                
                context.conversation_history.append(current_interaction)
                context.last_agent_response = state.get("final_response")
                context.conversation_history = context.conversation_history[-10:]
                
                if len(context.conversation_history) >= 2:
                    recent_queries = [h["user_query"] for h in context.conversation_history[-3:]]
                    context.conversation_summary = f"Recent topics: {', '.join(context.recent_topics)}. Last queries: {'; '.join(recent_queries[-2:])}"
                
                # Extract insights
                if state.get("final_response"):
                    response = state["final_response"]
                    if "$" in response and any(word in response.lower() for word in ["spent", "budget", "total"]):
                        import re
                        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', response)
                        if amounts:
                            insight = f"Recent discussion: {amounts[0]} mentioned"
                            if insight not in context.key_insights:
                                context.key_insights.append(insight)

                context.key_insights = context.key_insights[-5:]

                print(f"[DEBUG] 💾 Stored interaction: {context.last_user_query[:50]}... -> {len(state.get('final_response', ''))} chars")
                print(f"[DEBUG] 📚 Total conversation history: {len(context.conversation_history)} interactions")

            state["execution_path"].append("memory_updater")
            print("✅ Memory updated with query-response history")

        except Exception as e:
            print(f"❌ Memory update error: {e}")

        return state

    def _irrelevant_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle irrelevant queries with natural responses"""
        
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
                elif "banking" in context.recent_topics:
                    suggestions = "\n\nI can continue helping you explore banking products that match your financial needs."
            else:
                suggestions = "\n\nI can help you understand where your money goes each month, set up budgets, explore banking products, or see how your spending compares to others. What interests you most?"
            
            state["primary_response"] = base_response + suggestions
            state["execution_path"].append("irrelevant_handler")
            
            print(f"✅ Handled irrelevant query naturally")
            
        except Exception as e:
            print(f"❌ Irrelevant handler error: {e}")
            state["primary_response"] = f"That's outside my area of expertise, but I'm here to help you with your finances! I can analyze your spending patterns, help you create budgets, explore banking products, or show you how you compare to similar customers. What would you like to explore?"
            
        return state

    def _error_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle routing and execution errors"""

        error_message = state.get("error", "Unknown routing error")
        print(f"🔧 [DEBUG] Handling router error: {error_message}")
        
        user_query = state["user_query"]

        if "Invalid query" in error_message:
            state["final_response"] = "I'd be happy to help! Could you tell me a bit more about what you'd like to know about your finances? I can help with spending analysis, budgeting, banking products, or comparing your habits to others."
        elif "Intent classification" in error_message or "routing" in error_message.lower():
            state["final_response"] = "I want to make sure I understand what you're looking for. Are you interested in seeing your spending patterns, setting up a budget, exploring banking products, or something else financial? Just let me know!"
        elif "No valid query" in error_message:
            state["final_response"] = "I'm here to help with your financial questions! You can ask me about your spending, budgets, banking products, or how you compare to other customers. What would you like to know?"
        else:
            state["final_response"] = f"I ran into a small hiccup while processing your request. No worries though! Try asking me about your spending patterns, budget management, banking products, or financial comparisons. I'm here to help!"

        state["execution_path"].append("error_handler")
        return state

    def _route_to_agents(self, state: MultiAgentState) -> str:
        """Enhanced routing to agents including RAG"""

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
        elif primary_agent == "rag":
            return "rag"
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
        """Enhanced chat interface with RAG agent support"""

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
            config = {"configurable": {"thread_id": session_id}}
            final_state = self.graph.invoke(initial_state, config=config)

            agent_used = "unknown"
            if final_state.get("routing_decision"):
                agent_used = final_state["routing_decision"].get("primary_agent", "unknown")

            message_count = 0
            if final_state.get("conversation_context"):
                message_count = final_state["conversation_context"].message_count

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
            print(f"❌ Enhanced router execution error: {e}")
            traceback.print_exc()
            
            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": "I encountered a system error. Please try again with a simpler question about your spending, budget, or banking services.",
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
    """Enhanced interactive chat demo"""
    
    print("=" * 60)
    print("💬 Enhanced Personal Finance Chat Demo - User ID: 430")
    print("🔧 Type 'quit' to exit")
    print("=" * 60)

    client_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv"
    overall_csv = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/overall_data.csv"

    try:
        # Use the enhanced router
        router = EnhancedPersonalFinanceRouter(
            client_csv_path=client_csv,
            overall_csv_path=overall_csv
        )

        client_id = 430
        session_id = f"enhanced_demo_session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        print(f"\n🎬 **ENHANCED DEMO MODE**")
        print(f"Session ID: {session_id}")
        print("-" * 40)

        # Test queries including RAG scenarios
        test_queries = [
            "What credit cards do you offer?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n👤 **Test {i}**: {query}")
            
            result = router.chat(
                client_id=client_id,
                user_query=query,
                session_id=session_id
            )

            print(f"🤖 **{result['agent_used'].upper()}**: {result['response']}")
            print(f"⚙️ *Success: {result['success']}*")
            
            if result.get('error'):
                print(f"❌ *Error: {result['error']}*")
            else:
                print("✅ *Test passed!*")

        print("\nNow you can continue the conversation...")
        print("-" * 40)

        # Interactive mode
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for using Enhanced Personal Finance Assistant!")
                    break
                elif not user_input:
                    continue

                result = router.chat(
                    client_id=client_id,
                    user_query=user_input,
                    session_id=session_id
                )

                agent_emoji = {
                    'spending': '📊',
                    'budget': '💰',
                    'rag': '🔍',
                    'error': '❌',
                    'unknown': '🤖'
                }.get(result['agent_used'], '🤖')
                
                agent_name = result['agent_used'].upper()
                
                print(f"\n{agent_emoji} **{agent_name}**: {result['response']}")
                
                if result.get('execution_path'):
                    path_display = ' → '.join(result['execution_path'])
                    print(f"🛤️ *Execution Path: {path_display}*")
                
                print(f"🎯 *Messages: {result.get('message_count', 0)} | Success: {result['success']}*")
                
                if result.get('error'):
                    print(f"\n⚠️ *Technical note: {result['error']}*")

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize enhanced router: {e}")


if __name__ == "__main__":
    interactive_chat_demo()