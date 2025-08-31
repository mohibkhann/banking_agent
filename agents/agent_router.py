import json
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
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


# NEW: Enhanced Follow-up Routing Model
class FollowUpRouting(BaseModel):
    """Enhanced routing decision for follow-up questions that can invoke other agents"""
    
    requires_agent_execution: bool = Field(
        description="True if follow-up needs to execute another agent, False for conversational response only"
    )
    target_agent: Optional[Literal["spending", "budget", "rag"]] = Field(
        default=None,
        description="Which agent to execute if requires_agent_execution is True"
    )
    enhanced_query: Optional[str] = Field(
        default=None,
        description="Enhanced query to send to target agent if agent execution is needed"
    )
    context_integration_needed: bool = Field(
        description="Whether previous conversation context needs to be integrated into the new query"
    )
    response_strategy: Literal["conversational", "agent_execution", "hybrid"] = Field(
        description="Strategy: 'conversational' for chat response, 'agent_execution' for agent call, 'hybrid' for both"
    )
    confidence: float = Field(
        description="Routing confidence", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Explanation of why this routing decision was made"
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
    follow_up_routing: Optional[Dict[str, Any]]  # NEW: For follow-up routing decisions
    primary_response: Optional[str]
    secondary_response: Optional[str]
    final_response: str
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    session_id: str
    enhanced_query: Optional[str]
    is_follow_up: Optional[bool]
    previous_sql_queries: Optional[List[str]]
    previous_agent_data: Optional[Dict[str, Any]]


class EnhancedPersonalFinanceRouter:
    """
    Enhanced router with smart follow-up capabilities that can invoke other agents
    """

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
    ):

        self.spending_agent = SpendingAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False  # We'll handle memory at router level
        )

        self.budget_agent = BudgetAgent(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path,
            model_name=model_name,
            memory=False  # We'll handle memory at router level
        )
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
        
        # NEW: Set up follow-up routing parser
        self.followup_parser = PydanticOutputParser(pydantic_object=FollowUpRouting)

        # Conversation contexts (in-memory cache)
        self.contexts: Dict[str, ConversationContext] = {}

        # Build enhanced router graph
        self.graph = self._build_router_graph()

    def _build_router_graph(self) -> StateGraph:
        """Build enhanced router with smart follow-up routing capabilities"""

        workflow = StateGraph(MultiAgentState)

        # Enhanced router workflow nodes
        workflow.add_node("context_manager", self._context_manager_node)
        workflow.add_node("query_router", self._query_router_node)
        workflow.add_node("spending_agent_node", self._spending_agent_node)
        workflow.add_node("budget_agent_node", self._budget_agent_node)
        workflow.add_node("rag_agent_node", self._rag_agent_node)
        workflow.add_node("follow_up_handler", self._follow_up_handler_node)  # Enhanced
        workflow.add_node("irrelevant_handler", self._irrelevant_handler_node)
        workflow.add_node("response_synthesizer", self._response_synthesizer_node)
        workflow.add_node("memory_updater", self._memory_updater_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point
        workflow.set_entry_point("context_manager")

        # Enhanced routing logic with smart follow-up routing
        workflow.add_edge("context_manager", "query_router")
        workflow.add_conditional_edges(
            "query_router",
            self._route_to_agents,
            {
                "spending": "spending_agent_node",
                "budget": "budget_agent_node",
                "rag": "rag_agent_node",
                "follow_up": "follow_up_handler",  
                "irrelevant": "irrelevant_handler",
                "error": "error_handler"
            }
        )

        workflow.add_edge("spending_agent_node", "response_synthesizer")
        workflow.add_edge("budget_agent_node", "response_synthesizer")
        workflow.add_edge("rag_agent_node", "response_synthesizer")
        workflow.add_edge("irrelevant_handler", "response_synthesizer")
        
        # NEW: Enhanced follow-up routing with agent execution capability
        workflow.add_conditional_edges(
            "follow_up_handler",
            self._route_from_follow_up,
            {
                "synthesize": "response_synthesizer",
                "irrelevant": "irrelevant_handler",
                "spending": "spending_agent_node",  # NEW: Can route to spending agent
                "budget": "budget_agent_node",     # NEW: Can route to budget agent  
                "rag": "rag_agent_node"            # NEW: Can route to RAG agent
            }
        )
        
        workflow.add_edge("response_synthesizer", "memory_updater")
        workflow.add_edge("memory_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=None)

    def _route_from_follow_up(self, state: MultiAgentState) -> str:
        """Enhanced routing from follow-up handler - can now route to agents"""
        follow_up_routing = state.get("follow_up_routing", {})
        
        # Check if follow-up determined we need agent execution
        if follow_up_routing.get("requires_agent_execution"):
            target_agent = follow_up_routing.get("target_agent")
            if target_agent in ["spending", "budget", "rag"]:
                print(f"[DEBUG] 🔀 Follow-up routing to {target_agent} agent")
                return target_agent
        
        # Check fallback routing for irrelevant
        routing_decision = state.get("routing_decision", {})
        if routing_decision.get("primary_agent") == "irrelevant":
            return "irrelevant"
        
        # Default to synthesize if we have a response
        if state.get("primary_response"):
            return "synthesize"
        
        return "irrelevant"

    def _context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced conversation context and memory management"""

        try:
            user_query = state["user_query"].strip()
            state["messages"].append(HumanMessage(content=user_query))

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
    
    def _build_comprehensive_followup_context(self, context, current_query: str) -> str:
        """Build comprehensive context for follow-up question handling"""
        
        context_parts = []
        
        # Current follow-up analysis
        context_parts.append(f"""**CURRENT FOLLOW-UP ANALYSIS:**
    - Follow-up question: "{current_query}"
    - Follow-up type: {'Methodology question' if any(word in current_query.lower() for word in ['how', 'calculate', 'method']) else 'Detail request' if any(word in current_query.lower() for word in ['which', 'what', 'when', 'where']) else 'Clarification request'}
    - Context dependency: HIGH (requires previous conversation context)""")

        # Complete conversation reconstruction
        if hasattr(context, 'conversation_history') and context.conversation_history:
            context_parts.append(f"""**COMPLETE CONVERSATION HISTORY:**""")
            
            for i, conv in enumerate(context.conversation_history, 1):
                if conv.get("user_query") and conv.get("agent_response"):
                    agent_used = conv.get("agent_used", "unknown")
                    success = conv.get("success", False)
                    timestamp = conv.get("timestamp", "unknown")
                    
                    context_parts.append(f"""
    **Interaction {i} [{agent_used.upper()} Agent] - {timestamp}:**
    User Query: "{conv['user_query']}"
    System Response: "{conv['agent_response']}"
    Success: {success}
    ---""")
        
        # Topic and insight analysis
        context_parts.append(f"""**FINANCIAL DISCUSSION ANALYSIS:**
    - Primary topics: {', '.join(context.recent_topics) if context.recent_topics else 'None'}
    - Key insights shared: {', '.join(getattr(context, 'key_insights', [])) if hasattr(context, 'key_insights') and context.key_insights else 'None'}
    - Last agent: {context.last_agent_used} (this agent handled the most recent financial analysis)
    - Conversation depth: {len(context.conversation_history) if hasattr(context, 'conversation_history') else 0} total interactions""")

        # Meta-conversation context
        context_parts.append(f"""**META-CONVERSATION CONTEXT:**
    - Session maturity: {'Advanced' if context.message_count > 3 else 'Intermediate' if context.message_count > 1 else 'New'}
    - Context richness: {'Very High' if len(getattr(context, 'conversation_history', [])) > 2 else 'High' if len(getattr(context, 'conversation_history', [])) > 1 else 'Medium'}
    - Financial engagement level: {'Deep discussion' if len(context.recent_topics) > 1 else 'Single topic focus' if context.recent_topics else 'Initial inquiry'}
    - Conversation summary: {getattr(context, 'conversation_summary', 'Active financial discussion in progress')}""")

        return "\n".join(context_parts)
    
    def _has_sufficient_follow_up_context(self, context, query: str) -> bool:
        """Determine if we have sufficient context to handle a follow-up question"""
        
        if not context:
            return False
        
        # Must have conversation history
        if not hasattr(context, 'conversation_history') or not context.conversation_history:
            return False
        
        # Must have recent financial topics
        if not context.recent_topics or not any(topic in ['spending', 'budget', 'banking'] for topic in context.recent_topics):
            return False
        
        # Must have used a financial agent recently
        if not context.last_agent_used or context.last_agent_used in ['irrelevant', None]:
            return False
        
        # Must have at least one successful previous interaction
        successful_interactions = [conv for conv in context.conversation_history if conv.get('success', False)]
        if not successful_interactions:
            return False
        
        return True

    def _query_router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Enhanced LLM-based routing with comprehensive context analysis"""

        try:
            context = state.get("conversation_context")
            enhanced_query = state.get("enhanced_query")
            original_query = state["user_query"]
            query_for_routing = enhanced_query if enhanced_query else original_query

            print(f"[DEBUG] Enhanced LLM router analyzing: {original_query}")

            # Build comprehensive context for LLM
            rich_context = self._build_comprehensive_context(context)

            # Enhanced routing prompt (same as before)
            routing_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are an elite financial conversation analyst and routing specialist for GX Bank's AI assistant system. Your expertise lies in understanding the nuanced context of financial conversations and routing queries with exceptional accuracy.

    **CONVERSATION INTELLIGENCE CONTEXT:**
    {rich_context}

    **YOUR ROUTING MISSION:**
    You must analyze the user's query with deep contextual understanding and route it to exactly ONE of these destinations:

    🎯 **ROUTING OPTIONS:**

    1. **follow_up** - Route here when:
    - User is asking clarifying questions about previous financial analysis ("which months?", "how calculated?", "what timeframe?")
    - Asking for more details about something already discussed ("show me more", "break that down", "explain that")
    - Making implicit references to previous conversation ("what about that?", "those numbers", "the calculation")
    - Questioning methodology or asking for specifics about previous results
    - Short queries with pronouns that clearly reference prior financial discussion
    - Any query that builds upon or seeks clarification of previous financial analysis

    2. **spending** - Route here for:
    - Personal transaction analysis and spending pattern queries
    - Questions about where money was spent, how much was spent
    - Category-based spending analysis, merchant analysis
    - Time-period spending analysis (monthly, quarterly, historical)
    - Spending comparisons and spending behavior analysis
    - Transaction-level inquiries and spending summaries

    3. **budget** - Route here for:
    - Budget creation, modification, or deletion requests
    - Budget performance tracking and budget vs actual analysis
    - Overspending alerts and budget optimization
    - Budget goal setting and budget planning
    - Questions about budget limits, budget categories
    - Budget management and financial planning

    4. **rag** - Route here for:
    - External banking product information (credit cards, loans, mortgages)
    - Bank policy questions, service offerings, rates and terms
    - Product comparison requests and eligibility questions
    - Banking service inquiries and product recommendations
    - Questions combining user's financial data with external banking products
    - Affordability analysis requiring both personal data and banking product info

    5. **irrelevant** - ONLY route here when:
    - Query is completely unrelated to personal finance, banking, or money management
    - General knowledge questions with no financial context
    - Topics like weather, sports, entertainment with no financial angle
    - Questions that have ZERO connection to banking, spending, budgeting, or financial services

    **CRITICAL ROUTING INTELLIGENCE:**

    🔍 **Context-First Analysis:**
    - If there's active financial conversation context, heavily weight that in your decision
    - Previous financial discussions make follow-up questions about methodology, timeframes, calculations, or details highly likely to be follow-ups
    - Questions like "which months", "how did you calculate", "what timeframe", "show me details" in financial context are almost certainly follow-ups

    🧠 **Semantic Understanding:**
    - Look beyond keywords to understand INTENT and CONTEXT
    - A question like "which months did you calculate?" after spending analysis is clearly a follow-up, not irrelevant
    - Pronouns ("it", "that", "this") often reference previous financial discussions
    - Temporal references ("then", "those", "when you said") indicate follow-up context

    ⚡ **High-Confidence Routing Rules:**
    - If previous conversation was financial AND current query asks for clarification/details → follow_up (99% confidence)
    - If query references previous analysis methodology → follow_up (99% confidence)
    - If query is asking "how/when/where/which" about something just discussed → follow_up (99% confidence)
    - If user says "what about X" where X was mentioned in recent financial conversation → follow_up (99% confidence)

    **RESPONSE FORMAT:**
    You must respond with exactly this format:

    ROUTE: [follow_up|spending|budget|rag|irrelevant]
    CONFIDENCE: [0.0-1.0]
    REASONING: [Detailed explanation of your routing decision with specific context references]
    CONTEXT_WEIGHT: [none|low|medium|high|critical]

    **ROUTING EXAMPLES FOR CALIBRATION:**

    Example 1:
    Previous: "Show me my spending for last 6 months"
    Current: "Which months did you calculate?"
    ROUTE: follow_up
    REASONING: Clear follow-up asking for methodology details about previous spending analysis

    Example 2:  
    Previous: "Create a budget for groceries"
    Current: "How much should I spend on restaurants?"
    ROUTE: budget
    REASONING: New budget-related question, not following up on groceries budget

    Example 3:
    Previous: "My night transaction analysis"
    Current: "What about day transactions?"
    ROUTE: spending
    REASONING: Related spending question but asking for new analysis, not clarification

    Example 4:
    Previous: Financial discussion about spending
    Current: "What's the weather like?"
    ROUTE: irrelevant
    REASONING: Completely unrelated to financial context despite previous conversation

    **ADVANCED CONTEXTUAL CUES:**
    - Interrogative words (which, when, where, how, what) + financial context = likely follow_up
    - Demonstrative pronouns (that, this, those) + financial context = likely follow_up
    - Requests for "more details", "breakdown", "specifics" + financial context = likely follow_up
    - Questions about "calculation", "methodology", "timeframe" + financial context = follow_up
    - Implicit references to previous numbers, dates, or analysis = follow_up"""),
                (
                    "human", 
                    f"""Analyze this query with your full contextual intelligence and route it appropriately:

    **CURRENT QUERY:** "{query_for_routing}"

    **ORIGINAL USER INPUT:** "{original_query}"

    Apply your deep contextual analysis and provide routing decision in the specified format."""
                )
            ])

            # Invoke the enhanced LLM routing
            response = self.llm.invoke(
                routing_prompt.format_messages(
                    query_for_routing=query_for_routing,
                    original_query=original_query,
                    rich_context=rich_context
                )
            )
            
            # Parse the structured response
            routing_dict = self._parse_enhanced_response(response.content, query_for_routing)
            
            state["routing_decision"] = routing_dict
            state["is_follow_up"] = (routing_dict["primary_agent"] == "follow_up")
            state["execution_path"].append("query_router")

            print(f"[DEBUG] Enhanced routing: {routing_dict['primary_agent']} (confidence: {routing_dict['confidence']:.2f})")
            print(f"[DEBUG] Context weight: {routing_dict.get('context_weight', 'unknown')}")
            print(f"[DEBUG] Reasoning: {routing_dict['reasoning'][:100]}...")

        except Exception as e:
            print(f"[DEBUG] Enhanced routing failed: {e}")
            # Simple emergency fallback
            state["routing_decision"] = {
                "is_relevant": True,
                "primary_agent": "spending",  # Safe default
                "query_category": "emergency_fallback",
                "confidence": 0.3,
                "reasoning": "Emergency fallback due to routing error"
            }
            state["is_follow_up"] = False

        return state
    
    def _build_comprehensive_context(self, context) -> str:
        """Build ultra-comprehensive context for LLM routing decisions"""
        
        if not context:
            return """**CONTEXT STATUS:** Fresh conversation start
    **CONVERSATION HISTORY:** None
    **TOPICS DISCUSSED:** None
    **LAST AGENT:** None
    **FINANCIAL CONTEXT:** None established
    **CONTEXT STRENGTH:** Zero - treat as completely new interaction"""

        context_parts = []
        
        # Conversation status and metrics
        context_parts.append(f"""**CONVERSATION STATUS:**
    - Message #{context.message_count} in ongoing financial conversation
    - Session duration: Active conversation in progress
    - Last interaction: {context.last_interaction.strftime('%H:%M')}
    - Conversation maturity: {'Established' if context.message_count > 3 else 'Early stage'}""")

        # Topic and agent context
        context_parts.append(f"""**RECENT FINANCIAL TOPICS:**
    - Primary topics discussed: {', '.join(context.recent_topics) if context.recent_topics else 'None yet'}
    - Last agent used: {context.last_agent_used or 'None (first interaction)'}
    - Topic depth: {'Deep discussion' if len(context.recent_topics) > 2 else 'Surface level' if context.recent_topics else 'No topics yet'}""")

        # Conversation flow analysis
        if hasattr(context, 'conversation_history') and context.conversation_history:
            context_parts.append(f"""**CONVERSATION FLOW ANALYSIS:**
    - Total interactions: {len(context.conversation_history)}
    - Conversation pattern: {'Multi-turn financial discussion' if len(context.conversation_history) > 1 else 'Single financial query'}""")
            
            # Recent conversation details
            context_parts.append(f"""**IMMEDIATE CONVERSATION HISTORY:**""")
            
            for i, conv in enumerate(context.conversation_history[-3:], 1):  # Last 3 interactions
                if conv.get("user_query") and conv.get("agent_response"):
                    user_q = conv["user_query"]
                    agent_resp = conv["agent_response"]
                    agent_used = conv.get("agent_used", "unknown")
                    
                    # Truncate for prompt efficiency but keep key information
                    user_display = user_q if len(user_q) <= 100 else f"{user_q[:100]}..."
                    agent_display = agent_resp if len(agent_resp) <= 200 else f"{agent_resp[:200]}..."
                    
                    context_parts.append(f"""
    **Interaction {i} (Agent: {agent_used}):**
    - User asked: "{user_display}"
    - System responded: "{agent_display}"
    - Success: {conv.get('success', 'Unknown')}""")
        
        # Key insights and data points
        if hasattr(context, 'key_insights') and context.key_insights:
            context_parts.append(f"""**KEY FINANCIAL DATA POINTS DISCUSSED:**
    - Insights mentioned: {', '.join(context.key_insights)}
    - Data context: Active financial analysis with specific numbers/amounts""")

        # Conversation summary and context strength
        context_strength = "CRITICAL" if context.message_count > 2 and context.recent_topics else \
                        "HIGH" if context.message_count > 1 and context.recent_topics else \
                        "MEDIUM" if context.recent_topics else "LOW"
        
        context_parts.append(f"""**CONTEXTUAL INTELLIGENCE ASSESSMENT:**
    - Context strength: {context_strength}
    - Financial discussion active: {'YES' if any(topic in ['spending', 'budget', 'banking'] for topic in (context.recent_topics or [])) else 'NO'}
    - Follow-up likelihood: {'VERY HIGH' if context_strength in ['CRITICAL', 'HIGH'] else 'MODERATE' if context_strength == 'MEDIUM' else 'LOW'}
    - Summary: {getattr(context, 'conversation_summary', 'No summary available')}""")

        return "\n".join(context_parts)
        
    def _parse_enhanced_response(self, response_text: str, query: str) -> Dict[str, Any]:
        """Parse the enhanced LLM routing response"""
        
        try:
            lines = response_text.strip().split('\n')
            
            # Extract components
            route = None
            confidence = 0.8
            reasoning = "LLM routing completed"
            context_weight = "medium"
            
            for line in lines:
                line = line.strip()
                if line.startswith("ROUTE:"):
                    route = line.split(":", 1)[1].strip().lower()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        confidence = 0.8
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("CONTEXT_WEIGHT:"):
                    context_weight = line.split(":", 1)[1].strip().lower()
            
            # Validate route
            valid_routes = ["follow_up", "spending", "budget", "rag", "irrelevant"]
            if route not in valid_routes:
                print(f"[DEBUG] Invalid route '{route}', defaulting to spending")
                route = "spending"
                confidence = 0.3
                reasoning = f"Invalid route detected, defaulted to spending. Original: {reasoning}"
            
            return {
                "is_relevant": route != "irrelevant",
                "primary_agent": route,
                "query_category": f"enhanced_{route}",
                "confidence": confidence,
                "reasoning": reasoning,
                "context_weight": context_weight
            }
            
        except Exception as e:
            print(f"[DEBUG] Enhanced parsing failed: {e}")
            return {
                "is_relevant": True,
                "primary_agent": "spending",
                "query_category": "parsing_fallback",
                "confidence": 0.3,
                "reasoning": f"Parsing failed, emergency fallback. Error: {str(e)}"
            }

    def _follow_up_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """ENHANCED: Follow-up handler with smart agent routing capabilities"""
        
        try:
            print("[DEBUG] 🔄 Executing ENHANCED follow-up handler with agent routing...")
            
            context = state.get("conversation_context")
            user_query = state["user_query"]
            
            # Check if we have sufficient context - if not, route to irrelevant
            if not self._has_sufficient_follow_up_context(context, user_query):
                print("[DEBUG] Insufficient context for follow-up, routing to irrelevant handler")
                
                state["routing_decision"] = {
                    "is_relevant": False,
                    "primary_agent": "irrelevant",
                    "query_category": "insufficient_follow_up_context",
                    "confidence": 0.95,
                    "reasoning": "Follow-up detected but insufficient conversation context - treating as out of scope"
                }
                state["execution_path"].append("follow_up_handler_context_insufficient")
                return state
            
            # NEW: Smart Follow-up Routing Analysis
            comprehensive_context = self._build_comprehensive_followup_context(context, user_query)
            
            # Enhanced follow-up routing prompt
            followup_routing_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an advanced follow-up routing specialist for GX Bank's AI assistant. Your job is to determine if a follow-up question requires executing another agent or can be handled conversationally.

**COMPREHENSIVE CONVERSATION CONTEXT:**
{comprehensive_context}

**YOUR MISSION:**
Analyze the follow-up question and determine the optimal handling strategy:

🎯 **ROUTING DECISION OPTIONS:**

1. **CONVERSATIONAL RESPONSE** - Handle with conversation only when:
   - User asks for clarification on previous response ("what did you mean by X?")
   - Asking about methodology/calculation details that can be explained ("how did you calculate that?")
   - Simple follow-up questions that reference previous data ("which months were those?")
   - Requesting more details about already-provided information

2. **AGENT EXECUTION** - Execute another agent when:
   - User requests NEW analysis based on previous discussion ("based on my spending, create a budget")
   - Cross-domain requests ("given those credit cards, which fits my spending pattern?")
   - Requests that need fresh data processing ("now show me budget vs that spending")
   - Complex queries combining previous context with new analysis needs

3. **HYBRID APPROACH** - Both conversational + agent execution when:
   - User wants explanation AND new analysis
   - Complex requests needing context integration

**AGENT SELECTION RULES:**
- **spending** agent: For transaction analysis, spending patterns, category breakdowns
- **budget** agent: For budget creation, budget analysis, overspending checks
- **rag** agent: For banking products, external services, policy questions

**CONTEXT INTEGRATION:**
- If agent execution needed, enhance the query with conversation context
- Build queries that reference previous findings naturally
- Ensure new analysis builds on previous discussion

**CRITICAL EXAMPLES:**

Follow-up: "Based on the credit cards you showed me, which one suits my spending pattern?"
→ requires_agent_execution: TRUE, target_agent: "rag", enhanced_query: "Based on my spending patterns, recommend the most suitable credit card from GX Bank's offerings"

Follow-up: "Now create a budget based on that spending analysis"
→ requires_agent_execution: TRUE, target_agent: "budget", enhanced_query: "Create a monthly budget based on my recent spending patterns and categories"

Follow-up: "What did you mean by overspending in restaurants?"
→ requires_agent_execution: FALSE, response_strategy: "conversational"

Follow-up: "Which months did you use for that calculation?"
→ requires_agent_execution: FALSE, response_strategy: "conversational"

**FORMAT INSTRUCTIONS:**
{format_instructions}

Analyze the follow-up question and provide structured routing decision:"""),
                ("human", f"""Current follow-up question: "{user_query}"

Based on the comprehensive conversation context, determine the optimal handling strategy for this follow-up question.""")
            ])
            
            formatted_followup_prompt = followup_routing_prompt.partial(
                format_instructions=self.followup_parser.get_format_instructions(),
                comprehensive_context=comprehensive_context
            )
            
            try:
                followup_chain = formatted_followup_prompt | self.llm | self.followup_parser
                followup_routing = followup_chain.invoke({"user_query": user_query})
                followup_routing_dict = followup_routing.model_dump()
                
                print(f"[DEBUG] 🎯 Follow-up routing: {followup_routing_dict['response_strategy']}")
                print(f"[DEBUG] 🔧 Requires agent execution: {followup_routing_dict['requires_agent_execution']}")
                if followup_routing_dict.get('target_agent'):
                    print(f"[DEBUG] 🎯 Target agent: {followup_routing_dict['target_agent']}")
                
                state["follow_up_routing"] = followup_routing_dict
                
                # If agent execution is required, set up enhanced query and routing
                if followup_routing_dict["requires_agent_execution"]:
                    target_agent = followup_routing_dict.get("target_agent")
                    enhanced_query = followup_routing_dict.get("enhanced_query")
                    
                    if target_agent and enhanced_query:
                        # Prepare for agent execution
                        state["enhanced_query"] = enhanced_query
                        state["user_query"] = enhanced_query  # Override with enhanced query
                        
                        # Update routing decision to point to target agent
                        state["routing_decision"] = {
                            "is_relevant": True,
                            "primary_agent": target_agent,
                            "query_category": f"follow_up_to_{target_agent}",
                            "confidence": 0.95,
                            "reasoning": f"Follow-up question requires {target_agent} agent execution with enhanced query"
                        }
                        
                        print(f"[DEBUG] ✅ Enhanced follow-up will execute {target_agent} agent with query: {enhanced_query[:60]}...")
                        state["execution_path"].append("follow_up_handler_agent_routing")
                        return state
                
                # Handle conversational response
                if followup_routing_dict["response_strategy"] in ["conversational", "hybrid"]:
                    # Generate conversational response using context
                    conversational_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""You are a professional GX Bank financial assistant with access to the complete conversation history. A customer is asking a follow-up question.

**COMPREHENSIVE CONVERSATION CONTEXT:**
{comprehensive_context}

**YOUR MISSION:**
Provide a detailed, specific, and helpful conversational answer to the customer's follow-up question using ALL available context from the previous conversation. You should:

- Answer with Specific Details: Use actual data, timeframes, calculations, or methodology from previous conversation
- Reference Previous Discussion: Naturally reference what was discussed before  
- Provide Complete Information: Don't be vague - give specific, actionable details
- Maintain Professional Tone: Sound like a knowledgeable GX Bank representative
- Build on Previous Analysis: Extend or clarify previous financial analysis
- Use Conversation Flow: Acknowledge the ongoing nature of the discussion

**RESPONSE STYLE:**
- Professional but conversational tone
- Specific details from previous conversation
- Direct answers to methodology, timeframe, or calculation questions
- Natural acknowledgment of conversation continuity
- Helpful and informative banking advisor voice

**CRITICAL INSTRUCTIONS:**
- Never say "I don't have access" - you have full context
- Never deflect - answer the specific follow-up question directly
- Use the rich context to provide detailed, specific answers
- Reference previous numbers, dates, analysis methods naturally"""),
                        ("human", f"""The customer's follow-up question is: "{user_query}"

Using the comprehensive conversation context provided, give a detailed and specific answer to their follow-up question.""")
                    ])
                    
                    response = self.llm.invoke(
                        conversational_prompt.format_messages(
                            user_query=user_query,
                            comprehensive_context=comprehensive_context
                        )
                    )
                    
                    state["primary_response"] = response.content
                    state["execution_path"].append("follow_up_handler_conversational")
                    
                    print("[DEBUG] ✅ Generated conversational follow-up response")
                    return state
                    
            except Exception as parse_error:
                print(f"[DEBUG] Follow-up routing parsing failed: {parse_error}")
                # Fallback to simple conversational response
                return self._fallback_conversational_response(state, comprehensive_context)
            
        except Exception as e:
            print(f"[DEBUG] Enhanced follow-up handler error: {e}")
            state["primary_response"] = "I encountered an issue processing your follow-up question. Could you please rephrase or ask a new question?"
            state["execution_path"].append("follow_up_handler_error")
        
        return state

    def _fallback_conversational_response(self, state: MultiAgentState, comprehensive_context: str) -> MultiAgentState:
        """Fallback conversational response when routing parsing fails"""
        
        try:
            user_query = state["user_query"]
            
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a GX Bank financial assistant handling a follow-up question. 

**CONVERSATION CONTEXT:**
{comprehensive_context}

Provide a helpful response to the customer's follow-up question based on the conversation context available."""),
                ("human", f"""Customer follow-up: "{user_query}"

Please provide a helpful response based on our previous conversation.""")
            ])
            
            response = self.llm.invoke(
                fallback_prompt.format_messages(
                    user_query=user_query,
                    comprehensive_context=comprehensive_context
                )
            )
            
            state["primary_response"] = response.content
            state["execution_path"].append("follow_up_handler_fallback_conversational")
            
            print("[DEBUG] ✅ Generated fallback conversational response")
            
        except Exception as e:
            print(f"[DEBUG] Fallback conversational response failed: {e}")
            state["primary_response"] = "I understand you're following up on our previous conversation. Could you please provide a bit more context so I can help you better?"
            state["execution_path"].append("follow_up_handler_fallback_error")
        
        return state

    def _spending_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute spending agent query (updated to handle enhanced queries from follow-ups)"""

        try:
            print("📊 [DEBUG] Executing spending agent query...")

            # Check for enhanced query from follow-up handler
            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process.")
                state["error"] = "No valid query provided to spending agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your spending again."
                return state
            
            # Log if we're processing an enhanced follow-up query
            if enhanced_query:
                print(f"[DEBUG] 🔄 Processing enhanced follow-up query: {enhanced_query[:60]}...")
            
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
        """Execute budget agent query (updated to handle enhanced queries from follow-ups)"""

        try:
            print("💰 [DEBUG] Executing budget agent query...")

            # Check for enhanced query from follow-up handler
            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process.")
                state["error"] = "No valid query provided to budget agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about your budget again."
                return state

            # Log if we're processing an enhanced follow-up query
            if enhanced_query:
                print(f"[DEBUG] 🔄 Processing enhanced follow-up query: {enhanced_query[:60]}...")

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
        """Execute RAG agent query (updated to handle enhanced queries from follow-ups)"""

        try:
            print("🔍 [DEBUG] Executing RAG agent query...")

            # Check for enhanced query from follow-up handler
            original_query = state.get("user_query")
            enhanced_query = state.get("enhanced_query")
            query_to_process = enhanced_query if enhanced_query else original_query
            
            if not query_to_process:
                print(f"[DEBUG] ❌ No valid query to process.")
                state["error"] = "No valid query provided to RAG agent"
                state["primary_response"] = "I couldn't process your query. Please try asking about banking products or services again."
                return state

            # Log if we're processing an enhanced follow-up query
            if enhanced_query:
                print(f"[DEBUG] 🔄 Processing enhanced follow-up query: {enhanced_query[:60]}...")

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
        """Enhanced response synthesis with follow-up awareness"""

        try:
            print("🔧 [DEBUG] Synthesizing final response...")

            primary_response = state.get("primary_response", "")
            context = state.get("conversation_context")
            routing = state.get("routing_decision", {})
            follow_up_routing = state.get("follow_up_routing", {})

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
                # For spending/budget agents and follow-ups, add contextual enhancements
                prefix = ""
                
                # Enhanced follow-up context integration
                if follow_up_routing and follow_up_routing.get("requires_agent_execution"):
                    target_agent = follow_up_routing.get("target_agent", agent_used)
                    prefix = f"Building on our previous discussion, here's your {target_agent} analysis... "
                elif context and context.message_count > 1:
                    if context.last_agent_used and context.last_agent_used != agent_used:
                        prefix = f"Switching to {agent_used} analysis... "
                
                state["final_response"] = prefix + primary_response
                
                # Record the model's reply
                if state.get("final_response"):
                    state["messages"].append(AIMessage(content=state["final_response"]))
                
                # Add intelligent cross-agent suggestions (enhanced for follow-ups)
                recent_topics = context.recent_topics if context else []
                suggestion = ""
                
                # Smart suggestions based on current and previous context
                if agent_used == "spending" and "banking" in recent_topics:
                    suggestion = "\n\n💡 *Based on this spending analysis, would you like me to recommend suitable banking products?*"
                elif agent_used == "budget" and "banking" in recent_topics:
                    suggestion = "\n\n💡 *I can help you find banking products that align with this budget.*"
                elif agent_used == "spending" and "budget" not in recent_topics:
                    suggestion = "\n\n💡 *Would you like me to create a budget based on this spending analysis?*"
                elif agent_used == "budget" and context and context.message_count <= 2:
                    suggestion = "\n\n🎯 *I can also analyze your historical spending patterns or help you explore banking products.*"
                elif agent_used == "rag" and any(topic in recent_topics for topic in ["spending", "budget"]):
                    suggestion = "\n\n💡 *Would you like me to analyze how this product fits with your spending patterns or budget?*"
                
                state["final_response"] += suggestion

            state["execution_path"].append("response_synthesizer")
            print("✅ Response synthesis complete")

        except Exception as e:
            print(f"❌ Response synthesis error: {e}")
            state["final_response"] = state.get("primary_response", "I apologize, but I encountered an issue processing your request. Please try asking about your spending, budget, or banking services.")

        return state

    def _memory_updater_node(self, state: MultiAgentState) -> MultiAgentState:
        """Update conversation memory with enhanced follow-up tracking"""

        try:
            print("💾 [DEBUG] Updating conversation memory...")

            context = state.get("conversation_context")
            if context:
                # Determine the actual query used (might be enhanced from follow-up)
                query_used = state.get("enhanced_query") if state.get("enhanced_query") else context.last_user_query
                
                # Safely get routing decision
                routing_decision = state.get("routing_decision", {})
                agent_used = routing_decision.get("primary_agent") if routing_decision else "unknown"
                
                # Enhanced interaction record with follow-up metadata
                current_interaction = {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": context.last_user_query,  # Original user query
                    "enhanced_query": state.get("enhanced_query"),  # Enhanced query if follow-up
                    "agent_response": state.get("final_response"),
                    "agent_used": agent_used,
                    "was_follow_up": state.get("is_follow_up", False),
                    "follow_up_routing": state.get("follow_up_routing"),
                    "success": state.get("error") is None
                }
                
                context.conversation_history.append(current_interaction)
                context.last_agent_response = state.get("final_response")
                context.conversation_history = context.conversation_history[-10:]
                
                # Enhanced conversation summary with follow-up awareness
                if len(context.conversation_history) >= 2:
                    recent_queries = [h.get("user_query", "") for h in context.conversation_history[-3:] if h.get("user_query")]
                    follow_up_count = sum(1 for h in context.conversation_history if h.get("was_follow_up", False))
                    
                    context.conversation_summary = f"Recent topics: {', '.join(context.recent_topics)}. Last queries: {'; '.join(recent_queries[-2:])}. Follow-ups: {follow_up_count}"
                
                # Extract insights with enhanced follow-up context
                final_response = state.get("final_response")
                if final_response:
                    if "$" in final_response and any(word in final_response.lower() for word in ["spent", "budget", "total"]):
                        import re
                        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', final_response)
                        if amounts:
                            insight = f"Recent discussion: {amounts[0]} mentioned"
                            if insight not in context.key_insights:
                                context.key_insights.append(insight)
                    
                    # Track cross-agent patterns
                    follow_up_routing = state.get("follow_up_routing")
                    if follow_up_routing and follow_up_routing.get("requires_agent_execution"):
                        insight = f"Follow-up led to {agent_used} analysis"
                        if insight not in context.key_insights:
                            context.key_insights.append(insight)

                context.key_insights = context.key_insights[-5:]

                print(f"[DEBUG] 💾 Stored interaction: {context.last_user_query[:50] if context.last_user_query else 'Unknown'}... -> {len(final_response) if final_response else 0} chars")
                print(f"[DEBUG] 📚 Total conversation history: {len(context.conversation_history)} interactions")
                
                # Log follow-up pattern if detected
                if state.get("is_follow_up"):
                    follow_up_routing = state.get("follow_up_routing", {})
                    if follow_up_routing and follow_up_routing.get("requires_agent_execution"):
                        print(f"[DEBUG] 🔄 Follow-up pattern: {follow_up_routing.get('target_agent')} agent execution")

            state["execution_path"].append("memory_updater")
            print("✅ Memory updated with enhanced follow-up tracking")

        except Exception as e:
            print(f"❌ Memory update error: {e}")
            # Log the error but don't fail the entire process
            import traceback
            traceback.print_exc()

        return state

    def _irrelevant_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle irrelevant queries with natural responses"""
        
        try:
            print("🚫 [DEBUG] Handling irrelevant query...")
            
            user_query = state["user_query"]
            context = state.get("conversation_context")
            
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly personal banking assistant of GX Bank. The user has maybe asked a question that's outside your expertise area of personal finance, spending, and budgeting.

Your response should be:
1. Warm and understanding
2. Briefly acknowledge their question
3. Naturally redirect to what you can help with
4. Offer specific examples relevant to banking/finance
5. If the user says Hello, greet them also with referring them to as GX Member
6. If the user says who are you reply with I am your GX Bank Assistant

Keep it polite and professional - like a bank employee redirecting the conversation.

AVOID:
- Technical explanations about your limitations
- Formal language like "I'm designed to" or "My domain is"
- Long lists of capabilities
- Apologetic tone

BE NATURAL AND HELPFUL:
- "That's outside my area, but I'm here to help with your finances!"
- "I focus on helping with your money matters"
- Give 2-3 specific examples of what you can do"""),
                ("human", """The user asked: "{query}"

Provide a brief, natural response that redirects them to finance topics.""")
            ])
            
            response = self.llm.invoke(
                response_prompt.format_messages(query=user_query)
            )
            
            base_response = response.content
            
            # Enhanced suggestions with follow-up context awareness
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

        raw_err = state.get("error")
        error_message = (raw_err if isinstance(raw_err, str) and raw_err.strip()
                        else str(raw_err) if raw_err is not None
                        else "Unknown routing error")
        error_lc = error_message.lower()
        print(f"🔧 [DEBUG] Handling router error: {error_message}")
        
        user_query = state["user_query"]

        if "Invalid query" in error_lc:
            state["final_response"] = "I'd be happy to help! Could you tell me a bit more about what you'd like to know about your finances? I can help with spending analysis, budgeting, banking products, or comparing your habits to others."
        elif "Intent classification" in error_lc or "routing" in error_lc.lower():
            state["final_response"] = "I want to make sure I understand what you're looking for. Are you interested in seeing your spending patterns, setting up a budget, exploring banking products, or something else financial? Just let me know!"
        elif "No valid query" in error_lc:
            state["final_response"] = "I'm here to help with your financial questions! You can ask me about your spending, budgets, banking products, or how you compare to other customers. What would you like to know?"
        else:
            state["final_response"] = f"I ran into a small hiccup while processing your request. No worries though! Try asking me about your spending patterns, budget management, banking products, or financial comparisons. I'm here to help!"

        state["execution_path"].append("error_handler")
        return state
    

    def _route_to_agents(self, state: MultiAgentState) -> str:
        """Enhanced routing to agents including RAG and follow_up"""

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
        elif primary_agent == "follow_up":
            return "follow_up"
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
        """Enhanced chat interface with smart follow-up capabilities"""

        if not session_id:
            session_id = f"session_{client_id}_{datetime.now().strftime('%Y%m%d_%H')}"

        initial_state = MultiAgentState(
            client_id=client_id,
            user_query=user_query,
            conversation_context=None,
            routing_decision=None,
            follow_up_routing=None,  # NEW: Enhanced follow-up routing
            primary_response=None,
            secondary_response=None,
            final_response="",
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[],
            session_id=session_id,
            enhanced_query=None,
            is_follow_up=None,           
            previous_sql_queries=None,   
            previous_agent_data=None     
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

            # Enhanced response with follow-up metadata
            follow_up_info = {}
            if final_state.get("follow_up_routing"):
                follow_up_routing = final_state["follow_up_routing"]
                follow_up_info = {
                    "was_follow_up": final_state.get("is_follow_up", False),
                    "required_agent_execution": follow_up_routing.get("requires_agent_execution", False),
                    "target_agent": follow_up_routing.get("target_agent"),
                    "response_strategy": follow_up_routing.get("response_strategy"),
                    "enhanced_query": final_state.get("enhanced_query")
                }

            return {
                "client_id": client_id,
                "session_id": session_id,
                "query": user_query,
                "response": final_state.get("final_response", "No response generated"),
                "agent_used": agent_used,
                "execution_path": final_state.get("execution_path", []),
                "message_count": message_count,
                "follow_up_info": follow_up_info,  # NEW: Enhanced follow-up information
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
        """Get conversation summary for a session with enhanced follow-up insights"""
        
        if session_id in self.contexts:
            context = self.contexts[session_id]
            
            # Enhanced analytics
            follow_up_count = 0
            agent_executions_from_followups = 0
            
            if hasattr(context, 'conversation_history'):
                follow_up_count = sum(1 for h in context.conversation_history if h.get("was_follow_up", False))
                agent_executions_from_followups = sum(1 for h in context.conversation_history 
                                                    if h.get("was_follow_up", False) and h.get("follow_up_routing", {}).get("requires_agent_execution", False))
            
            return {
                "session_id": session_id,
                "client_id": context.client_id,
                "session_start": context.session_start.isoformat(),
                "message_count": context.message_count,
                "recent_topics": context.recent_topics,
                "last_agent_used": context.last_agent_used,
                "key_insights": context.key_insights,
                "conversation_summary": context.conversation_summary,
                "enhanced_analytics": {  # NEW: Enhanced conversation analytics
                    "follow_up_questions": follow_up_count,
                    "agent_executions_from_followups": agent_executions_from_followups,
                    "conversation_depth": "Deep" if context.message_count > 5 else "Medium" if context.message_count > 2 else "Surface",
                    "cross_agent_pattern": len(set([h.get("agent_used") for h in getattr(context, 'conversation_history', []) if h.get("agent_used")])) > 1
                }
            }
        else:
            return {"error": "Session not found"}


def interactive_chat_demo():
    """Enhanced interactive chat demo showcasing smart follow-up capabilities"""
    
    print("=" * 60)
    print("💬 ENHANCED Personal Finance Chat Demo with Smart Follow-ups")
    print("🔧 User ID: 430 | Now supports cross-agent follow-up questions!")
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

        # Enhanced test queries showcasing follow-up capabilities
        test_queries = [
            "What are my spending patterns?",
            "Based on that spending, create a budget for me",  # This should be a follow-up that executes budget agent
            "Which categories were highest in my spending?",   # This should be conversational follow-up
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n👤 **Test {i}**: {query}")
            
            result = router.chat(
                client_id=client_id,
                user_query=query,
                session_id=session_id
            )

            print(f"**{result['agent_used'].upper()}**: {result['response'][:200]}...")
            print(f"*Success: {result['success']}*")
            
            # Enhanced follow-up information
            follow_up_info = result.get('follow_up_info', {})
            if follow_up_info.get('was_follow_up'):
                print(f"🔄 **Follow-up detected!**")
                print(f"   - Agent execution required: {follow_up_info.get('required_agent_execution')}")
                if follow_up_info.get('target_agent'):
                    print(f"   - Target agent: {follow_up_info['target_agent']}")
                if follow_up_info.get('enhanced_query'):
                    print(f"   - Enhanced query: {follow_up_info['enhanced_query'][:60]}...")
            
            if result.get('error'):
                print(f"❌ *Error: {result['error']}*")
            else:
                print("✅ *Test passed!*")

        print("\nNow you can continue the conversation and test follow-up capabilities...")
        print("Try questions like:")
        print("• 'Based on my spending, what credit card suits me?'")
        print("• 'Now create a budget from that analysis'")
        print("• 'Which months did you use for that calculation?'")
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
                
                # Enhanced follow-up information display
                follow_up_info = result.get('follow_up_info', {})
                if follow_up_info.get('was_follow_up'):
                    print(f"\n🔄 **Follow-up Pattern Detected:**")
                    print(f"   • Strategy: {follow_up_info.get('response_strategy', 'unknown')}")
                    if follow_up_info.get('required_agent_execution'):
                        print(f"   • Executed {follow_up_info.get('target_agent', 'unknown')} agent")
                        if follow_up_info.get('enhanced_query'):
                            print(f"   • Enhanced query: {follow_up_info['enhanced_query'][:50]}...")
                    else:
                        print(f"   • Provided conversational response")
                
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

        # Show enhanced conversation summary
        print("\n" + "=" * 60)
        print("📊 **ENHANCED CONVERSATION SUMMARY**")
        summary = router.get_conversation_summary(session_id)
        if summary.get('enhanced_analytics'):
            analytics = summary['enhanced_analytics']
            print(f"• Total messages: {summary.get('message_count', 0)}")
            print(f"• Follow-up questions: {analytics.get('follow_up_questions', 0)}")
            print(f"• Agent executions from follow-ups: {analytics.get('agent_executions_from_followups', 0)}")
            print(f"• Conversation depth: {analytics.get('conversation_depth', 'Unknown')}")
            print(f"• Cross-agent patterns: {'Yes' if analytics.get('cross_agent_pattern') else 'No'}")
            print(f"• Topics covered: {', '.join(summary.get('recent_topics', []))}")

    except Exception as e:
        print(f"❌ Failed to initialize enhanced router: {e}")


if __name__ == "__main__":
    interactive_chat_demo()