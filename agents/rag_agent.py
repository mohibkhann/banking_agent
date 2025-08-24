import json
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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

# Import our new tools and existing infrastructure
from tools.tavily_tools import (
    search_banking_products,
    search_bank_policies_and_services,
    BankingProductQuery,
    TavilySearchRequest
)
from tools.semantic_cache import (
    semantic_cache_search,
    semantic_cache_add,
    CacheQuery
)
from tools.brand_translator import (
    translate_bank_content,
    detect_content_type,
    BrandTranslationRequest
)

# Import existing data store for potential collaboration
from data_store.data_store import DataStore

load_dotenv()


# Enhanced Pydantic Models for RAG Agent with Real Collaboration
class RAGIntentClassification(BaseModel):
    """Structured intent classification for RAG queries"""
    
    query_type: Literal["banking_product", "policy_service", "hybrid", "collaboration_needed"] = Field(
        description="Type of RAG query to handle"
    )
    product_focus: Optional[str] = Field(
        default=None,
        description="Specific product type: credit_card, loan, investment, account"
    )
    requires_external_data: bool = Field(
        description="Whether external banking data is needed"
    )
    collaboration_agents: List[str] = Field(
        default=[],
        description="Other agents needed: ['spending', 'budget'] or []"
    )
    user_context_needed: bool = Field(
        description="Whether user's personal financial data is relevant"
    )
    search_strategy: str = Field(
        description="Search approach: focused, broad, comparative"
    )
    confidence: float = Field(
        description="Classification confidence", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of classification"
    )


class CollaborationRequest(BaseModel):
    """Structured request for agent collaboration"""
    
    target_agent: str = Field(description="Agent to collaborate with")
    user_query: str = Field(description="Original user query")
    context_query: str = Field(description="Specific query for the target agent")
    expected_data_type: str = Field(description="Type of data expected from collaboration")


class RAGAgentState(TypedDict):
    """State for RAG Agent workflow"""
    
    client_id: int
    user_query: str
    intent: Optional[Dict[str, Any]]
    cache_result: Optional[Dict[str, Any]]
    external_data: Optional[List[Dict[str, Any]]]
    collaborative_data: Dict[str, Any]
    raw_content: Optional[List[Dict[str, Any]]]
    translated_content: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    analysis_type: Optional[str]


class RAGAgent:
    """
    Enhanced RAG Agent with real collaboration capabilities.
    Can actually invoke spending and budget agents when needed.
    """

    def __init__(
        self,
        client_csv_path: Optional[str] = None,
        overall_csv_path: Optional[str] = None,
        model_name: str = "gpt-4o",
        memory: bool = True,
        # NEW: Accept agent instances for collaboration
        spending_agent: Optional[Any] = None,
        budget_agent: Optional[Any] = None,
    ):
        print(f"🔍 Initializing Enhanced RAG Agent with real collaboration...")
        
        # Initialize DataStore only if CSV paths provided (for collaboration)
        if client_csv_path and overall_csv_path:
            print(f"📋 Client data: {client_csv_path}")
            print(f"📊 Overall data: {overall_csv_path}")
            self.data_store = DataStore(
                client_csv_path=client_csv_path, 
                overall_csv_path=overall_csv_path
            )
        else:
            print("📋 Running in external-only mode (no collaboration)")
            self.data_store = None

        # Store agent references for real collaboration
        self.spending_agent = spending_agent
        self.budget_agent = budget_agent

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Set up structured output parser
        self.intent_parser = PydanticOutputParser(
            pydantic_object=RAGIntentClassification
        )

        # Setup memory
        self.memory = MemorySaver() if memory else None

        # Build the enhanced RAG workflow graph
        self.graph = self._build_graph()
        print("✅ Enhanced RAG Agent initialized with real collaboration!")

    def _build_graph(self) -> StateGraph:
        """Build the RAG Agent LangGraph workflow with real collaboration"""

        workflow = StateGraph(RAGAgentState)

        # RAG-specific workflow nodes
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("cache_checker", self._cache_checker_node)
        workflow.add_node("external_data_fetcher", self._external_data_fetcher_node)
        workflow.add_node("collaboration_coordinator", self._enhanced_collaboration_coordinator_node)  # Enhanced
        workflow.add_node("brand_translator", self._brand_translator_node)
        workflow.add_node("response_synthesizer", self._enhanced_response_synthesizer_node)  # Enhanced
        workflow.add_node("cache_updater", self._cache_updater_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point
        workflow.set_entry_point("intent_classifier")

        # Routing logic
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {
                "check_cache": "cache_checker",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "cache_checker",
            self._route_after_cache,
            {
                "fetch_external": "external_data_fetcher",
                "collaborate": "collaboration_coordinator",
                "translate": "brand_translator"
            }
        )

        workflow.add_conditional_edges(
            "external_data_fetcher",
            self._route_after_external_data,
            {
                "collaborate": "collaboration_coordinator",
                "translate": "brand_translator"
            }
        )

        workflow.add_edge("collaboration_coordinator", "brand_translator")
        workflow.add_edge("brand_translator", "response_synthesizer")
        workflow.add_edge("response_synthesizer", "cache_updater")
        workflow.add_edge("cache_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)

    def _intent_classifier_node(self, state: RAGAgentState) -> RAGAgentState:
        """Enhanced intent classification for RAG queries"""

        try:
            print(f"🔍 [DEBUG] Classifying RAG intent for: {state['user_query']}")

            classification_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an AI assistant that classifies user queries for a banking RAG (Retrieval-Augmented Generation) system.

Analyze the user's query and determine:

1. **Query Type:**
- "banking_product": Questions about credit cards, loans, accounts, investments
- "policy_service": Questions about bank policies, services, fees, procedures
- "hybrid": Questions combining products with personal context
- "collaboration_needed": Complex queries requiring personal financial data analysis

2. **Product Focus:** (if applicable)
- credit_card, debit_card, home_loan, auto_loan, investment, account, insurance

3. **Data Requirements:**
- requires_external_data: true if need Bank of America product/policy information
- user_context_needed: true if user's personal financial data is relevant
- collaboration_agents: ["spending"] or ["budget"] or ["spending", "budget"] or []

4. **Search Strategy:**
- "focused": Specific product/policy search
- "broad": General information gathering
- "comparative": Comparison shopping or analysis

**EXAMPLES:**

"What credit cards do you offer?" 
→ banking_product, credit_card, external_data=true, user_context=false, focused

"Based on my spending, which credit card suits me?"
→ hybrid, credit_card, external_data=true, user_context=true, collaboration=["spending"], comparative

"Can I afford a Honda City based on my budget and your loan rates?"
→ collaboration_needed, auto_loan, external_data=true, user_context=true, collaboration=["spending", "budget"], comparative

"Based on my spendings how can i afford ford escape let me know with the car loan policies"
→ collaboration_needed, auto_loan, external_data=true, user_context=true, collaboration=["spending", "budget"], comparative

{format_instructions}

Analyze this query and provide structured classification:"""),
                ("human", "{user_query}"),
            ])

            # Format the prompt with parser instructions
            formatted_prompt = classification_prompt.partial(
                format_instructions=self.intent_parser.get_format_instructions()
            )

            # Create chain with structured output
            classification_chain = formatted_prompt | self.llm | self.intent_parser

            try:
                intent_result = classification_chain.invoke(
                    {"user_query": state["user_query"]}
                )

                # Convert Pydantic model to dict
                intent_dict = intent_result.model_dump()

                print(f"[DEBUG] RAG intent classified as: {intent_dict['query_type']}")
                print(f"[DEBUG] Strategy: {intent_dict['search_strategy']}, Confidence: {intent_dict['confidence']}")
                print(f"[DEBUG] Collaboration needed: {intent_dict['collaboration_agents']}")

                state["intent"] = intent_dict
                state["analysis_type"] = intent_dict["query_type"]
                state["execution_path"].append("intent_classifier")

                state["messages"].append(
                    AIMessage(
                        content=f"RAG query classified as {intent_dict['query_type']} with {intent_dict['search_strategy']} strategy. Processing..."
                    )
                )

            except Exception as parse_error:
                print(f"[DEBUG] Structured parsing failed, using fallback: {parse_error}")

                # Fallback classification
                fallback_result = self._fallback_rag_classification(state["user_query"])
                state["intent"] = fallback_result
                state["analysis_type"] = fallback_result["query_type"]
                state["execution_path"].append("intent_classifier")

        except Exception as e:
            print(f"[DEBUG] RAG intent classification failed: {e}")
            state["error"] = f"RAG intent classification error: {str(e)}"

        return state

    def _fallback_rag_classification(self, user_query: str) -> Dict[str, Any]:
        """Enhanced fallback RAG intent classification using keywords"""

        query_lower = user_query.lower()

        # Enhanced keyword detection
        product_keywords = {
            "credit_card": ["credit card", "card", "cashback", "rewards"],
            "home_loan": ["mortgage", "home loan", "house", "property"],
            "auto_loan": ["car loan", "auto loan", "vehicle", "honda", "toyota", "ford", "escape"],
            "investment": ["invest", "portfolio", "stocks", "mutual fund"],
            "account": ["checking", "savings", "account"]
        }

        # Enhanced collaboration indicators
        personal_indicators = ["my", "i spend", "based on my", "my budget", "afford", "my spending", "my spendings"]
        policy_indicators = ["policy", "fee", "rate", "service", "offer", "policies"]

        # Determine product focus
        product_focus = None
        for product, keywords in product_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                product_focus = product
                break

        # Enhanced collaboration detection
        has_personal_context = any(indicator in query_lower for indicator in personal_indicators)
        has_policy_focus = any(indicator in query_lower for indicator in policy_indicators)
        
        # Specific patterns that definitely need collaboration
        needs_spending_analysis = any(phrase in query_lower for phrase in 
                                    ["based on my spending", "my spending", "afford", "how much did i spend"])
        needs_budget_analysis = any(phrase in query_lower for phrase in 
                                  ["my budget", "budget", "afford"])

        # Determine collaboration agents needed
        collaboration_agents = []
        if needs_spending_analysis:
            collaboration_agents.append("spending")
        if needs_budget_analysis:
            collaboration_agents.append("budget")

        # Determine query type
        if has_personal_context and product_focus and collaboration_agents:
            query_type = "collaboration_needed"
        elif has_personal_context and product_focus:
            query_type = "hybrid"
            collaboration_agents = ["spending"] if "spend" in query_lower else []
        elif product_focus:
            query_type = "banking_product"
        else:
            query_type = "policy_service"

        return {
            "query_type": query_type,
            "product_focus": product_focus,
            "requires_external_data": True,
            "collaboration_agents": collaboration_agents,
            "user_context_needed": has_personal_context,
            "search_strategy": "comparative" if "vs" in query_lower or "compare" in query_lower else "focused",
            "confidence": 0.8,  # Higher confidence for enhanced fallback
            "reasoning": "Enhanced fallback keyword-based classification with collaboration detection"
        }

    def _cache_checker_node(self, state: RAGAgentState) -> RAGAgentState:
        """Check semantic cache for similar queries"""

        try:
            print("💾 [DEBUG] Checking semantic cache...")

            intent = state.get("intent", {})
            
            cache_result = semantic_cache_search.invoke({
                "query_text": state["user_query"],
                "intent_type": intent.get("query_type"),
                "product_type": intent.get("product_focus"),
                "similarity_threshold": 0.87  # Slightly higher threshold for RAG
            })

            state["cache_result"] = cache_result
            state["execution_path"].append("cache_checker")

            if cache_result.get("cache_hit"):
                print(f"[DEBUG] ✅ Cache HIT: similarity={cache_result['similarity_score']:.3f}")
                # Use cached data directly
                cached_data = cache_result.get("cached_data", {})
                state["external_data"] = cached_data.get("search_results", [])
                state["raw_content"] = cached_data.get("external_data", {}).get("raw_content", [])
                state["collaborative_data"] = cached_data.get("external_data", {}).get("collaborative_data", {})
            else:
                print(f"[DEBUG] ❌ Cache MISS: best_similarity={cache_result['similarity_score']:.3f}")

        except Exception as e:
            print(f"[DEBUG] Cache check failed: {e}")
            state["cache_result"] = {"cache_hit": False, "error": str(e)}

        return state

    def _external_data_fetcher_node(self, state: RAGAgentState) -> RAGAgentState:
        """Fetch external banking data using Tavily"""

        try:
            print("🌐 [DEBUG] Fetching external banking data...")

            intent = state.get("intent", {})
            cache_result = state.get("cache_result", {})

            # Skip if we have cached data
            if cache_result.get("cache_hit"):
                print("[DEBUG] Using cached external data")
                state["execution_path"].append("external_data_fetcher_cached")
                return state

            external_data = []

            # Determine search approach based on intent
            if intent.get("query_type") == "banking_product" and intent.get("product_focus"):
                # Product-specific search
                print(f"[DEBUG] Searching for banking products: {intent['product_focus']}")
                
                product_result = search_banking_products.invoke({
                    "user_query": state["user_query"],
                    "product_type": intent["product_focus"],
                    "user_criteria": None,  # Will be filled by collaboration if needed
                    "max_results": 5
                })

                if not product_result.get("error"):
                    # Convert TavilySearchResult objects to dictionaries
                    results = product_result.get("results", [])
                    for result in results:
                        if hasattr(result, 'title'):  # It's a Pydantic model
                            external_data.append({
                                "title": result.title,
                                "content": result.content,
                                "url": result.url,
                                "score": result.score
                            })
                        else:  # It's already a dictionary
                            external_data.append(result)

            elif intent.get("query_type") in ["policy_service", "hybrid", "collaboration_needed"]:
                # Policy/service search
                print("[DEBUG] Searching for bank policies and services")
                
                focus_area = "rates" if "rate" in state["user_query"].lower() else "general"
                
                policy_result = search_bank_policies_and_services.invoke({
                    "user_query": state["user_query"],
                    "focus_area": focus_area,
                    "include_rates": True
                })

                if not policy_result.get("error"):
                    # Convert TavilySearchResult objects to dictionaries
                    results = policy_result.get("results", [])
                    for result in results:
                        if hasattr(result, 'title'):  # It's a Pydantic model
                            external_data.append({
                                "title": result.title,
                                "content": result.content,
                                "url": result.url,
                                "score": getattr(result, 'score', 0.0)
                            })
                        else:  # It's already a dictionary
                            external_data.append(result)

            # Store raw content for brand translation (now all dictionaries)
            state["external_data"] = external_data
            state["raw_content"] = [
                {
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content_type": detect_content_type(result.get("content", ""))
                }
                for result in external_data
            ]

            state["execution_path"].append("external_data_fetcher")
            print(f"[DEBUG] ✅ Fetched {len(external_data)} external data sources")

        except Exception as e:
            print(f"[DEBUG] External data fetching failed: {e}")
            state["error"] = f"External data fetch error: {e}"

        return state

    def _enhanced_collaboration_coordinator_node(self, state: RAGAgentState) -> RAGAgentState:
        """ENHANCED: Real collaboration with other agents"""

        try:
            print("🤝 [DEBUG] Coordinating with other agents...")

            intent = state.get("intent", {})
            collaboration_agents = intent.get("collaboration_agents", [])

            # FIXED: Always initialize collaborative_data as empty dict
            collaborative_data = {}

            if not collaboration_agents:
                print("[DEBUG] No collaboration needed")
                # FIXED: Still set empty dict instead of leaving as None
                state["collaborative_data"] = collaborative_data
                state["execution_path"].append("collaboration_coordinator_skip")
                return state

            # REAL COLLABORATION: Actually invoke other agents
            client_id = state["client_id"]
            user_query = state["user_query"]

            for agent_name in collaboration_agents:
                try:
                    if agent_name == "spending" and self.spending_agent:
                        print(f"[DEBUG] 📊 Collaborating with SPENDING agent...")
                        
                        # Generate spending-specific query from user query
                        spending_query = self._generate_spending_query(user_query)
                        
                        # Actually invoke the spending agent
                        spending_result = self.spending_agent.process_query(
                            client_id=client_id,
                            user_query=spending_query
                        )
                        
                        if spending_result.get("success"):
                            collaborative_data["spending_analysis"] = {
                                "query": spending_query,
                                "response": spending_result.get("response"),
                                "analysis_type": spending_result.get("analysis_type"),
                                "raw_data": spending_result.get("raw_data", []),
                                "sql_executed": spending_result.get("sql_executed", [])
                            }
                            print(f"[DEBUG] ✅ Spending collaboration successful")
                        else:
                            print(f"[DEBUG] ❌ Spending collaboration failed: {spending_result.get('error')}")
                            collaborative_data["spending_analysis"] = {
                                "error": spending_result.get("error"),
                                "fallback_note": "Could not retrieve spending data"
                            }

                    elif agent_name == "budget" and self.budget_agent:
                        print(f"[DEBUG] 💰 Collaborating with BUDGET agent...")
                        
                        # Generate budget-specific query from user query
                        budget_query = self._generate_budget_query(user_query)
                        
                        # Actually invoke the budget agent
                        budget_result = self.budget_agent.process_query(
                            client_id=client_id,
                            user_query=budget_query
                        )
                        
                        if budget_result.get("success"):
                            collaborative_data["budget_analysis"] = {
                                "query": budget_query,
                                "response": budget_result.get("response"),
                                "analysis_type": budget_result.get("analysis_type"),
                                "budget_operations": budget_result.get("budget_operations", 0)
                            }
                            print(f"[DEBUG] ✅ Budget collaboration successful")
                        else:
                            print(f"[DEBUG] ❌ Budget collaboration failed: {budget_result.get('error')}")
                            collaborative_data["budget_analysis"] = {
                                "error": budget_result.get("error"),
                                "fallback_note": "Could not retrieve budget data"
                            }

                    else:
                        print(f"[DEBUG] ⚠️ Agent {agent_name} not available for collaboration")
                        collaborative_data[f"{agent_name}_analysis"] = {
                            "error": f"{agent_name.title()} agent not available",
                            "fallback_note": f"Would integrate with {agent_name} agent for personal financial context"
                        }

                except Exception as e:
                    print(f"[DEBUG] ❌ Collaboration with {agent_name} failed: {e}")
                    collaborative_data[f"{agent_name}_analysis"] = {
                        "error": str(e),
                        "fallback_note": f"Failed to collaborate with {agent_name} agent"
                    }

            # FIXED: Always set collaborative_data as dict (never None)
            state["collaborative_data"] = collaborative_data
            state["execution_path"].append("collaboration_coordinator")
            
            print(f"[DEBUG] ✅ Collaboration complete with {len(collaborative_data)} agents")

        except Exception as e:
            print(f"[DEBUG] Collaboration coordination failed: {e}")
            # FIXED: Even on error, set empty dict instead of None
            state["collaborative_data"] = {"error": str(e)}

        return state

    def _generate_spending_query(self, user_query: str) -> str:
        """Generate spending-specific query from user query"""
        query_lower = user_query.lower()
        
        # Extract spending-related intent from user query
        if "last month" in query_lower:
            return "How much did I spend last month?"
        elif "afford" in query_lower:
            return "What are my recent spending patterns and monthly totals?"
        elif "spending" in query_lower:
            return "Show me my spending breakdown by category for recent months"
        else:
            return "What are my recent spending patterns?"

    def _generate_budget_query(self, user_query: str) -> str:
        """Generate budget-specific query from user query"""
        query_lower = user_query.lower()
        
        # Extract budget-related intent from user query
        if "afford" in query_lower:
            return "How am I doing against my budgets and what's my available budget?"
        elif "budget" in query_lower:
            return "How am I following my budgets?"
        else:
            return "What's my current budget status?"

    def _brand_translator_node(self, state: RAGAgentState) -> RAGAgentState:
        """Translate Bank of America content to GX Bank branding"""

        try:
            print("🔄 [DEBUG] Translating content to GX Bank branding...")

            raw_content = state.get("raw_content", [])
            
            if not raw_content:
                print("[DEBUG] No content to translate")
                state["translated_content"] = []
                state["execution_path"].append("brand_translator_skip")
                return state

            translated_content = []

            for content_item in raw_content:
                if not content_item.get("content"):
                    continue

                try:
                    translation_result = translate_bank_content.invoke({
                        "content": content_item["content"],
                        "content_type": content_item.get("content_type", "general"),
                        "target_tone": "professional",
                        "preserve_accuracy": True,
                        "context": {
                            "source_title": content_item.get("title", ""),
                            "source_url": content_item.get("url", "")
                        }
                    })

                    if translation_result.get("success"):
                        translated_content.append({
                            "original_content": content_item["content"],
                            "translated_content": translation_result["translated_content"],
                            "title": content_item.get("title", ""),
                            "url": content_item.get("url", ""),
                            "content_type": content_item.get("content_type", "general"),
                            "confidence_score": translation_result.get("confidence_score", 0.0),
                            "changes_made": translation_result.get("changes_made", [])
                        })
                    else:
                        # Keep original if translation fails
                        translated_content.append({
                            "original_content": content_item["content"],
                            "translated_content": content_item["content"],
                            "title": content_item.get("title", ""),
                            "url": content_item.get("url", ""),
                            "content_type": content_item.get("content_type", "general"),
                            "confidence_score": 0.5,
                            "changes_made": ["Translation failed, using original"]
                        })

                except Exception as translation_error:
                    print(f"[DEBUG] Translation failed for item: {translation_error}")
                    continue

            state["translated_content"] = translated_content
            state["execution_path"].append("brand_translator")
            print(f"[DEBUG] ✅ Translated {len(translated_content)} content pieces")

        except Exception as e:
            print(f"[DEBUG] Brand translation failed: {e}")
            state["translated_content"] = []

        return state

    def _enhanced_response_synthesizer_node(self, state: RAGAgentState) -> RAGAgentState:
        """ENHANCED: Synthesize final response using all available data including real collaboration results"""

        response_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a knowledgeable GX Bank representative helping customers with banking inquiries.

    You have access to:
    1. Current GX Bank product information and policies
    2. User's query: "{user_query}"
    3. User's personal financial context from our collaboration with internal agents
    4. Real spending and budget data when available

    Your task is to provide a comprehensive, helpful response that:

    ✅ **ANSWERS DIRECTLY** - Address the user's specific question
    ✅ **INTEGRATES PERSONAL DATA** - Use actual spending/budget data when available
    ✅ **PROVIDES ACTIONABLE ADVICE** - Give specific recommendations based on their data
    ✅ **USES GX BANK BRANDING** - All content is already translated to GX Bank
    ✅ **MAINTAINS ACCURACY** - All rates, terms, and policies are factual
    ✅ **SOUNDS NATURAL** - Like a helpful bank employee who has access to their account

    **RESPONSE STYLE:**
    - Professional but personalized tone
    - Start with acknowledging their specific situation if we have data
    - Make sure that if you have data make use of it if you dont have data no need to say "I dont have access to your financial data"
    - Provide specific details from both banking information AND their personal data
    - Give concrete recommendations based on their spending/budget patterns
    - End with helpful next steps

    **IMPORTANT:** When you have real spending or budget data, use it prominently in your response. For example:
    - "Based on your spending of $X last month..."
    - "Given your current budget status..."
    - "Looking at your spending patterns, I recommend..."

    Query Type: {analysis_type}
    Search Strategy: {search_strategy}

    **CRITICAL:** Never mention "Bank of America", "BoA", or reference external data sources. Everything should appear as native GX Bank information."""
            ),
            (
                "human",
                """User Query: "{user_query}"

    Available GX Bank Information:
    {banking_content}

    REAL Personal Financial Data from Collaboration:
    {collaboration_results}

    Please provide a comprehensive, personalized response as a GX Bank representative who has access to their account data."""
            )
        ])

        try:
            translated_content = state.get("translated_content", [])
            collaborative_data = state.get("collaborative_data", {})  # Default to empty dict
            intent = state.get("intent", {})

            # Prepare banking content
            banking_content_parts = []
            for item in translated_content:
                if item.get("translated_content"):
                    banking_content_parts.append(
                        f"**{item.get('title', 'Banking Information')}**\n{item['translated_content']}"
                    )

            banking_content = "\n\n".join(banking_content_parts) if banking_content_parts else "No specific banking information retrieved."

            # FIXED: Handle None collaborative_data properly
            collaboration_results = "No personal financial data available."
            
            if collaborative_data:  # Check if not None and not empty
                result_parts = []
                
                # Process spending analysis results
                spending_data = collaborative_data.get("spending_analysis", {})
                if spending_data and "response" in spending_data:
                    result_parts.append(f"**Spending Analysis:**\n{spending_data['response']}")
                    
                    # Add raw data if available
                    if spending_data.get("raw_data"):
                        raw_data = spending_data["raw_data"]
                        for data_chunk in raw_data:
                            results = data_chunk.get("results", [])
                            if results:
                                for result in results:
                                    if isinstance(result, dict) and "total_spent" in result:
                                        result_parts.append(f"- Total spending: ${result['total_spent']:,.2f}")
                
                # Process budget analysis results
                budget_data = collaborative_data.get("budget_analysis", {})
                if budget_data and "response" in budget_data:
                    result_parts.append(f"**Budget Analysis:**\n{budget_data['response']}")
                
                # Handle errors gracefully
                for key, value in collaborative_data.items():
                    if isinstance(value, dict) and value.get("error"):
                        result_parts.append(f"**{key.replace('_', ' ').title()}:** Unable to retrieve data - {value.get('fallback_note', 'Service temporarily unavailable')}")
                
                collaboration_results = "\n\n".join(result_parts) if result_parts else "Personal financial data collaboration completed but no specific data returned."

            print(f"[DEBUG] Generating enhanced response with {len(translated_content)} content pieces and collaboration results")

            response = self.llm.invoke(
                response_prompt.format_messages(
                    user_query=state["user_query"],
                    analysis_type=intent.get("query_type", "general"),
                    search_strategy=intent.get("search_strategy", "focused"),
                    banking_content=banking_content,
                    collaboration_results=collaboration_results
                )
            )

            state["response"] = response.content
            state["execution_path"].append("response_synthesizer")

            print(f"[DEBUG] ✅ Generated enhanced response length: {len(response.content)} characters")

        except Exception as e:
            print(f"[DEBUG] Enhanced response synthesis failed: {e}")
            
            # Enhanced fallback response that includes collaboration data
            collaborative_data = state.get("collaborative_data", {})  # FIXED: Default to empty dict
            translated_content = state.get("translated_content", [])
            
            fallback_parts = [f"Based on your inquiry about '{state['user_query']}':"]
            
            # FIXED: Safely check collaborative_data
            if collaborative_data:  # Only process if not None and not empty
                # Include spending data in fallback
                spending_data = collaborative_data.get("spending_analysis", {})
                if spending_data and "response" in spending_data:
                    fallback_parts.append(f"\n**Your Spending Information:**\n{spending_data['response']}")
                
                # Include budget data in fallback
                budget_data = collaborative_data.get("budget_analysis", {})
                if budget_data and "response" in budget_data:
                    fallback_parts.append(f"\n**Your Budget Status:**\n{budget_data['response']}")
            
            # Include external banking info
            if translated_content:
                first_content = translated_content[0].get("translated_content", "")
                fallback_parts.append(f"\n**GX Bank Information:**\n{first_content[:300]}...")
            
            if len(fallback_parts) == 1:
                fallback_parts.append("\nI'm here to help with all your GX Bank needs. Please let me know what specific information you're looking for!")
            
            state["response"] = "\n".join(fallback_parts)

        return state

    def _cache_updater_node(self, state: RAGAgentState) -> RAGAgentState:
        """Update semantic cache with new query and results including collaboration data"""

        try:
            print("💾 [DEBUG] Updating semantic cache...")

            cache_result = state.get("cache_result", {})
            
            # Only update cache if we didn't use cached data
            if cache_result.get("cache_hit"):
                print("[DEBUG] Skipping cache update - used cached data")
                state["execution_path"].append("cache_updater_skip")
                return state

            intent = state.get("intent", {})
            external_data = state.get("external_data", [])
            translated_content = state.get("translated_content", [])
            collaborative_data = state.get("collaborative_data", {}) 

            # Prepare data for caching (now all dictionaries)
            search_results = []
            for item in external_data:
                search_results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0.0)
                })

            safe_collaborative_data = collaborative_data if collaborative_data is not None else {}

            cache_add_result = semantic_cache_add.invoke({
                "query_text": state["user_query"],
                "search_results": search_results,
                "ai_response": state.get("response"),
                "external_data": {
                    "raw_content": state.get("raw_content", []),
                    "translated_content": translated_content,
                    "collaborative_data": safe_collaborative_data  
                },
                "intent_type": intent.get("query_type"),
                "product_type": intent.get("product_focus"),
                "search_strategy": intent.get("search_strategy"),
                "source_queries": [state["user_query"]],
                "ttl_hours": 24  # Shorter TTL for personal data
            })

            if cache_add_result.get("success"):
                print(f"[DEBUG] ✅ Added to cache with collaboration data: {cache_add_result['cache_id']}")
            else:
                print(f"[DEBUG] ❌ Cache update failed: {cache_add_result.get('error')}")

            state["execution_path"].append("cache_updater")

        except Exception as e:
            print(f"[DEBUG] Cache update failed: {e}")
            # Don't let cache failures break the entire flow
            state["execution_path"].append("cache_updater_error")

        return state

    def _error_handler_node(self, state: RAGAgentState) -> RAGAgentState:
        """Handle RAG-specific errors with helpful suggestions"""

        error_message = state.get("error", "Unknown RAG error occurred")
        print(f"🔧 [DEBUG] Handling RAG error: {error_message}")

        user_query = state["user_query"]

        # Provide specific help based on error type
        if "RAG intent" in error_message:
            suggestion = "I'd be happy to help! You can ask me about:\n• GX Bank credit cards and their benefits\n• Home loans and mortgage rates\n• Investment options and account types\n• Banking policies and services\n• Questions about your spending and budgets"
        elif "External data" in error_message:
            suggestion = "I'm having trouble accessing our latest product information right now. However, I can still help you with general banking questions and analyze your personal financial data. What specific product or service are you interested in?"
        elif "Collaboration" in error_message:
            suggestion = "I can provide information about our banking products and services. For personalized recommendations, I may need to access your account information. What would you like to know about GX Bank's offerings?"
        else:
            suggestion = "I'm here to help with all your GX Bank needs! You can ask about credit cards, loans, accounts, investment options, your spending patterns, or any banking services."

        state["response"] = f"""I want to make sure I provide you with the best information about GX Bank's services.

💡 **Here's how I can help:** {suggestion}

What specific GX Bank product or service would you like to learn more about?"""

        state["execution_path"].append("error_handler")
        return state

    def _route_after_intent(self, state: RAGAgentState) -> str:
        """Route after intent classification"""
        
        if state.get("error"):
            return "error"
        
        intent = state.get("intent")
        if not intent:
            state["error"] = "No RAG intent was classified"
            return "error"
        
        return "check_cache"

    def _route_after_cache(self, state: RAGAgentState) -> str:
        """Route after cache check"""
        
        cache_result = state.get("cache_result", {})
        intent = state.get("intent", {})
        
        if cache_result.get("cache_hit"):
            if intent.get("collaboration_agents"):
                return "collaborate"
            else:
                return "translate"
        else:
            # Need to fetch external data
            return "fetch_external"

    def _route_after_external_data(self, state: RAGAgentState) -> str:
        """Route after external data fetch"""
        
        intent = state.get("intent", {})
        
        if intent.get("collaboration_agents"):
            return "collaborate"
        else:
            return "translate"

    def process_query(
        self, 
        client_id: int, 
        user_query: str, 
        config: Dict = None
    ) -> Dict[str, Any]:
        """Process a RAG query with enhanced collaboration capabilities"""

        initial_state = RAGAgentState(
            client_id=client_id,
            user_query=user_query,
            intent=None,
            cache_result=None,
            external_data=None,
            collaborative_data={},  # FIXED: Initialize as empty dict, not None
            raw_content=None,
            translated_content=None,
            response=None,
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[],
            analysis_type=None,
        )

        try:
            final_state = self.graph.invoke(initial_state, config=config or {})

            # FIXED: Enhanced result reporting with comprehensive null safety
            collaboration_summary = {}
            collaborative_data = final_state.get("collaborative_data")
            
            print(f"[DEBUG] 🔍 Final collaborative_data type: {type(collaborative_data)}")
            print(f"[DEBUG] 🔍 Final collaborative_data value: {collaborative_data}")
            
            # SUPER SAFE: Handle all possible cases
            if collaborative_data is None:
                print("[DEBUG] ⚠️ collaborative_data is None")
                collaboration_summary = {}
            elif not isinstance(collaborative_data, dict):
                print(f"[DEBUG] ⚠️ collaborative_data is not a dict: {type(collaborative_data)}")
                collaboration_summary = {}
            else:
                # Only iterate if it's definitely a dict
                try:
                    for agent_name, data in collaborative_data.items():
                        if isinstance(data, dict):
                            collaboration_summary[agent_name] = {
                                "success": "error" not in data,
                                "has_data": "response" in data
                            }
                    print(f"[DEBUG] ✅ Collaboration summary built: {collaboration_summary}")
                except Exception as iter_error:
                    print(f"[DEBUG] ❌ Error iterating collaborative_data: {iter_error}")
                    collaboration_summary = {}

            return {
                "client_id": client_id,
                "query": user_query,
                "response": final_state.get("response"),
                "analysis_type": final_state.get("analysis_type"),
                "external_data_sources": len(final_state.get("external_data", [])),
                "cache_hit": final_state.get("cache_result", {}).get("cache_hit", False),
                "collaboration_used": bool(collaborative_data),
                "collaboration_summary": collaboration_summary,
                "content_translated": len(final_state.get("translated_content", [])),
                "execution_path": final_state.get("execution_path"),
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"⚠️ Enhanced RAG graph execution error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "client_id": client_id,
                "query": user_query,
                "response": "I encountered a system error while processing your banking inquiry. Please try again with a simpler question about GX Bank's products or services.",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False,
            }


# Export enhanced components
__all__ = [
    "RAGAgent",
    "RAGIntentClassification",
    "CollaborationRequest",
    "RAGAgentState"
]