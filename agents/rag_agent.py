import json
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal

import time

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
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

# Import tools
from tools.tavily_tools import (
    search_banking_products,
    search_bank_policies_and_services
)
from tools.semantic_cache import (
    semantic_cache_search,
    semantic_cache_add
)
from tools.brand_translator import (
    translate_bank_content,
    detect_content_type)

load_dotenv()


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


class RAGAgentState(TypedDict):
    
    client_id: int
    user_query: str
    intent: Optional[Dict[str, Any]]
    cache_result: Optional[Dict[str, Any]]
    external_data: Optional[List[Dict[str, Any]]]
    collaborative_data: Dict[str, Any]
    raw_response: Optional[str]  
    final_response: Optional[str]  
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    analysis_type: Optional[str]


class RAGAgent:

    def __init__(
        self,
        client_csv_path: Optional[str] = None,
        overall_csv_path: Optional[str] = None,
        model_name: str = "gpt-4o",
        memory: bool = True,
        spending_agent: Optional[Any] = None,
        budget_agent: Optional[Any] = None,
    ):
        print(f"🔍 Initializing Optimized RAG Agent...")
        
        if client_csv_path and overall_csv_path:
            print(f"📋 Client data: {client_csv_path}")
            print(f"📊 Overall data: {overall_csv_path}")
            # Import here to avoid circular imports
            from data_store.data_store import DataStore
            self.data_store = DataStore(
                client_csv_path=client_csv_path, 
                overall_csv_path=overall_csv_path
            )
        else:
            print("📋 Running in external-only mode")
            self.data_store = None

        # Store agent references for collaboration
        self.spending_agent = spending_agent
        self.budget_agent = budget_agent

        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),            
            temperature=0,
        )

        # Set up structured output parser
        self.intent_parser = PydanticOutputParser(
            pydantic_object=RAGIntentClassification
        )

        # Setup memory
        self.memory = MemorySaver() if memory else None

        # Build the optimized workflow graph
        self.graph = self._build_optimized_graph()
        print("✅ Optimized RAG Agent initialized!")

    def _build_optimized_graph(self) -> StateGraph:
        """Build the optimized RAG workflow with post-response translation"""

        workflow = StateGraph(RAGAgentState)

        # Optimized workflow nodes
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("cache_checker", self._cache_checker_node)
        workflow.add_node("external_data_fetcher", self._external_data_fetcher_node)
        workflow.add_node("collaboration_coordinator", self._collaboration_coordinator_node)
        workflow.add_node("raw_response_generator", self._raw_response_generator_node)  # New: Generate raw response
        workflow.add_node("response_translator", self._response_translator_node)        # New: Translate final response
        workflow.add_node("cache_updater", self._cache_updater_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Entry point
        workflow.set_entry_point("intent_classifier")

        # Optimized routing logic
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
                "generate_response": "raw_response_generator"
            }
        )

        workflow.add_conditional_edges(
            "external_data_fetcher",
            self._route_after_external_data,
            {
                "collaborate": "collaboration_coordinator",
                "generate_response": "raw_response_generator"
            }
        )

        workflow.add_edge("collaboration_coordinator", "raw_response_generator")
        workflow.add_edge("raw_response_generator", "response_translator")
        workflow.add_edge("response_translator", "cache_updater")
        workflow.add_edge("cache_updater", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.memory)

    def _intent_classifier_node(self, state: RAGAgentState) -> RAGAgentState:
        """Enhanced intent classification for RAG queries"""
        # Same as before - no changes needed
        try:
            print(f"🔍 [DEBUG] Classifying RAG intent for: {state['user_query']}")

            classification_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an AI assistant that classifies user queries for a banking RAG system.

Analyze the user's query and determine the classification details as before...

{format_instructions}

Analyze this query and provide structured classification:"""),
                ("human", "{user_query}"),
            ])

            formatted_prompt = classification_prompt.partial(
                format_instructions=self.intent_parser.get_format_instructions()
            )

            classification_chain = formatted_prompt | self.llm | self.intent_parser

            try:
                intent_result = classification_chain.invoke(
                    {"user_query": state["user_query"]}
                )

                intent_dict = intent_result.model_dump()

                print(f"[DEBUG] RAG intent classified as: {intent_dict['query_type']}")
                print(f"[DEBUG] Strategy: {intent_dict['search_strategy']}, Confidence: {intent_dict['confidence']}")

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
                fallback_result = self._fallback_rag_classification(state["user_query"])
                state["intent"] = fallback_result
                state["analysis_type"] = fallback_result["query_type"]
                state["execution_path"].append("intent_classifier")

        except Exception as e:
            print(f"[DEBUG] RAG intent classification failed: {e}")
            state["error"] = f"RAG intent classification error: {str(e)}"

        return state

    def _cache_checker_node(self, state: RAGAgentState) -> RAGAgentState:
        """Check semantic cache for similar queries"""
        # Same as before - no changes needed
        try:
            print("💾 [DEBUG] Checking semantic cache...")

            intent = state.get("intent", {})
            
            cache_result = semantic_cache_search.invoke({
                "query_text": state["user_query"],
                "intent_type": intent.get("query_type"),
                "product_type": intent.get("product_focus"),
                "similarity_threshold": 0.87
            })

            state["cache_result"] = cache_result
            state["execution_path"].append("cache_checker")

            if cache_result.get("cache_hit"):
                print(f"[DEBUG] ✅ Cache HIT: similarity={cache_result['similarity_score']:.3f}")
                cached_data = cache_result.get("cached_data", {})
                state["external_data"] = cached_data.get("search_results", [])
                state["collaborative_data"] = cached_data.get("external_data", {}).get("collaborative_data", {})
                # NEW: If we have cached final response, use it directly
                if cache_result.get("cached_data", {}).get("ai_response"):
                    state["final_response"] = cache_result["cached_data"]["ai_response"]
            else:
                print(f"[DEBUG] ❌ Cache MISS: best_similarity={cache_result['similarity_score']:.3f}")

        except Exception as e:
            print(f"[DEBUG] Cache check failed: {e}")
            state["cache_result"] = {"cache_hit": False, "error": str(e)}

        return state

    def _external_data_fetcher_node(self, state: RAGAgentState) -> RAGAgentState:
        """Fetch external banking data - OPTIMIZED: No translation here"""

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

            # Fetch data based on intent (same as before)
            if intent.get("query_type") == "banking_product" and intent.get("product_focus"):
                print(f"[DEBUG] Searching for banking products: {intent['product_focus']}")
                
                product_result = search_banking_products.invoke({
                    "user_query": state["user_query"],
                    "product_type": intent["product_focus"],
                    "user_criteria": None,
                    "max_results": 5
                })

                if not product_result.get("error"):
                    results = product_result.get("results", [])
                    for result in results:
                        if hasattr(result, 'title'):
                            external_data.append({
                                "title": result.title,
                                "content": result.content,
                                "url": result.url,
                                "score": result.score
                            })
                        else:
                            external_data.append(result)

            elif intent.get("query_type") in ["policy_service", "hybrid", "collaboration_needed"]:
                print("[DEBUG] Searching for bank policies and services")
                
                focus_area = "rates" if "rate" in state["user_query"].lower() else "general"
                
                policy_result = search_bank_policies_and_services.invoke({
                    "user_query": state["user_query"],
                    "focus_area": focus_area,
                    "include_rates": True
                })

                if not policy_result.get("error"):
                    results = policy_result.get("results", [])
                    for result in results:
                        if hasattr(result, 'title'):
                            external_data.append({
                                "title": result.title,
                                "content": result.content,
                                "url": result.url,
                                "score": getattr(result, 'score', 0.0)
                            })
                        else:
                            external_data.append(result)

            # OPTIMIZATION: Store raw content WITHOUT translation
            state["external_data"] = external_data
            state["execution_path"].append("external_data_fetcher")
            
            print(f"[DEBUG] ✅ Fetched {len(external_data)} external data sources (no translation yet)")

        except Exception as e:
            print(f"[DEBUG] External data fetching failed: {e}")
            state["error"] = f"External data fetch error: {e}"

        return state

    def _collaboration_coordinator_node(self, state: RAGAgentState) -> RAGAgentState:
        """Real collaboration with other agents"""
        # Same as before - no changes needed
        try:
            print("🤝 [DEBUG] Coordinating with other agents...")

            intent = state.get("intent", {})
            collaboration_agents = intent.get("collaboration_agents", [])
            collaborative_data = {}

            if not collaboration_agents:
                print("[DEBUG] No collaboration needed")
                state["collaborative_data"] = collaborative_data
                state["execution_path"].append("collaboration_coordinator_skip")
                return state

            client_id = state["client_id"]
            user_query = state["user_query"]

            for agent_name in collaboration_agents:
                try:
                    if agent_name == "spending" and self.spending_agent:
                        print(f"[DEBUG] 📊 Collaborating with SPENDING agent...")
                        spending_query = self._generate_spending_query(user_query)
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
                            print(f"[DEBUG] ❌ Spending collaboration failed")
                            collaborative_data["spending_analysis"] = {
                                "error": spending_result.get("error"),
                                "fallback_note": "Could not retrieve spending data"
                            }

                    elif agent_name == "budget" and self.budget_agent:
                        print(f"[DEBUG] 💰 Collaborating with BUDGET agent...")
                        budget_query = self._generate_budget_query(user_query)
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
                            print(f"[DEBUG] ❌ Budget collaboration failed")
                            collaborative_data["budget_analysis"] = {
                                "error": budget_result.get("error"),
                                "fallback_note": "Could not retrieve budget data"
                            }

                except Exception as e:
                    print(f"[DEBUG] ❌ Collaboration with {agent_name} failed: {e}")
                    collaborative_data[f"{agent_name}_analysis"] = {
                        "error": str(e),
                        "fallback_note": f"Failed to collaborate with {agent_name} agent"
                    }

            state["collaborative_data"] = collaborative_data
            state["execution_path"].append("collaboration_coordinator")
            
            print(f"[DEBUG] ✅ Collaboration complete with {len(collaborative_data)} agents")

        except Exception as e:
            print(f"[DEBUG] Collaboration coordination failed: {e}")
            state["collaborative_data"] = {"error": str(e)}

        return state

    def _raw_response_generator_node(self, state: RAGAgentState) -> RAGAgentState:
        """NEW: Generate response using raw Bank of America data (no translation yet)"""

        response_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a knowledgeable Bank of America representative helping customers with banking inquiries.

You have access to:
1. Current Bank of America product information and policies
2. User's query: "{user_query}"
3. User's personal financial context from collaboration with internal agents
4. Real spending and budget data when available

Your task is to provide a comprehensive, helpful response that:

✅ **ANSWERS DIRECTLY** - Address the user's specific question
✅ **INTEGRATES PERSONAL DATA** - Use actual spending/budget data when available  
✅ **PROVIDES ACTIONABLE ADVICE** - Give specific recommendations based on their data
✅ **USES BANK OF AMERICA BRANDING** - This will be translated later, so use original BoA content
✅ **MAINTAINS ACCURACY** - All rates, terms, and policies are factual
✅ **SOUNDS NATURAL** - Like a helpful Bank of America employee

**IMPORTANT:** 
- Use "Bank of America" naturally in your response
- Include actual rates, terms, and product names from the data
- Reference "Bank of America" products and services as they appear in the source data
- The response will be translated to our bank's branding later, so be accurate with BoA details

Query Type: {analysis_type}
Search Strategy: {search_strategy}"""),
            (
                "human",
                """User Query: "{user_query}"

Available Bank of America Information:
{banking_content}

REAL Personal Financial Data from Collaboration:
{collaboration_results}

Please provide a comprehensive response as a Bank of America representative."""
            )
        ])

        try:
            external_data = state.get("external_data", [])
            collaborative_data = state.get("collaborative_data", {})
            intent = state.get("intent", {})

            # Prepare banking content from raw external data (no translation)
            banking_content_parts = []
            for item in external_data:
                if item.get("content"):
                    banking_content_parts.append(
                        f"**{item.get('title', 'Banking Information')}**\n{item['content']}"
                    )

            banking_content = "\n\n".join(banking_content_parts) if banking_content_parts else "No specific banking information retrieved."

            # Process collaboration results
            collaboration_results = "No personal financial data available."
            
            if collaborative_data:
                result_parts = []
                
                spending_data = collaborative_data.get("spending_analysis", {})
                if spending_data and "response" in spending_data:
                    result_parts.append(f"**Spending Analysis:**\n{spending_data['response']}")
                    
                    if spending_data.get("raw_data"):
                        raw_data = spending_data["raw_data"]
                        for data_chunk in raw_data:
                            results = data_chunk.get("results", [])
                            if results:
                                for result in results:
                                    if isinstance(result, dict) and "total_spent" in result:
                                        result_parts.append(f"- Total spending: ${result['total_spent']:,.2f}")
                
                budget_data = collaborative_data.get("budget_analysis", {})
                if budget_data and "response" in budget_data:
                    result_parts.append(f"**Budget Analysis:**\n{budget_data['response']}")
                
                collaboration_results = "\n\n".join(result_parts) if result_parts else "Personal financial data collaboration completed but no specific data returned."

            print(f"[DEBUG] Generating RAW response (Bank of America branding) with {len(external_data)} content pieces")

            response = self.llm.invoke(
                response_prompt.format_messages(
                    user_query=state["user_query"],
                    analysis_type=intent.get("query_type", "general"),
                    search_strategy=intent.get("search_strategy", "focused"),
                    banking_content=banking_content,
                    collaboration_results=collaboration_results
                )
            )

            state["raw_response"] = response.content
            state["execution_path"].append("raw_response_generator")

            print(f"[DEBUG] ✅ Generated raw response length: {len(response.content)} characters")

        except Exception as e:
            print(f"[DEBUG] Raw response generation failed: {e}")
            
            # Enhanced fallback response
            collaborative_data = state.get("collaborative_data", {})
            external_data = state.get("external_data", [])
            
            fallback_parts = [f"Based on your inquiry about '{state['user_query']}':"]
            
            if collaborative_data:
                spending_data = collaborative_data.get("spending_analysis", {})
                if spending_data and "response" in spending_data:
                    fallback_parts.append(f"\n**Your Spending Information:**\n{spending_data['response']}")
                
                budget_data = collaborative_data.get("budget_analysis", {})
                if budget_data and "response" in budget_data:
                    fallback_parts.append(f"\n**Your Budget Status:**\n{budget_data['response']}")
            
            if external_data:
                first_content = external_data[0].get("content", "")
                fallback_parts.append(f"\n**Bank of America Information:**\n{first_content[:300]}...")
            
            if len(fallback_parts) == 1:
                fallback_parts.append("\nI'm here to help with all your Bank of America needs. Please let me know what specific information you're looking for!")
            
            state["raw_response"] = "\n".join(fallback_parts)

        return state

    def _response_translator_node(self, state: RAGAgentState) -> RAGAgentState:
        """NEW: Translate the final response from Bank of America to GX Bank branding"""

        try:
            print("🔄 [DEBUG] Translating final response to GX Bank branding...")

            raw_response = state.get("raw_response")
            if not raw_response:
                print("[DEBUG] No raw response to translate")
                state["final_response"] = "I apologize, but I couldn't generate a response to your query."
                state["execution_path"].append("response_translator_skip")
                return state

            # Detect content type for better translation
            content_type = detect_content_type(raw_response)
            
            # Translate the entire response in one call
            translation_result = translate_bank_content.invoke({
                "content": raw_response,
                "content_type": content_type,
                "target_tone": "professional",
                "preserve_accuracy": True,
                "context": {
                    "user_query": state["user_query"],
                    "analysis_type": state.get("analysis_type", "general")
                }
            })

            if translation_result.get("success"):
                state["final_response"] = translation_result["translated_content"]
                print(f"[DEBUG] ✅ Response translated successfully (confidence: {translation_result.get('confidence_score', 0.0):.2f})")
                print(f"[DEBUG] Changes made: {len(translation_result.get('changes_made', []))}")
            else:
                print(f"[DEBUG] ❌ Translation failed: {translation_result.get('error')}")
                # Use basic fallback translation
                state["final_response"] = self._basic_brand_replacement(raw_response)

            state["execution_path"].append("response_translator")

        except Exception as e:
            print(f"[DEBUG] Response translation failed: {e}")
            # Emergency fallback - basic string replacement
            raw_response = state.get("raw_response", "")
            state["final_response"] = self._basic_brand_replacement(raw_response)
            state["execution_path"].append("response_translator_fallback")

        return state

    def _basic_brand_replacement(self, content: str) -> str:
        """Emergency fallback for brand replacement"""
        import re
        
        replacements = {
            r'\bBank of America\b': 'GX Bank',
            r'\bBofA\b': 'GX Bank',
            r'\bBoA\b': 'GX Bank',
            r'\bMerrill Lynch\b': 'GX Investment Services',
            r'\bMerrill\b': 'GX Investment'
        }
        
        result = content
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result

    def _cache_updater_node(self, state: RAGAgentState) -> RAGAgentState:
        """Update semantic cache with new query and results"""

        try:
            print("💾 [DEBUG] Updating semantic cache...")

            cache_result = state.get("cache_result", {})
            
            if cache_result.get("cache_hit"):
                print("[DEBUG] Skipping cache update - used cached data")
                state["execution_path"].append("cache_updater_skip")
                return state

            intent = state.get("intent", {})
            external_data = state.get("external_data", [])
            collaborative_data = state.get("collaborative_data", {})

            # Prepare data for caching
            search_results = []
            for item in external_data:
                search_results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0.0)
                })

            safe_collaborative_data = collaborative_data if collaborative_data is not None else {}

            # Cache with the FINAL translated response
            cache_add_result = semantic_cache_add.invoke({
                "query_text": state["user_query"],
                "search_results": search_results,
                "ai_response": state.get("final_response"),  # Cache the translated response
                "external_data": {
                    "raw_content": external_data,  # Raw content for future reference
                    "collaborative_data": safe_collaborative_data
                },
                "intent_type": intent.get("query_type"),
                "product_type": intent.get("product_focus"),
                "search_strategy": intent.get("search_strategy"),
                "source_queries": [state["user_query"]],
                "ttl_hours": 24
            })

            if cache_add_result.get("success"):
                print(f"[DEBUG] ✅ Added translated response to cache: {cache_add_result['cache_id']}")
            else:
                print(f"[DEBUG] ❌ Cache update failed: {cache_add_result.get('error')}")

            state["execution_path"].append("cache_updater")

        except Exception as e:
            print(f"[DEBUG] Cache update failed: {e}")
            state["execution_path"].append("cache_updater_error")

        return state

    def _error_handler_node(self, state: RAGAgentState) -> RAGAgentState:
        """Handle RAG-specific errors"""

        error_message = state.get("error", "Unknown RAG error occurred")
        print(f"🔧 [DEBUG] Handling RAG error: {error_message}")

        user_query = state["user_query"]

        if "RAG intent" in error_message:
            suggestion = "I'd be happy to help! You can ask me about:\n• GX Bank credit cards and their benefits\n• Home loans and mortgage rates\n• Investment options and account types\n• Banking policies and services\n• Questions about your spending and budgets"
        elif "External data" in error_message:
            suggestion = "I'm having trouble accessing our latest product information right now. However, I can still help you with general banking questions and analyze your personal financial data. What specific product or service are you interested in?"
        elif "Collaboration" in error_message:
            suggestion = "I can provide information about our banking products and services. For personalized recommendations, I may need to access your account information. What would you like to know about GX Bank's offerings?"
        else:
            suggestion = "I'm here to help with all your GX Bank needs! You can ask about credit cards, loans, accounts, investment options, your spending patterns, or any banking services."

        state["final_response"] = f"""I want to make sure I provide you with the best information about GX Bank's services.

💡 **Here's how I can help:** {suggestion}

What specific GX Bank product or service would you like to learn more about?"""

        state["execution_path"].append("error_handler")
        return state

    # Helper methods (same as before)
    def _generate_spending_query(self, user_query: str) -> str:
        """Generate spending-specific query from user query"""
        query_lower = user_query.lower()
        
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
        
        if "afford" in query_lower:
            return "How am I doing against my budgets and what's my available budget?"
        elif "budget" in query_lower:
            return "How am I following my budgets?"
        else:
            return "What's my current budget status?"

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
            "confidence": 0.8,
            "reasoning": "Enhanced fallback keyword-based classification with collaboration detection"
        }

    # Routing methods
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
            # Check if we have a complete cached response
            if cache_result.get("cached_data", {}).get("ai_response"):
                return "generate_response"  # Skip to response generation (no translation needed)
            elif intent.get("collaboration_agents"):
                return "collaborate"
            else:
                return "generate_response"
        else:
            # Need to fetch external data
            return "fetch_external"

    def _route_after_external_data(self, state: RAGAgentState) -> str:
        """Route after external data fetch"""
        
        intent = state.get("intent", {})
        
        if intent.get("collaboration_agents"):
            return "collaborate"
        else:
            return "generate_response"

    def process_query(
        self, 
        client_id: int, 
        user_query: str, 
        config: Dict = None
    ) -> Dict[str, Any]:
        """Process a RAG query with optimized workflow"""

        initial_state = RAGAgentState(
            client_id=client_id,
            user_query=user_query,
            intent=None,
            cache_result=None,
            external_data=None,
            collaborative_data={},
            raw_response=None,
            final_response=None,
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[],
            analysis_type=None,
        )

        try:
            final_state = self.graph.invoke(initial_state, config=config or {})

            # Enhanced result reporting with performance metrics
            collaboration_summary = {}
            collaborative_data = final_state.get("collaborative_data", {})
            
            if isinstance(collaborative_data, dict):
                for agent_name, data in collaborative_data.items():
                    if isinstance(data, dict):
                        collaboration_summary[agent_name] = {
                            "success": "error" not in data,
                            "has_data": "response" in data
                        }

            # Calculate optimization metrics
            execution_path = final_state.get("execution_path", [])
            translation_optimized = "response_translator" in execution_path
            cache_used = "cache_checker" in execution_path and final_state.get("cache_result", {}).get("cache_hit", False)

            return {
                "client_id": client_id,
                "query": user_query,
                "response": final_state.get("final_response"),  # Return final translated response
                "analysis_type": final_state.get("analysis_type"),
                "external_data_sources": len(final_state.get("external_data", [])),
                "cache_hit": cache_used,
                "collaboration_used": bool(collaborative_data),
                "collaboration_summary": collaboration_summary,
                "execution_path": execution_path,
                "optimization_metrics": {
                    "translation_calls": 1 if translation_optimized else 0,  # Only 1 translation call
                    "raw_response_length": len(final_state.get("raw_response", "")),
                    "final_response_length": len(final_state.get("final_response", "")),
                    "cache_utilized": cache_used,
                    "nodes_executed": len(execution_path)
                },
                "error": final_state.get("error"),
                "timestamp": datetime.now().isoformat(),
                "success": final_state.get("error") is None,
            }

        except Exception as e:
            print(f"⚠️ Optimized RAG graph execution error: {e}")
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


# Performance comparison function
def compare_performance(original_agent, optimized_agent, client_id: int, test_queries: List[str]) -> Dict[str, Any]:
    """Compare performance between original and optimized RAG agents"""
    
    import time
    
    results = {
        "original_agent": {"total_time": 0, "translation_calls": 0, "responses": []},
        "optimized_agent": {"total_time": 0, "translation_calls": 0, "responses": []}
    }
    
    print("🔍 Starting performance comparison...")
    
    for query in test_queries:
        print(f"\n📝 Testing query: {query[:50]}...")
        
        # Test original agent
        start_time = time.time()
        original_result = original_agent.process_query(client_id, query)
        original_time = time.time() - start_time
        
        results["original_agent"]["total_time"] += original_time
        results["original_agent"]["responses"].append({
            "query": query,
            "time": original_time,
            "success": original_result.get("success", False)
        })
        
        # Test optimized agent  
        start_time = time.time()
        optimized_result = optimized_agent.process_query(client_id, query)
        optimized_time = time.time() - start_time
        
        results["optimized_agent"]["total_time"] += optimized_time
        results["optimized_agent"]["responses"].append({
            "query": query,
            "time": optimized_time,
            "success": optimized_result.get("success", False),
            "optimization_metrics": optimized_result.get("optimization_metrics", {})
        })
        
        print(f"   Original: {original_time:.2f}s | Optimized: {optimized_time:.2f}s | Improvement: {((original_time - optimized_time) / original_time * 100):.1f}%")
    
    # Calculate overall metrics
    original_avg = results["original_agent"]["total_time"] / len(test_queries)
    optimized_avg = results["optimized_agent"]["total_time"] / len(test_queries)
    improvement = ((original_avg - optimized_avg) / original_avg * 100) if original_avg > 0 else 0
    
    results["summary"] = {
        "queries_tested": len(test_queries),
        "original_avg_time": original_avg,
        "optimized_avg_time": optimized_avg,
        "performance_improvement_percent": improvement,
        "total_time_saved": results["original_agent"]["total_time"] - results["optimized_agent"]["total_time"]
    }
    
    return results


# Demo function
def demo_optimized_rag_agent():
    """Demonstrate the optimized RAG agent"""
    
    print("🚀 OPTIMIZED RAG AGENT DEMO")
    print("=" * 60)
    
    # Initialize optimized agent (you'll need to provide actual paths)
    try:
        optimized_agent = RAGAgent(
            client_csv_path="path/to/Banking_Data.csv",
            overall_csv_path="path/to/overall_data.csv",
            model_name="gpt-4o",
            memory=False
        )
        
        test_queries = [
            "What credit cards does GX Bank offer?",
            "Based on my spending, which credit card suits me best?"
        ]
        
        client_id = 430
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            print("-" * 50)
            
            start_time = time.time()
            result = optimized_agent.process_query(client_id=client_id, user_query=query)
            end_time = time.time()
            
            print(f"✅ Success: {result.get('success', False)}")
            print(f"⏱️  Processing Time: {end_time - start_time:.2f}s")
            print(f"🔄 Translation Calls: {result.get('optimization_metrics', {}).get('translation_calls', 0)}")
            print(f"🗂️  Execution Path: {' → '.join(result.get('execution_path', []))}")
            
            response = result.get("response", "No response")
            print(f"\n💬 Response Preview: {response[:200]}...")
            
            if result.get("optimization_metrics"):
                metrics = result["optimization_metrics"]
                print(f"\n📊 Optimization Metrics:")
                print(f"   - Cache Used: {metrics.get('cache_utilized', False)}")
                print(f"   - Nodes Executed: {metrics.get('nodes_executed', 0)}")
                print(f"   - Response Length: {metrics.get('final_response_length', 0)} chars")
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    demo_optimized_rag_agent()