import os
import json
import time
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from pydantic import BaseModel, Field, validator

load_dotenv()


# Pydantic Models for Tavily Integration
class TavilySearchRequest(BaseModel):
    """Structured request model for Tavily searches"""
    
    query: str = Field(
        description="The search query to execute",
        min_length=3,
        max_length=400
    )
    search_depth: str = Field(
        default="advanced",
        description="Search depth: 'basic' or 'advanced'"
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Specific domains to include in search"
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Domains to exclude from search"
    )
    max_results: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )
    include_images: bool = Field(
        default=False,
        description="Whether to include images in results"
    )
    include_answer: bool = Field(
        default=True,
        description="Whether to include AI-generated answer"
    )
    
    @validator('search_depth')
    def validate_search_depth(cls, v):
        if v not in ['basic', 'advanced']:
            raise ValueError("search_depth must be 'basic' or 'advanced'")
        return v


class TavilySearchResult(BaseModel):
    """Individual search result from Tavily"""
    
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the source")
    content: str = Field(description="Content/snippet from the source")
    score: float = Field(description="Relevance score", ge=0.0, le=1.0)
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date if available"
    )


class TavilySearchResponse(BaseModel):
    """Complete response from Tavily API"""
    
    query: str = Field(description="Original search query")
    follow_up_questions: Optional[List[str]] = Field(
        default=None,
        description="Suggested follow-up questions"
    )
    answer: Optional[str] = Field(
        default=None,
        description="AI-generated answer summary"
    )
    results: List[TavilySearchResult] = Field(
        description="List of search results"
    )
    response_time: float = Field(description="Response time in seconds")
    images: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Image results if requested"
    )


class BankingProductQuery(BaseModel):
    """Structured query for banking products"""
    
    product_type: str = Field(
        description="Type of banking product: credit_card, debit_card, home_loan, auto_loan, investment, account"
    )
    user_criteria: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User-specific criteria (income, credit score, etc.)"
    )
    specific_requirements: Optional[List[str]] = Field(
        default=None,
        description="Specific features or requirements"
    )
    location: Optional[str] = Field(
        default=None,
        description="Geographic location for region-specific products"
    )
    
    @validator('product_type')
    def validate_product_type(cls, v):
        valid_types = [
            'credit_card', 'debit_card', 'home_loan', 'auto_loan', 
            'investment', 'account', 'insurance', 'savings'
        ]
        if v not in valid_types:
            raise ValueError(f"product_type must be one of {valid_types}")
        return v


class SearchQueryGeneration(BaseModel):
    """Generated search queries with context"""
    
    primary_query: str = Field(description="Main search query")
    secondary_queries: List[str] = Field(
        default=[],
        description="Additional supporting queries"
    )
    search_strategy: str = Field(
        description="Strategy used: focused, broad, or comparative"
    )
    bank_focus: str = Field(
        default="Bank of America",
        description="Primary bank to focus on"
    )
    expected_domains: List[str] = Field(
        default=[],
        description="Expected domains for relevant results"
    )

class TavilyClient:
    """Enhanced Tavily API client with error handling and rate limiting"""
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.base_url = "https://api.tavily.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search(self, request: TavilySearchRequest) -> TavilySearchResponse:
        """Execute search with structured request/response"""
        
        self._rate_limit()
        
        payload = {
            "api_key": self.api_key,
            "query": request.query,
            "search_depth": request.search_depth,
            "max_results": request.max_results,
            "include_images": request.include_images,
            "include_answer": request.include_answer
        }
        
        # Add optional parameters
        if request.include_domains:
            payload["include_domains"] = request.include_domains
        if request.exclude_domains:
            payload["exclude_domains"] = request.exclude_domains
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            response.raise_for_status()
            data = response.json()
            
            # Transform to our Pydantic model
            results = []
            for result in data.get("results", []):
                results.append(TavilySearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    published_date=result.get("published_date")
                ))
            
            return TavilySearchResponse(
                query=request.query,
                follow_up_questions=data.get("follow_up_questions"),
                answer=data.get("answer"),
                results=results,
                response_time=response_time,
                images=data.get("images")
            )
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Tavily API request failed: {e}")
        except Exception as e:
            raise Exception(f"Tavily search error: {e}")


# Global Tavily client instance
_tavily_client: Optional[TavilyClient] = None

def _get_tavily_client() -> TavilyClient:
    """Get or create Tavily client instance"""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


# Search Query Templates for Banking Products
BANKING_SEARCH_TEMPLATES = {
    "credit_card": {
        "general": "Bank of America credit cards {criteria} benefits rewards cashback",
        "specific": "Bank of America {card_name} credit card features benefits requirements",
        "comparison": "Bank of America credit cards compare {user_profile} best options"
    },
    "debit_card": {
        "general": "Bank of America debit cards ATM fees checking account benefits",
        "specific": "Bank of America {card_name} debit card features ATM network",
        "comparison": "Bank of America checking accounts debit card options compare"
    },
    "home_loan": {
        "general": "Bank of America mortgage home loan rates {location} requirements",
        "specific": "Bank of America {loan_type} mortgage rates terms conditions",
        "comparison": "Bank of America home loan options first time buyer {income_range}"
    },
    "auto_loan": {
        "general": "Bank of America auto loan rates {vehicle_type} financing options",
        "specific": "Bank of America auto loan {car_brand} {car_model} financing rates",
        "comparison": "Bank of America auto loan rates vs competitors {credit_score_range}"
    },
    "investment": {
        "general": "Bank of America investment options Merrill Lynch advisory services",
        "specific": "Bank of America {investment_type} portfolio management fees",
        "comparison": "Bank of America investment accounts IRA 401k options"
    },
    "account": {
        "general": "Bank of America checking savings account types fees benefits",
        "specific": "Bank of America {account_type} account features minimum balance",
        "comparison": "Bank of America account options {customer_type} best choice"
    }
}


@tool
def generate_banking_search_queries(
    user_query: str,
    product_query: BankingProductQuery
) -> SearchQueryGeneration:
    """
    Generate optimized search queries for banking products using LLM and templates.
    """
    
    # Get relevant template
    templates = BANKING_SEARCH_TEMPLATES.get(
        product_query.product_type, 
        BANKING_SEARCH_TEMPLATES["credit_card"]
    )
    
    # Build context for LLM
    context_parts = [
        f"Product Type: {product_query.product_type}",
        f"User Query: {user_query}"
    ]
    
    if product_query.user_criteria:
        criteria_str = ", ".join([f"{k}: {v}" for k, v in product_query.user_criteria.items()])
        context_parts.append(f"User Criteria: {criteria_str}")
    
    if product_query.specific_requirements:
        context_parts.append(f"Requirements: {', '.join(product_query.specific_requirements)}")
    
    if product_query.location:
        context_parts.append(f"Location: {product_query.location}")
    
    context = "\n".join(context_parts)
    
    # Generate queries using LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at generating search queries for banking products.

Generate 1-3 optimized search queries for Tavily to find Bank of America banking product information.

REQUIREMENTS:
- ALWAYS include "Bank of America" in the primary query
- Focus on specific product features, rates, and requirements
- Include user criteria when relevant
- Make queries specific enough to get actionable information
- Avoid overly broad queries

AVAILABLE TEMPLATES:
{templates}

Generate queries that will return:
1. Product features and benefits
2. Rates, fees, and requirements
3. Comparison information when relevant
4. Price of the product user is considering to buy or explore

Return JSON with:
- primary_query: Main search query
- secondary_queries: 0-2 additional supporting queries
- search_strategy: "focused", "broad", or "comparative"
"""),
        ("human", """User Context:
{context}

Generate optimized search queries for this banking inquiry.""")
    ])
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),            
            temperature=0,
        )   
        response = llm.invoke(
            prompt.format_messages(
                templates=json.dumps(templates, indent=2),
                context=context
            )
        )
        
        # Parse JSON response
        try:
            response_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback to template-based generation
            response_data = _generate_fallback_queries(product_query, user_query)
        
        return SearchQueryGeneration(
            primary_query=response_data.get("primary_query", ""),
            secondary_queries=response_data.get("secondary_queries", []),
            search_strategy=response_data.get("search_strategy", "focused"),
            bank_focus="Bank of America",
            expected_domains=[
                "bankofamerica.com",
                "merrillynch.com", 
                "nerdwallet.com",
                "bankrate.com",
                "creditkarma.com"
            ]
        )
        
    except Exception as e:
        print(f"[DEBUG] LLM query generation failed: {e}, using fallback")
        return SearchQueryGeneration(
            primary_query=_generate_fallback_queries(product_query, user_query)["primary_query"],
            secondary_queries=[],
            search_strategy="focused",
            bank_focus="Bank of America",
            expected_domains=["bankofamerica.com"]
        )


def _generate_fallback_queries(
    product_query: BankingProductQuery, 
    user_query: str
) -> Dict[str, Any]:
    """Generate fallback queries using templates"""
    
    templates = BANKING_SEARCH_TEMPLATES.get(product_query.product_type, {})
    general_template = templates.get("general", "Bank of America {product_type}")
    
    # Simple substitution
    criteria = ""
    if product_query.user_criteria:
        criteria = " ".join([str(v) for v in product_query.user_criteria.values()])
    
    primary_query = general_template.format(
        criteria=criteria,
        product_type=product_query.product_type.replace("_", " ")
    )
    
    return {
        "primary_query": primary_query,
        "secondary_queries": [],
        "search_strategy": "focused"
    }


@tool
def execute_tavily_search(
    search_request: TavilySearchRequest
) -> TavilySearchResponse:
    """
    Execute a Tavily search with structured request/response handling.
    """
    
    try:
        client = _get_tavily_client()
        
        print(f"[DEBUG] Executing Tavily search: {search_request.query[:60]}...")
        
        response = client.search(search_request)
        
        print(f"[DEBUG] Tavily search completed: {len(response.results)} results in {response.response_time:.2f}s")
        
        return response
        
    except Exception as e:
        print(f"[DEBUG] Tavily search failed: {e}")
        # Return empty response instead of failing
        return TavilySearchResponse(
            query=search_request.query,
            answer=None,
            results=[],
            response_time=0.0,
            follow_up_questions=None
        )


@tool
def search_banking_products(
    user_query: str,
    product_type: str,
    user_criteria: Optional[Dict[str, Any]] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    High-level tool to search for banking products with automatic query optimization.
    """
    
    try:
        # Create structured product query
        product_query = BankingProductQuery(
            product_type=product_type,
            user_criteria=user_criteria or {},
            specific_requirements=None,
            location=None
        )
        
        # Generate optimized search queries
        query_generation = generate_banking_search_queries.invoke({
            "user_query": user_query,
            "product_query": product_query
        })
        
        all_results = []
        
        # Execute primary search
        primary_request = TavilySearchRequest(
            query=query_generation.primary_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_domains=["bankofamerica.com"] if query_generation.search_strategy == "focused" else None
        )
        
        primary_response = execute_tavily_search.invoke({"search_request": primary_request})
        all_results.extend(primary_response.results)
        
        # Execute secondary searches if needed
        for secondary_query in query_generation.secondary_queries[:2]:  # Limit to 2 additional searches
            secondary_request = TavilySearchRequest(
                query=secondary_query,
                search_depth="basic",
                max_results=3,
                include_answer=False
            )
            
            secondary_response = execute_tavily_search.invoke({"search_request": secondary_request})
            all_results.extend(secondary_response.results)
        
        # Combine and deduplicate results
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            if result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return {
            "query": user_query,
            "product_type": product_type,
            "search_strategy": query_generation.search_strategy,
            "primary_query": query_generation.primary_query,
            "ai_answer": primary_response.answer,
            "results": unique_results[:max_results],
            "total_results_found": len(unique_results),
            "search_time": primary_response.response_time,
            "follow_up_questions": primary_response.follow_up_questions
        }
        
    except Exception as e:
        return {
            "query": user_query,
            "product_type": product_type,
            "error": f"Banking product search failed: {e}",
            "results": [],
            "total_results_found": 0
        }


@tool
def search_bank_policies_and_services(
    user_query: str,
    focus_area: str = "general",
    include_rates: bool = True
) -> Dict[str, Any]:
    """
    Search for general bank policies, services, and information.
    """
    
    # Define focus-specific search queries
    focus_queries = {
        "general": "Bank of America services policies customer benefits",
        "rates": "Bank of America current interest rates savings checking CD",
        "fees": "Bank of America fees structure ATM overdraft monthly maintenance",
        "policies": "Bank of America policies terms conditions customer rights",
        "locations": "Bank of America branch locations ATM network services",
        "digital": "Bank of America mobile app online banking digital services"
    }
    
    base_query = focus_queries.get(focus_area, focus_queries["general"])
    
    # Enhance with user query context
    enhanced_query = f"{base_query} {user_query}"
    
    try:
        search_request = TavilySearchRequest(
            query=enhanced_query,
            search_depth="advanced",
            max_results=8,
            include_answer=True,
            include_domains=["bankofamerica.com", "merrillynch.com"]
        )
        
        response = execute_tavily_search.invoke({"search_request": search_request})
        
        return {
            "query": user_query,
            "focus_area": focus_area,
            "search_query": enhanced_query,
            "ai_summary": response.answer,
            "results": response.results,
            "follow_up_questions": response.follow_up_questions,
            "search_time": response.response_time
        }
        
    except Exception as e:
        return {
            "query": user_query,
            "focus_area": focus_area,
            "error": f"Policy search failed: {e}",
            "results": [],
            "ai_summary": None
        }


# Export main components
__all__ = [
    "TavilySearchRequest",
    "TavilySearchResponse", 
    "TavilySearchResult",
    "BankingProductQuery",
    "SearchQueryGeneration",
    "TavilyClient",
    "generate_banking_search_queries",
    "execute_tavily_search",
    "search_banking_products",
    "search_bank_policies_and_services"
]