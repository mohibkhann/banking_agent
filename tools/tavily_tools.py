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

# Enhanced Search Query Templates for Banking Products with Car Pricing
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
    # ENHANCED: Auto loan templates with car pricing focus
    "auto_loan": {
        "general": "Bank of America auto loan rates {vehicle_type} financing options",
        "specific": "Bank of America auto loan {car_brand} {car_model} financing rates",
        "comparison": "Bank of America auto loan rates vs competitors {credit_score_range}",
        # NEW: Car pricing specific templates
        "car_pricing": "{car_brand} {car_model} {year} price MSRP cost new used",
        "affordability": "{car_brand} {car_model} price financing monthly payment calculator",
        "market_price": "{car_brand} {car_model} {year} market value current price range"
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

# NEW: Car-focused search domains for better pricing results
CAR_PRICING_DOMAINS = [
    "edmunds.com",
    "kbb.com", 
    "cargurus.com",
    "autotrader.com",
    "cars.com",
    "truecar.com",
    "carfax.com"
]

# Enhanced Pydantic models (keeping existing ones and adding new)
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

class BankingProductQuery(BaseModel):
    """Enhanced structured query for banking products with car pricing support"""
    
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
    # NEW: Car-specific fields
    car_details: Optional[Dict[str, str]] = Field(
        default=None,
        description="Car details: brand, model, year for auto loan queries"
    )
    needs_car_pricing: bool = Field(
        default=False,
        description="Whether the query needs actual car pricing information"
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
    """Enhanced generated search queries with car pricing context"""
    
    primary_query: str = Field(description="Main search query")
    secondary_queries: List[str] = Field(
        default=[],
        description="Additional supporting queries"
    )
    car_pricing_queries: List[str] = Field(
        default=[],
        description="Specific car pricing queries if needed"
    )
    search_strategy: str = Field(
        description="Strategy used: focused, broad, comparative, or car_pricing"
    )
    bank_focus: str = Field(
        default="Bank of America",
        description="Primary bank to focus on"
    )
    expected_domains: List[str] = Field(
        default=[],
        description="Expected domains for relevant results"
    )

# Keep existing TavilyClient class unchanged...
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
    
    def search(self, request: TavilySearchRequest):
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
            
            return {
                "query": request.query,
                "follow_up_questions": data.get("follow_up_questions"),
                "answer": data.get("answer"),
                "results": data.get("results", []),
                "response_time": response_time,
                "images": data.get("images")
            }
            
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

@tool
def generate_banking_search_queries(
    user_query: str,
    product_query: BankingProductQuery
) -> SearchQueryGeneration:
    """
    ENHANCED: Generate optimized search queries for banking products with car pricing support.
    """
    
    # Get relevant template
    templates = BANKING_SEARCH_TEMPLATES.get(
        product_query.product_type, 
        BANKING_SEARCH_TEMPLATES["credit_card"]
    )
    
    # Build enhanced context for LLM
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
    
    # NEW: Car-specific context
    if product_query.car_details:
        car_str = ", ".join([f"{k}: {v}" for k, v in product_query.car_details.items()])
        context_parts.append(f"Car Details: {car_str}")
    
    if product_query.needs_car_pricing:
        context_parts.append("PRIORITY: User needs actual car pricing information")
    
    context = "\n".join(context_parts)
    
    # Enhanced query generation prompt with car pricing focus
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at generating search queries for banking products with special focus on car pricing for auto loans.

Generate 1-4 optimized search queries for Tavily to find relevant information.

**CRITICAL CAR PRICING RULES:**
- If query mentions specific car (brand/model) + affordability/financing → MUST include car pricing queries
- For auto loan queries mentioning specific cars → prioritize actual car prices over just loan info
- Car pricing queries should target: Edmunds, KBB, CarGurus, AutoTrader, Cars.com
- Include both new and used car pricing when relevant

**QUERY GENERATION STRATEGY:**

For AUTO LOAN queries mentioning specific cars:
1. **Primary**: Bank of America auto loan rates and terms
2. **Car Pricing**: "[Car Brand] [Car Model] [Year] price MSRP current market value"
3. **Affordability**: "[Car Brand] [Car Model] financing monthly payment calculator"
4. **Comparison** (optional): Auto loan rates comparison

For OTHER banking products:
1. Focus on Bank of America specific information
2. Include user criteria when relevant
3. Make queries specific for actionable information

**ENHANCED TEMPLATES:**
{templates}

**CAR PRICING DOMAINS TO PRIORITIZE:**
- edmunds.com (MSRP, market value)
- kbb.com (Kelley Blue Book values)
- cargurus.com (market pricing)
- autotrader.com (current listings)
- cars.com (pricing data)

**RESPONSE FORMAT:**
Return JSON with:
- primary_query: Main Bank of America query
- secondary_queries: 0-2 additional Bank of America queries  
- car_pricing_queries: 0-2 car pricing specific queries (for auto loans with specific cars)
- search_strategy: "focused", "broad", "comparative", or "car_pricing"

**EXAMPLES:**

User: "Can I afford a Honda Accord with Bank of America auto loan?"
→ primary_query: "Bank of America auto loan rates Honda Accord financing"
→ car_pricing_queries: ["Honda Accord 2024 price MSRP market value", "Honda Accord financing monthly payment calculator"]
→ search_strategy: "car_pricing"

User: "What credit cards does Bank of America offer?"
→ primary_query: "Bank of America credit cards benefits rewards features"
→ car_pricing_queries: []
→ search_strategy: "focused"
"""),
        ("human", """User Context:
{context}

Generate optimized search queries for this banking inquiry. If it involves specific cars and affordability, prioritize car pricing information.""")
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
            # Enhanced fallback for car pricing queries
            response_data = _generate_enhanced_fallback_queries(product_query, user_query)
        
        # Determine expected domains based on search strategy
        expected_domains = ["bankofamerica.com", "merrillynch.com"]
        if response_data.get("search_strategy") == "car_pricing":
            expected_domains.extend(CAR_PRICING_DOMAINS)
        
        return SearchQueryGeneration(
            primary_query=response_data.get("primary_query", ""),
            secondary_queries=response_data.get("secondary_queries", []),
            car_pricing_queries=response_data.get("car_pricing_queries", []),
            search_strategy=response_data.get("search_strategy", "focused"),
            bank_focus="Bank of America",
            expected_domains=expected_domains
        )
        
    except Exception as e:
        print(f"[DEBUG] LLM query generation failed: {e}, using enhanced fallback")
        fallback_data = _generate_enhanced_fallback_queries(product_query, user_query)
        return SearchQueryGeneration(
            primary_query=fallback_data["primary_query"],
            secondary_queries=fallback_data.get("secondary_queries", []),
            car_pricing_queries=fallback_data.get("car_pricing_queries", []),
            search_strategy=fallback_data.get("search_strategy", "focused"),
            bank_focus="Bank of America",
            expected_domains=["bankofamerica.com"] + (CAR_PRICING_DOMAINS if fallback_data.get("search_strategy") == "car_pricing" else [])
        )

def _generate_enhanced_fallback_queries(
    product_query: BankingProductQuery, 
    user_query: str
) -> Dict[str, Any]:
    """Enhanced fallback query generation with car pricing support"""
    
    templates = BANKING_SEARCH_TEMPLATES.get(product_query.product_type, {})
    general_template = templates.get("general", "Bank of America {product_type}")
    
    # Build criteria string
    criteria = ""
    if product_query.user_criteria:
        criteria = " ".join([str(v) for v in product_query.user_criteria.values()])
    
    primary_query = general_template.format(
        criteria=criteria,
        product_type=product_query.product_type.replace("_", " ")
    )
    
    # Enhanced car pricing logic
    car_pricing_queries = []
    search_strategy = "focused"
    
    if product_query.product_type == "auto_loan":
        user_lower = user_query.lower()
        
        # Extract car details from query
        car_brands = ["honda", "toyota", "ford", "nissan", "chevrolet", "bmw", "mercedes", "audi"]
        car_models = ["accord", "camry", "fusion", "altima", "malibu", "civic", "corolla", "escape"]
        
        detected_brand = None
        detected_model = None
        
        for brand in car_brands:
            if brand in user_lower:
                detected_brand = brand.title()
                break
        
        for model in car_models:
            if model in user_lower:
                detected_model = model.title()
                break
        
        # If specific car mentioned + affordability context
        if (detected_brand or detected_model) and any(word in user_lower for word in ["afford", "price", "cost", "monthly", "payment"]):
            search_strategy = "car_pricing"
            
            if detected_brand and detected_model:
                car_pricing_queries = [
                    f"{detected_brand} {detected_model} 2024 price MSRP market value",
                    f"{detected_brand} {detected_model} financing monthly payment calculator"
                ]
            elif detected_brand:
                car_pricing_queries = [
                    f"{detected_brand} 2024 price range MSRP market value",
                    f"{detected_brand} auto loan financing rates"
                ]
    
    return {
        "primary_query": primary_query,
        "secondary_queries": [],
        "car_pricing_queries": car_pricing_queries,
        "search_strategy": search_strategy
    }

@tool
def execute_tavily_search(
    search_request: TavilySearchRequest
):
    """Execute a Tavily search with structured request/response handling."""
    
    try:
        client = _get_tavily_client()
        
        print(f"[DEBUG] Executing Tavily search: {search_request.query[:60]}...")
        
        response = client.search(search_request)
        
        print(f"[DEBUG] Tavily search completed: {len(response.get('results', []))} results in {response.get('response_time', 0):.2f}s")
        
        return response
        
    except Exception as e:
        print(f"[DEBUG] Tavily search failed: {e}")
        return {
            "query": search_request.query,
            "answer": None,
            "results": [],
            "response_time": 0.0,
            "follow_up_questions": None
        }

@tool
def search_banking_products(
    user_query: str,
    product_type: str,
    user_criteria: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
    car_details: Optional[Dict[str, str]] = None,
    needs_car_pricing: bool = False
) -> Dict[str, Any]:
    """
    ENHANCED: High-level tool to search for banking products with automatic car pricing support.
    """
    
    try:
        # Create enhanced structured product query
        product_query = BankingProductQuery(
            product_type=product_type,
            user_criteria=user_criteria or {},
            specific_requirements=None,
            location=None,
            car_details=car_details,
            needs_car_pricing=needs_car_pricing
        )
        
        # Auto-detect car pricing needs
        if product_type == "auto_loan" and not needs_car_pricing:
            user_lower = user_query.lower()
            car_indicators = ["honda", "toyota", "ford", "accord", "camry", "civic", "corolla"]
            affordability_indicators = ["afford", "price", "cost", "monthly", "payment"]
            
            if (any(car in user_lower for car in car_indicators) and 
                any(afford in user_lower for afford in affordability_indicators)):
                product_query.needs_car_pricing = True
                print(f"[DEBUG] Auto-detected car pricing needs for query: {user_query}")
        
        # Generate optimized search queries
        query_generation = generate_banking_search_queries.invoke({
            "user_query": user_query,
            "product_query": product_query
        })
        
        all_results = []
        
        # Execute primary search (Bank of America focus)
        primary_request = TavilySearchRequest(
            query=query_generation.primary_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_domains=["bankofamerica.com"] if query_generation.search_strategy != "car_pricing" else None
        )
        
        primary_response = execute_tavily_search.invoke({"search_request": primary_request})
        all_results.extend(primary_response.get('results', []))
        
        # Execute car pricing searches if needed
        if query_generation.car_pricing_queries:
            print(f"[DEBUG] Executing {len(query_generation.car_pricing_queries)} car pricing searches...")
            
            for car_query in query_generation.car_pricing_queries:
                car_request = TavilySearchRequest(
                    query=car_query,
                    search_depth="advanced",
                    max_results=3,
                    include_answer=True,
                    include_domains=CAR_PRICING_DOMAINS  # Focus on car pricing sites
                )
                
                car_response = execute_tavily_search.invoke({"search_request": car_request})
                all_results.extend(car_response.get('results', []))
        
        # Execute secondary searches
        for secondary_query in query_generation.secondary_queries[:2]:  # Limit to 2 additional
            secondary_request = TavilySearchRequest(
                query=secondary_query,
                search_depth="basic",
                max_results=3,
                include_answer=False
            )
            
            secondary_response = execute_tavily_search.invoke({"search_request": secondary_request})
            all_results.extend(secondary_response.get('results', []))
        
        # Combine and deduplicate results
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url)
        
        # Sort by relevance score if available
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return {
            "query": user_query,
            "product_type": product_type,
            "search_strategy": query_generation.search_strategy,
            "primary_query": query_generation.primary_query,
            "car_pricing_queries": query_generation.car_pricing_queries,
            "ai_answer": primary_response.get('answer'),
            "results": unique_results[:max_results * 2],  # Allow more results for car pricing
            "total_results_found": len(unique_results),
            "search_time": primary_response.get('response_time', 0),
            "follow_up_questions": primary_response.get('follow_up_questions'),
            "has_car_pricing_data": len(query_generation.car_pricing_queries) > 0
        }
        
    except Exception as e:
        return {
            "query": user_query,
            "product_type": product_type,
            "error": f"Enhanced banking product search failed: {e}",
            "results": [],
            "total_results_found": 0
        }

# Keep existing functions unchanged...
@tool
def search_bank_policies_and_services(
    user_query: str,
    focus_area: str = "general",
    include_rates: bool = True
) -> Dict[str, Any]:
    """Search for general bank policies, services, and information."""
    
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
            "ai_summary": response.get('answer'),
            "results": response.get('results', []),
            "follow_up_questions": response.get('follow_up_questions'),
            "search_time": response.get('response_time', 0)
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
    "BankingProductQuery", 
    "SearchQueryGeneration",
    "TavilyClient",
    "generate_banking_search_queries",
    "execute_tavily_search",
    "search_banking_products",
    "search_bank_policies_and_services"
]