# .the sole purpose of this script is to make sure the "GX Bank Identity"


import os
import json
import re
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import threading

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from pydantic import BaseModel, Field, validator

load_dotenv()


# Pydantic Models for Brand Translation
class BrandTranslationRequest(BaseModel):
    """Request for brand translation with context"""
    
    content: str = Field(
        description="Content to translate from Bank of America to GX Bank",
        min_length=1
    )
    content_type: str = Field(
        default="general",
        description="Type of content: product_description, policy, rates, general"
    )
    preserve_accuracy: bool = Field(
        default=True,
        description="Whether to preserve factual accuracy (rates, terms, etc.)"
    )
    target_tone: str = Field(
        default="professional",
        description="Target tone: professional, friendly, casual, formal"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for better translation"
    )
    
    @validator('content_type')
    def validate_content_type(cls, v):
        valid_types = ["product_description", "policy", "rates", "general", "marketing", "terms"]
        if v not in valid_types:
            raise ValueError(f"content_type must be one of {valid_types}")
        return v
    
    @validator('target_tone')
    def validate_tone(cls, v):
        valid_tones = ["professional", "friendly", "casual", "formal", "approachable"]
        if v not in valid_tones:
            raise ValueError(f"target_tone must be one of {valid_tones}")
        return v


class BrandTranslationResult(BaseModel):
    """Result of brand translation"""
    
    original_content: str = Field(description="Original Bank of America content")
    translated_content: str = Field(description="Translated GX Bank content")
    changes_made: List[str] = Field(description="List of changes made during translation")
    preserved_elements: List[str] = Field(description="Critical elements preserved (rates, terms)")
    confidence_score: float = Field(description="Translation confidence (0-1)", ge=0.0, le=1.0)
    content_type: str = Field(description="Type of content translated")
    translation_strategy: str = Field(description="Strategy used for translation")


class BrandTranslationCache(BaseModel):
    """Cached translation for performance"""
    
    content_hash: str = Field(description="Hash of original content")
    translated_content: str = Field(description="Cached translation")
    content_type: str = Field(description="Content type")
    created_at: str = Field(description="Cache creation timestamp")
    hit_count: int = Field(default=1, description="Number of cache hits")


# Smart Brand Translator Class
class SmartBrandTranslator:
    """
    AI-powered brand translator that intelligently converts Bank of America
    content to GX Bank branding while preserving accuracy and context.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        cache_size: int = 500,
        enable_caching: bool = True
    ):
        """
        Initialize smart brand translator.
        
        Args:
            model_name: OpenAI model for translation
            cache_size: Maximum cache entries
            enable_caching: Whether to cache translations
        """
        
        self.model_name = model_name
        self.cache_size = cache_size
        self.enable_caching = enable_caching
        
        # Initialize LLM
        try:

            self.llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )

            print(f"✅ Initialized brand translator with {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to initialize LLM: {e}")
            self.llm = None
        
        # Translation cache
        self.translation_cache: Dict[str, BrandTranslationCache] = {}
        self.cache_lock = threading.RLock()
        
        # Core translation prompt
        self.translation_prompt = self._build_translation_prompt()
        
    def _build_translation_prompt(self) -> ChatPromptTemplate:
        """Build the core translation prompt template"""
        
        return ChatPromptTemplate.from_messages([
            ("system", """You are a professional brand translator specializing in financial services content.

Your task is to translate Bank of America content to GX Bank while maintaining:
✅ ACCURACY: All rates, terms, fees, and factual information
✅ CONTEXT: Financial meaning and regulatory compliance  
✅ TONE: Professional banking language appropriate for {target_tone} communication

TRANSLATION RULES:

🏦 BRAND REPLACEMENTS:
- "Bank of America" → "GX Bank"
- "BofA" → "GX Bank"  
- "Merrill Lynch" → "GX Investment Services"
- Any BoA-specific product names → equivalent GX names

🔒 PRESERVE EXACTLY:
- Interest rates, APRs, fees (keep all numbers)
- Legal terms and conditions
- Regulatory language and disclaimers
- Account requirements and minimums
- Contact information (convert to GX format)

💡 SMART ADAPTATION:
- Maintain the original structure and flow
- Keep financial accuracy paramount
- Adapt marketing language to GX Bank's voice
- Preserve all substantive content

CONTENT TYPE: {content_type}
TONE TARGET: {target_tone}

Respond with JSON:
{{
    "translated_content": "The translated content with GX Bank branding",
    "changes_made": ["List of specific changes made"],
    "preserved_elements": ["Critical elements kept exactly as-is"],
    "confidence_score": 0.95,
    "translation_strategy": "Description of approach used"
}}

Important: Be thorough but natural. The result should read as if GX Bank originally created this content."""),
            ("human", """Original Bank of America Content:
{content}

Context: {context}

Please translate this to GX Bank branding while following all the rules above.""")
        ])
    
    def _generate_content_hash(self, content: str, content_type: str) -> str:
        """Generate hash for caching"""
        import hashlib
        content_key = f"{content}_{content_type}"
        return hashlib.md5(content_key.encode()).hexdigest()
    
    def _check_cache(self, content_hash: str) -> Optional[BrandTranslationCache]:
        """Check if translation is cached"""
        if not self.enable_caching:
            return None
            
        with self.cache_lock:
            cached = self.translation_cache.get(content_hash)
            if cached:
                cached.hit_count += 1
                return cached
            return None
    
    def _add_to_cache(
        self, 
        content_hash: str, 
        translated_content: str, 
        content_type: str
    ):
        """Add translation to cache"""
        if not self.enable_caching:
            return
            
        with self.cache_lock:
            # Evict oldest if cache full
            if len(self.translation_cache) >= self.cache_size:
                oldest_key = min(
                    self.translation_cache.keys(),
                    key=lambda k: self.translation_cache[k].hit_count
                )
                del self.translation_cache[oldest_key]
            
            self.translation_cache[content_hash] = BrandTranslationCache(
                content_hash=content_hash,
                translated_content=translated_content,
                content_type=content_type,
                created_at=json.dumps({"timestamp": "now"}),  # Simplified for POC
                hit_count=1
            )
    
    def translate(self, request: BrandTranslationRequest) -> BrandTranslationResult:
        """
        Translate content from Bank of America to GX Bank branding.
        """
        
        if not self.llm:
            raise ValueError("LLM not initialized - cannot perform translation")
        
        # Check cache first
        content_hash = self._generate_content_hash(request.content, request.content_type)
        cached = self._check_cache(content_hash)
        
        if cached:
            print(f"[DEBUG] Using cached translation (hit #{cached.hit_count})")
            return BrandTranslationResult(
                original_content=request.content,
                translated_content=cached.translated_content,
                changes_made=["Used cached translation"],
                preserved_elements=["All critical elements from cache"],
                confidence_score=0.98,  # High confidence for cached results
                content_type=request.content_type,
                translation_strategy="cached_result"
            )
        
        # Prepare context
        context_str = ""
        if request.context:
            context_parts = []
            for key, value in request.context.items():
                context_parts.append(f"{key}: {value}")
            context_str = "; ".join(context_parts)
        else:
            context_str = f"Standard {request.content_type} content requiring professional translation"
        
        try:
            # Execute LLM translation
            response = self.llm.invoke(
                self.translation_prompt.format_messages(
                    content=request.content,
                    content_type=request.content_type,
                    target_tone=request.target_tone,
                    context=context_str
                )
            )
            
            # Parse JSON response
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback: extract content between quotes or use full response
                translated_content = self._extract_translation_fallback(response.content)
                result_data = {
                    "translated_content": translated_content,
                    "changes_made": ["Applied basic brand name substitution"],
                    "preserved_elements": ["Attempted to preserve all factual content"],
                    "confidence_score": 0.7,
                    "translation_strategy": "fallback_extraction"
                }
            
            # Validate translation result
            translated_content = result_data.get("translated_content", "")
            if not translated_content or len(translated_content) < 10:
                raise ValueError("Translation result too short or empty")
            
            # Cache the result
            self._add_to_cache(content_hash, translated_content, request.content_type)
            
            # Create result
            return BrandTranslationResult(
                original_content=request.content,
                translated_content=translated_content,
                changes_made=result_data.get("changes_made", []),
                preserved_elements=result_data.get("preserved_elements", []),
                confidence_score=result_data.get("confidence_score", 0.85),
                content_type=request.content_type,
                translation_strategy=result_data.get("translation_strategy", "llm_based")
            )
            
        except Exception as e:
            print(f"[DEBUG] LLM translation failed: {e}")
            # Emergency fallback - simple regex replacement
            return self._emergency_fallback_translation(request)
    
    def _extract_translation_fallback(self, response_content: str) -> str:
        """Extract translation from malformed LLM response"""
        
        # Try to find content in quotes
        quote_patterns = [
            r'"translated_content":\s*"([^"]+)"',
            r'"([^"]*GX Bank[^"]*)"',
            r'translated[^:]*:\s*"([^"]+)"'
        ]
        
        for pattern in quote_patterns:
            match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        
        # If no quotes found, do basic replacement on the full response
        return self._basic_brand_replacement(response_content)
    
    def _basic_brand_replacement(self, content: str) -> str:
        """Basic brand name replacement as emergency fallback"""
        
        replacements = {
            r'\bBank of America\b': 'GX Bank',
            r'\bBofA\b': 'GX Bank',
            r'\bBoA\b': 'GX Bank',
            r'\bMerrill Lynch\b': 'GX Investment Services',
            r'\bMerrill\b': 'GX Investment',
            r'we at Bank of America': 'we at GX Bank',
            r'Bank of America offers': 'GX Bank offers',
            r'Bank of America provides': 'GX Bank provides'
        }
        
        result = content
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _emergency_fallback_translation(self, request: BrandTranslationRequest) -> BrandTranslationResult:
        """Emergency fallback when LLM fails"""
        
        translated_content = self._basic_brand_replacement(request.content)
        
        return BrandTranslationResult(
            original_content=request.content,
            translated_content=translated_content,
            changes_made=["Emergency fallback: basic brand name replacement"],
            preserved_elements=["All content preserved with minimal changes"],
            confidence_score=0.6,  # Lower confidence for fallback
            content_type=request.content_type,
            translation_strategy="emergency_fallback"
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get translation cache statistics"""
        
        with self.cache_lock:
            total_hits = sum(cache.hit_count for cache in self.translation_cache.values())
            
            return {
                "total_cached_translations": len(self.translation_cache),
                "total_cache_hits": total_hits,
                "cache_efficiency": total_hits / max(1, len(self.translation_cache)),
                "cache_size_limit": self.cache_size,
                "most_used_translation": max(
                    self.translation_cache.values(),
                    key=lambda x: x.hit_count,
                    default=None
                )
            }


# Global translator instance
_brand_translator: Optional[SmartBrandTranslator] = None

def _get_brand_translator() -> SmartBrandTranslator:
    """Get or create global brand translator instance"""
    global _brand_translator
    if _brand_translator is None:
        _brand_translator = SmartBrandTranslator()
    return _brand_translator


# Translation Tools
@tool
def translate_bank_content(
    content: str,
    content_type: str = "general",
    target_tone: str = "professional",
    preserve_accuracy: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Intelligently translate Bank of America content to GX Bank branding.
    
    Args:
        content: The content to translate
        content_type: Type of content (product_description, policy, rates, general)
        target_tone: Desired tone (professional, friendly, casual, formal)
        preserve_accuracy: Whether to preserve exact factual information
        context: Additional context for better translation
    """
    
    try:
        translator = _get_brand_translator()
        
        request = BrandTranslationRequest(
            content=content,
            content_type=content_type,
            preserve_accuracy=preserve_accuracy,
            target_tone=target_tone,
            context=context
        )
        
        result = translator.translate(request)
        
        return {
            "success": True,
            "original_content": result.original_content,
            "translated_content": result.translated_content,
            "changes_made": result.changes_made,
            "preserved_elements": result.preserved_elements,
            "confidence_score": result.confidence_score,
            "translation_strategy": result.translation_strategy,
            "content_type": result.content_type
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Translation failed: {e}",
            "original_content": content,
            "translated_content": content,  # Return original as fallback
            "confidence_score": 0.0
        }


@tool
def batch_translate_content(
    content_list: List[Dict[str, Any]],
    default_content_type: str = "general",
    default_tone: str = "professional"
) -> Dict[str, Any]:
    """
    Translate multiple pieces of content efficiently.
    
    Args:
        content_list: List of dicts with 'content' and optional 'content_type', 'tone'
        default_content_type: Default content type for items without one
        default_tone: Default tone for items without one
    """
    
    try:
        translator = _get_brand_translator()
        results = []
        
        for i, item in enumerate(content_list):
            if not isinstance(item, dict) or 'content' not in item:
                results.append({
                    "index": i,
                    "success": False,
                    "error": "Invalid item format - must have 'content' key"
                })
                continue
            
            request = BrandTranslationRequest(
                content=item['content'],
                content_type=item.get('content_type', default_content_type),
                target_tone=item.get('tone', default_tone),
                preserve_accuracy=item.get('preserve_accuracy', True),
                context=item.get('context')
            )
            
            try:
                result = translator.translate(request)
                results.append({
                    "index": i,
                    "success": True,
                    "original_content": result.original_content,
                    "translated_content": result.translated_content,
                    "confidence_score": result.confidence_score,
                    "changes_made": result.changes_made
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "original_content": item['content'],
                    "translated_content": item['content']
                })
        
        successful_translations = sum(1 for r in results if r.get('success', False))
        
        return {
            "total_items": len(content_list),
            "successful_translations": successful_translations,
            "success_rate": successful_translations / len(content_list) if content_list else 0,
            "results": results
        }
        
    except Exception as e:
        return {
            "total_items": len(content_list),
            "successful_translations": 0,
            "success_rate": 0.0,
            "error": f"Batch translation failed: {e}",
            "results": []
        }


@tool
def get_translation_cache_stats() -> Dict[str, Any]:
    """Get brand translation cache statistics."""
    
    try:
        translator = _get_brand_translator()
        return translator.get_cache_stats()
    except Exception as e:
        return {
            "error": f"Failed to get cache stats: {e}",
            "total_cached_translations": 0
        }


# Smart helper for automatic content type detection
@lru_cache(maxsize=100)
def detect_content_type(content: str) -> str:
    """Automatically detect content type based on keywords"""
    
    content_lower = content.lower()
    
    # Keywords for different content types
    type_keywords = {
        "rates": ["rate", "apr", "interest", "percentage", "%", "annual"],
        "policy": ["terms", "conditions", "policy", "agreement", "compliance"],
        "product_description": ["card", "account", "loan", "features", "benefits"],
        "marketing": ["offer", "promotion", "limited time", "apply now", "get started"],
        "terms": ["fee", "minimum", "maximum", "requirement", "eligibility"]
    }
    
    # Score each type
    scores = {}
    for content_type, keywords in type_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        scores[content_type] = score
    
    # Return highest scoring type, or 'general' if no clear winner
    if not scores or max(scores.values()) == 0:
        return "general"
    
    return max(scores, key=scores.get)


# Export main components
__all__ = [
    "BrandTranslationRequest",
    "BrandTranslationResult",
    "SmartBrandTranslator",
    "translate_bank_content", 
    "batch_translate_content",
    "get_translation_cache_stats",
    "detect_content_type"
]

