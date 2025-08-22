import os
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import OrderedDict
import threading

import numpy as np
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, validator
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# Pydantic Models for Semantic Caching
class CacheQuery(BaseModel):
    """Input query for semantic cache operations"""
    
    query_text: str = Field(
        description="The query text to search/cache",
        min_length=1,
        max_length=1000
    )
    intent_type: Optional[str] = Field(
        default=None,
        description="Type of intent: banking_product, policy, service, etc."
    )
    product_type: Optional[str] = Field(
        default=None,
        description="Product type if applicable: credit_card, loan, etc."
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional user context for better matching"
    )
    
    @validator('query_text')
    def normalize_query_text(cls, v):
        return v.strip().lower()


class CacheEntry(BaseModel):
    """Individual cache entry with embeddings and metadata"""
    
    cache_id: str = Field(description="Unique cache entry identifier")
    original_query: str = Field(description="Original query text")
    normalized_query: str = Field(description="Normalized query text")
    query_embedding: List[float] = Field(description="Query embedding vector")
    intent_type: Optional[str] = Field(description="Query intent type")
    product_type: Optional[str] = Field(description="Product type if applicable")
    
    # Cached data
    search_results: List[Dict[str, Any]] = Field(description="Cached search results")
    ai_response: Optional[str] = Field(description="Cached AI response")
    external_data: Dict[str, Any] = Field(description="Additional cached data")
    
    # Metadata
    created_at: datetime = Field(description="Cache creation timestamp")
    last_accessed: datetime = Field(description="Last access timestamp")
    access_count: int = Field(default=1, description="Number of times accessed")
    ttl_hours: int = Field(default=24, description="Time to live in hours")
    similarity_threshold: float = Field(default=0.85, description="Required similarity for match")
    
    # Search metadata
    search_strategy: Optional[str] = Field(description="Search strategy used")
    source_queries: List[str] = Field(default=[], description="Original search queries executed")
    
    class Config:
        # Allow numpy arrays to be serialized
        arbitrary_types_allowed = True
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def update_access(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheSearchResult(BaseModel):
    """Result from semantic cache search"""
    
    found: bool = Field(description="Whether a matching cache entry was found")
    cache_entry: Optional[CacheEntry] = Field(description="Matching cache entry if found")
    similarity_score: float = Field(description="Similarity score of best match")
    search_time_ms: float = Field(description="Search time in milliseconds")
    total_entries_checked: int = Field(description="Total cache entries checked")
    
    class Config:
        arbitrary_types_allowed = True


class CacheStats(BaseModel):
    """Cache performance statistics"""
    
    total_entries: int = Field(description="Total cache entries")
    hit_rate: float = Field(description="Cache hit rate percentage")
    average_similarity: float = Field(description="Average similarity of hits")
    memory_usage_mb: float = Field(description="Estimated memory usage in MB")
    oldest_entry: Optional[datetime] = Field(description="Oldest cache entry timestamp")
    newest_entry: Optional[datetime] = Field(description="Newest cache entry timestamp")
    entries_by_type: Dict[str, int] = Field(description="Entries grouped by intent type")
    expired_entries: int = Field(description="Number of expired entries")


# Semantic Cache Manager
class SemanticCacheManager:
    """
    Advanced semantic caching system with embeddings and intelligent matching.
    Designed for banking/financial queries with context awareness.
    """
    
    def __init__(
        self,
        max_cache_size: int = 1000,
        default_similarity_threshold: float = 0.85,
        embedding_model: str = "text-embedding-3-small",
        enable_persistence: bool = False,
        cache_file_path: str = "semantic_cache.pkl"
    ):
        """
        Initialize semantic cache manager.
        
        Args:
            max_cache_size: Maximum number of cache entries
            default_similarity_threshold: Default cosine similarity threshold
            embedding_model: OpenAI embedding model to use
            enable_persistence: Whether to persist cache to disk
            cache_file_path: Path for cache persistence file
        """
        
        self.max_cache_size = max_cache_size
        self.default_similarity_threshold = default_similarity_threshold
        self.enable_persistence = enable_persistence
        self.cache_file_path = cache_file_path
        
        # Initialize embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            print(f"✅ Initialized embeddings with model: {embedding_model}")
        except Exception as e:
            print(f"⚠️ Failed to initialize embeddings: {e}")
            self.embeddings = None
        
        # Cache storage - OrderedDict for LRU eviction
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.embedding_matrix: Optional[np.ndarray] = None
        self.cache_ids: List[str] = []
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_searches = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load persisted cache if enabled
        if self.enable_persistence:
            self._load_cache_from_disk()
        
        print(f"✅ SemanticCacheManager initialized with {len(self.cache)} entries")
    
    def _generate_cache_id(self, query: str, intent_type: str = None) -> str:
        """Generate unique cache ID"""
        content = f"{query}_{intent_type or 'general'}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query"""
        if not self.embeddings:
            return None
            
        try:
            embedding = self.embeddings.embed_query(query)
            return np.array(embedding)
        except Exception as e:
            print(f"[DEBUG] Embedding generation failed: {e}")
            return None
    
    def _rebuild_embedding_matrix(self):
        """Rebuild the embedding matrix for efficient similarity search"""
        if not self.cache:
            self.embedding_matrix = None
            self.cache_ids = []
            return
        
        embeddings_list = []
        cache_ids = []
        
        for cache_id, entry in self.cache.items():
            if entry.query_embedding:
                embeddings_list.append(entry.query_embedding)
                cache_ids.append(cache_id)
        
        if embeddings_list:
            self.embedding_matrix = np.array(embeddings_list)
            self.cache_ids = cache_ids
        else:
            self.embedding_matrix = None
            self.cache_ids = []
    
    def _evict_lru_entries(self, num_to_evict: int = 1):
        """Evict least recently used entries"""
        with self.lock:
            for _ in range(min(num_to_evict, len(self.cache))):
                if self.cache:
                    evicted_id, evicted_entry = self.cache.popitem(last=False)
                    print(f"[DEBUG] Evicted cache entry: {evicted_id} (accessed {evicted_entry.access_count} times)")
            
            # Rebuild embedding matrix after eviction
            self._rebuild_embedding_matrix()
    
    def _clean_expired_entries(self):
        """Remove expired cache entries"""
        with self.lock:
            expired_ids = []
            for cache_id, entry in self.cache.items():
                if entry.is_expired():
                    expired_ids.append(cache_id)
            
            for cache_id in expired_ids:
                del self.cache[cache_id]
                print(f"[DEBUG] Removed expired cache entry: {cache_id}")
            
            if expired_ids:
                self._rebuild_embedding_matrix()
                print(f"[DEBUG] Cleaned {len(expired_ids)} expired entries")
    
    def search_cache(self, cache_query: CacheQuery) -> CacheSearchResult:
        """
        Search for similar queries in the cache using semantic similarity.
        """
        start_time = datetime.now()
        
        with self.lock:
            self.total_searches += 1
            
            # Clean expired entries periodically
            if self.total_searches % 50 == 0:
                self._clean_expired_entries()
            
            # Check if cache is empty
            if not self.cache or self.embedding_matrix is None:
                self.cache_misses += 1
                return CacheSearchResult(
                    found=False,
                    cache_entry=None,
                    similarity_score=0.0,
                    search_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    total_entries_checked=0
                )
            
            # Get embedding for query
            query_embedding = self._get_query_embedding(cache_query.query_text)
            if query_embedding is None:
                self.cache_misses += 1
                return CacheSearchResult(
                    found=False,
                    cache_entry=None,
                    similarity_score=0.0,
                    search_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    total_entries_checked=0
                )
            
            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embedding_matrix
            )[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            best_cache_id = self.cache_ids[best_idx]
            best_entry = self.cache[best_cache_id]
            
            # Apply contextual filtering
            similarity_threshold = cache_query.user_context.get(
                'similarity_threshold', 
                self.default_similarity_threshold
            ) if cache_query.user_context else self.default_similarity_threshold
            
            # Check intent type matching for additional filtering
            intent_match = True
            if cache_query.intent_type and best_entry.intent_type:
                intent_match = cache_query.intent_type == best_entry.intent_type
            
            # Product type matching
            product_match = True
            if cache_query.product_type and best_entry.product_type:
                product_match = cache_query.product_type == best_entry.product_type
            
            # Determine if we have a valid match
            is_match = (
                best_similarity >= similarity_threshold and
                intent_match and
                product_match and
                not best_entry.is_expired()
            )
            
            search_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if is_match:
                # Update access metadata
                best_entry.update_access()
                
                # Move to end (most recently used)
                self.cache.move_to_end(best_cache_id)
                
                self.cache_hits += 1
                
                print(f"[DEBUG] Cache HIT: similarity={best_similarity:.3f}, entry={best_cache_id[:8]}")
                
                return CacheSearchResult(
                    found=True,
                    cache_entry=best_entry,
                    similarity_score=best_similarity,
                    search_time_ms=search_time_ms,
                    total_entries_checked=len(self.cache)
                )
            else:
                self.cache_misses += 1
                
                print(f"[DEBUG] Cache MISS: best_similarity={best_similarity:.3f}, threshold={similarity_threshold:.3f}")
                
                return CacheSearchResult(
                    found=False,
                    cache_entry=None,
                    similarity_score=best_similarity,
                    search_time_ms=search_time_ms,
                    total_entries_checked=len(self.cache)
                )
    
    def add_to_cache(
        self,
        cache_query: CacheQuery,
        search_results: List[Dict[str, Any]],
        ai_response: Optional[str] = None,
        external_data: Optional[Dict[str, Any]] = None,
        search_strategy: Optional[str] = None,
        source_queries: Optional[List[str]] = None,
        ttl_hours: int = 24
    ) -> str:
        """
        Add a new entry to the semantic cache.
        """
        
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_cache_size:
                self._evict_lru_entries(max(1, len(self.cache) - self.max_cache_size + 1))
            
            # Get embedding
            query_embedding = self._get_query_embedding(cache_query.query_text)
            if query_embedding is None:
                print("[DEBUG] Cannot add to cache: embedding generation failed")
                return ""
            
            # Generate cache ID
            cache_id = self._generate_cache_id(cache_query.query_text, cache_query.intent_type)
            
            # Create cache entry
            cache_entry = CacheEntry(
                cache_id=cache_id,
                original_query=cache_query.query_text,
                normalized_query=cache_query.query_text.lower().strip(),
                query_embedding=query_embedding.tolist(),
                intent_type=cache_query.intent_type,
                product_type=cache_query.product_type,
                search_results=search_results,
                ai_response=ai_response,
                external_data=external_data or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_hours=ttl_hours,
                search_strategy=search_strategy,
                source_queries=source_queries or []
            )
            
            # Add to cache
            self.cache[cache_id] = cache_entry
            
            # Rebuild embedding matrix
            self._rebuild_embedding_matrix()
            
            # Persist if enabled
            if self.enable_persistence:
                self._save_cache_to_disk()
            
            print(f"[DEBUG] Added to cache: {cache_id[:8]} (total entries: {len(self.cache)})")
            
            return cache_id
    
    def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        
        with self.lock:
            if not self.cache:
                return CacheStats(
                    total_entries=0,
                    hit_rate=0.0,
                    average_similarity=0.0,
                    memory_usage_mb=0.0,
                    oldest_entry=None,
                    newest_entry=None,
                    entries_by_type={},
                    expired_entries=0
                )
            
            # Calculate hit rate
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            # Entry analysis
            entries_by_type = {}
            expired_count = 0
            timestamps = []
            
            for entry in self.cache.values():
                # Group by intent type
                intent_type = entry.intent_type or "unknown"
                entries_by_type[intent_type] = entries_by_type.get(intent_type, 0) + 1
                
                # Check expiration
                if entry.is_expired():
                    expired_count += 1
                
                # Collect timestamps
                timestamps.append(entry.created_at)
            
            # Estimate memory usage (rough calculation)
            estimated_memory = 0
            for entry in self.cache.values():
                estimated_memory += len(json.dumps(entry.search_results, default=str)) * 2  # Rough estimate
                estimated_memory += len(entry.query_embedding) * 8  # Float64
                if entry.ai_response:
                    estimated_memory += len(entry.ai_response) * 2
            
            memory_mb = estimated_memory / (1024 * 1024)
            
            return CacheStats(
                total_entries=len(self.cache),
                hit_rate=hit_rate,
                average_similarity=0.0,  # Would need to track this separately
                memory_usage_mb=memory_mb,
                oldest_entry=min(timestamps) if timestamps else None,
                newest_entry=max(timestamps) if timestamps else None,
                entries_by_type=entries_by_type,
                expired_entries=expired_count
            )
    
    def clear_cache(self, keep_stats: bool = True):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.embedding_matrix = None
            self.cache_ids = []
            
            if not keep_stats:
                self.cache_hits = 0
                self.cache_misses = 0
                self.total_searches = 0
            
            print("[DEBUG] Cache cleared")
    
    def _save_cache_to_disk(self):
        """Save cache to disk for persistence"""
        try:
            cache_data = {
                'cache': dict(self.cache),
                'stats': {
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'total_searches': self.total_searches
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"[DEBUG] Failed to save cache to disk: {e}")
    
    def _load_cache_from_disk(self):
        """Load cache from disk if available"""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Restore cache entries
                for cache_id, entry_dict in cache_data.get('cache', {}).items():
                    if isinstance(entry_dict, dict):
                        try:
                            entry = CacheEntry(**entry_dict)
                            if not entry.is_expired():
                                self.cache[cache_id] = entry
                        except Exception as e:
                            print(f"[DEBUG] Failed to restore cache entry {cache_id}: {e}")
                
                # Restore stats
                stats = cache_data.get('stats', {})
                self.cache_hits = stats.get('cache_hits', 0)
                self.cache_misses = stats.get('cache_misses', 0)
                self.total_searches = stats.get('total_searches', 0)
                
                # Rebuild embedding matrix
                self._rebuild_embedding_matrix()
                
                print(f"[DEBUG] Loaded {len(self.cache)} cache entries from disk")
                
        except Exception as e:
            print(f"[DEBUG] Failed to load cache from disk: {e}")


# Global cache manager instance
_cache_manager: Optional[SemanticCacheManager] = None

def _get_cache_manager() -> SemanticCacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = SemanticCacheManager(
            max_cache_size=1000,
            default_similarity_threshold=0.85,
            enable_persistence=False  # Keep in memory for POC
        )
    return _cache_manager


# Tools for cache operations
@tool
def semantic_cache_search(
    query_text: str,
    intent_type: Optional[str] = None,
    product_type: Optional[str] = None,
    similarity_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Search semantic cache for similar queries.
    """
    
    try:
        cache_manager = _get_cache_manager()
        
        user_context = {}
        if similarity_threshold:
            user_context['similarity_threshold'] = similarity_threshold
        
        cache_query = CacheQuery(
            query_text=query_text,
            intent_type=intent_type,
            product_type=product_type,
            user_context=user_context
        )
        
        result = cache_manager.search_cache(cache_query)
        
        return {
            "cache_hit": result.found,
            "similarity_score": result.similarity_score,
            "search_time_ms": result.search_time_ms,
            "cached_data": {
                "search_results": result.cache_entry.search_results if result.cache_entry else [],
                "ai_response": result.cache_entry.ai_response if result.cache_entry else None,
                "external_data": result.cache_entry.external_data if result.cache_entry else {},
                "search_strategy": result.cache_entry.search_strategy if result.cache_entry else None,
                "source_queries": result.cache_entry.source_queries if result.cache_entry else []
            } if result.found else None,
            "cache_metadata": {
                "cache_id": result.cache_entry.cache_id if result.cache_entry else None,
                "created_at": result.cache_entry.created_at.isoformat() if result.cache_entry else None,
                "access_count": result.cache_entry.access_count if result.cache_entry else 0
            } if result.found else None
        }
        
    except Exception as e:
        return {
            "cache_hit": False,
            "error": f"Cache search failed: {e}",
            "similarity_score": 0.0,
            "search_time_ms": 0.0,
            "cached_data": None
        }


@tool
def semantic_cache_add(
    query_text: str,
    search_results: List[Dict[str, Any]],
    ai_response: Optional[str] = None,
    external_data: Optional[Dict[str, Any]] = None,
    intent_type: Optional[str] = None,
    product_type: Optional[str] = None,
    search_strategy: Optional[str] = None,
    source_queries: Optional[List[str]] = None,
    ttl_hours: int = 24
) -> Dict[str, Any]:
    """
    Add new entry to semantic cache.
    """
    
    try:
        cache_manager = _get_cache_manager()
        
        cache_query = CacheQuery(
            query_text=query_text,
            intent_type=intent_type,
            product_type=product_type
        )
        
        cache_id = cache_manager.add_to_cache(
            cache_query=cache_query,
            search_results=search_results,
            ai_response=ai_response,
            external_data=external_data or {},
            search_strategy=search_strategy,
            source_queries=source_queries or [],
            ttl_hours=ttl_hours
        )
        
        return {
            "success": bool(cache_id),
            "cache_id": cache_id,
            "message": f"Added to cache with ID: {cache_id[:8]}" if cache_id else "Failed to add to cache"
        }
        
    except Exception as e:
        return {
            "success": False,
            "cache_id": None,
            "error": f"Failed to add to cache: {e}"
        }


@tool
def get_semantic_cache_stats() -> Dict[str, Any]:
    """
    Get comprehensive semantic cache statistics.
    """
    
    try:
        cache_manager = _get_cache_manager()
        stats = cache_manager.get_cache_stats()
        
        return {
            "total_entries": stats.total_entries,
            "hit_rate_percentage": stats.hit_rate,
            "memory_usage_mb": stats.memory_usage_mb,
            "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
            "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
            "entries_by_type": stats.entries_by_type,
            "expired_entries": stats.expired_entries,
            "cache_performance": {
                "hits": cache_manager.cache_hits,
                "misses": cache_manager.cache_misses,
                "total_searches": cache_manager.total_searches
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get cache stats: {e}",
            "total_entries": 0,
            "hit_rate_percentage": 0.0
        }


# Export main components
__all__ = [
    "CacheQuery",
    "CacheEntry", 
    "CacheSearchResult",
    "CacheStats",
    "SemanticCacheManager",
    "semantic_cache_search",
    "semantic_cache_add",
    "get_semantic_cache_stats"
]