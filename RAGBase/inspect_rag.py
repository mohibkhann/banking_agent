import os
from typing import Dict, Any, List
import time

import pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class PineconeIndexInspector:
    """
    Inspector to check what's already stored in the Pinecone index
    without re-ingesting documents.
    """
    
    def __init__(self, index_name: str = "banking-agent"):
        self.index_name = index_name
        
        # Initialize OpenAI embeddings (same as used for ingestion)
        self.embeddings = self._initialize_openai()
        
        # Initialize Pinecone
        self.pinecone_client = self._initialize_pinecone()
        
        print(f"🔍 Pinecone Index Inspector")
        print(f"🌲 Index: {self.index_name}")
        print("=" * 50)

    def _initialize_openai(self) -> OpenAIEmbeddings:
        """Initialize OpenAI embeddings."""
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            dimensions=1024
        )

    def _initialize_pinecone(self) -> Pinecone:
        """Initialize Pinecone client."""
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        return Pinecone(api_key=api_key)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the Pinecone index."""
        
        print("\n📊 INDEX STATISTICS")
        print("-" * 30)
        
        try:
            index = self.pinecone_client.Index(self.index_name)
            stats = index.describe_index_stats()
            
            print(f"✅ Index Status: Active")
            print(f"📈 Total Vectors: {stats.total_vector_count:,}")
            print(f"📏 Vector Dimension: {stats.dimension}")
            
            # Show namespace stats if any
            if hasattr(stats, 'namespaces') and stats.namespaces:
                print(f"📂 Namespaces: {len(stats.namespaces)}")
                for namespace, ns_stats in stats.namespaces.items():
                    namespace_name = namespace if namespace else "default"
                    print(f"   └── {namespace_name}: {ns_stats.vector_count:,} vectors")
            else:
                print(f"📂 Namespaces: Using default namespace")
            
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'namespaces': getattr(stats, 'namespaces', {}),
                'status': 'active'
            }
            
        except Exception as e:
            print(f"❌ Failed to get index stats: {e}")
            return {'error': str(e)}

    def test_searches(self, queries: List[str] = None) -> Dict[str, Any]:
        """Test various search queries to see what's retrievable."""
        
        if queries is None:
            queries = [
                "credit card",
                "credit cards",
                "GX credit card", 
                "grocery shopping",
                "bank",
                "GX bank",
                "loan",
                "mortgage",
                "home loan",
                "auto loan",
                "financial services",
                "banking products"
            ]
        
        print(f"\n🔍 TESTING SEARCH QUERIES")
        print("-" * 40)
        
        try:
            # Initialize vector store
            vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            
            search_results = {}
            
            for i, query in enumerate(queries, 1):
                print(f"\n🔍 Query {i}: '{query}'")
                
                try:
                    results = vectorstore.similarity_search(query, k=3)
                    print(f"   ✅ Found: {len(results)} chunks")
                    
                    search_results[query] = {
                        'results_count': len(results),
                        'chunks': []
                    }
                    
                    # Show brief preview of each result
                    for j, doc in enumerate(results, 1):
                        chunk_info = {
                            'rank': j,
                            'content_preview': doc.page_content[:100] + "...",
                            'filename': doc.metadata.get('filename', 'unknown'),
                            'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                            'source': doc.metadata.get('source', 'unknown')
                        }
                        search_results[query]['chunks'].append(chunk_info)
                        
                        print(f"     {j}. File: {chunk_info['filename']} | Preview: {chunk_info['content_preview']}")
                
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
                    search_results[query] = {'error': str(e)}
                
                # Small delay between searches
                time.sleep(0.2)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Failed to initialize search: {e}")
            return {'error': str(e)}

    def get_sample_documents(self, num_samples: int = 5) -> Dict[str, Any]:
        """Get sample documents from the index to see actual content."""
        
        print(f"\n📄 SAMPLE DOCUMENTS")
        print("-" * 30)
        
        try:
            # Use a very generic search to get any documents
            vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            
            # Try different generic terms to get samples
            generic_queries = ["the", "and", "of", "a", "bank", "service"]
            all_samples = []
            
            for query in generic_queries:
                try:
                    results = vectorstore.similarity_search(query, k=num_samples)
                    all_samples.extend(results)
                    if len(all_samples) >= num_samples:
                        break
                except:
                    continue
            
            # Remove duplicates based on chunk_id
            unique_samples = []
            seen_chunk_ids = set()
            
            for doc in all_samples[:num_samples]:
                chunk_id = doc.metadata.get('chunk_id', 'unknown')
                if chunk_id not in seen_chunk_ids:
                    unique_samples.append(doc)
                    seen_chunk_ids.add(chunk_id)
            
            sample_results = {
                'total_samples': len(unique_samples),
                'samples': []
            }
            
            for i, doc in enumerate(unique_samples, 1):
                sample_info = {
                    'sample_number': i,
                    'filename': doc.metadata.get('filename', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                    'source': doc.metadata.get('source', 'unknown'),
                    'product_type': doc.metadata.get('product_type', 'unknown'),
                    'content_length': len(doc.page_content),
                    'full_content': doc.page_content,
                    'metadata': doc.metadata
                }
                sample_results['samples'].append(sample_info)
                
                print(f"\n📄 Sample {i}:")
                print(f"   Filename: {sample_info['filename']}")
                print(f"   Source: {sample_info['source']}")
                print(f"   Product Type: {sample_info['product_type']}")
                print(f"   Chunk ID: {sample_info['chunk_id']}")
                print(f"   Content Length: {sample_info['content_length']} chars")
                print(f"   Content:")
                print(f"   {'-' * 50}")
                print(f"   {doc.page_content}")
                print(f"   {'-' * 50}")
            
            return sample_results
            
        except Exception as e:
            print(f"❌ Failed to get sample documents: {e}")
            return {'error': str(e)}

    def run_full_inspection(self) -> Dict[str, Any]:
        """Run complete inspection of the Pinecone index."""
        
        print("🚀 STARTING PINECONE INDEX INSPECTION")
        print("=" * 50)
        
        inspection_results = {}
        
        # Step 1: Get index statistics
        inspection_results['stats'] = self.get_index_stats()
        
        # Step 2: Test search queries
        inspection_results['search_tests'] = self.test_searches()
        
        # Step 3: Get sample documents
        inspection_results['sample_documents'] = self.get_sample_documents()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 INSPECTION SUMMARY")
        print("=" * 50)
        
        stats = inspection_results.get('stats', {})
        search_tests = inspection_results.get('search_tests', {})
        samples = inspection_results.get('sample_documents', {})
        
        print(f"📈 Total Vectors: {stats.get('total_vectors', 'Unknown')}")
        print(f"🔍 Search Queries Tested: {len(search_tests)}")
        print(f"📄 Sample Documents Retrieved: {samples.get('total_samples', 0)}")
        
        # Count successful searches
        successful_searches = sum(1 for result in search_tests.values() 
                                if isinstance(result, dict) and result.get('results_count', 0) > 0)
        print(f"✅ Successful Searches: {successful_searches}/{len(search_tests)}")
        
        if successful_searches == 0:
            print("\n⚠️  WARNING: No search queries returned results!")
            print("   This might indicate:")
            print("   - Documents don't contain expected banking/financial content")
            print("   - Indexing issues")
            print("   - Search query mismatch with document content")
        
        return inspection_results

def main():
    """Main function to run Pinecone inspection."""
    
    try:
        inspector = PineconeIndexInspector(index_name="banking-agent")
        results = inspector.run_full_inspection()
        
        print(f"\n✅ Inspection completed successfully!")
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")

if __name__ == "__main__":
    main()