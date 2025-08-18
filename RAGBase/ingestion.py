import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class GXBankDocumentIngestion:
    """
    Document ingestion system for GX Bank knowledge base.
    Processes markdown documents and stores them in Pinecone vector store using OpenAI embeddings.
    """
    
    def __init__(
        self,
        docs_directory: str = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/rag_knwledge_base",
        index_name: str = "banking-agent",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.docs_directory = Path(docs_directory)
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize OpenAI embeddings
        self.embeddings = self._initialize_openai()
        self.embedding_dimension = 1024  # Match existing Pinecone index
        
        # Initialize Pinecone
        self.pinecone_client = self._initialize_pinecone()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Document metadata tracking
        self.processed_docs = {}
        self.metadata_file = Path("./document_metadata.json")
        
        print(f"📁 Docs directory: {self.docs_directory}")
        print(f"🌲 Pinecone index: {self.index_name}")
        print(f"🤖 Embedding model: OpenAI text-embedding-3-small (1024d)")
        print(f"📏 Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    def _initialize_openai(self) -> OpenAIEmbeddings:
        """Initialize OpenAI embeddings using OpenAI API key."""
        
        print("🔧 Initializing OpenAI embeddings...")
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # Use 3-small with custom dimensions
                openai_api_key=api_key,
                dimensions=1024  # Reduce to match existing index
            )
            
            print("✅ OpenAI embeddings initialized successfully")
            print("📍 Model: text-embedding-3-small (1024 dimensions)")
            return embeddings
            
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI embeddings: {e}")
            raise

    def _initialize_pinecone(self) -> Pinecone:
        """Initialize Pinecone client and ensure index exists."""
        
        print("🌲 Initializing Pinecone...")
        
        # Check for Pinecone API key
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists
            existing_indexes = pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                print(f"📝 Creating new Pinecone index: {self.index_name}")
                
                # Create index with appropriate dimensions
                pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                print("⏳ Waiting for index to be ready...")
                while not pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                print(f"✅ Index {self.index_name} created successfully")
            else:
                print(f"✅ Using existing Pinecone index: {self.index_name}")
            
            return pc
            
        except Exception as e:
            print(f"❌ Failed to initialize Pinecone: {e}")
            raise

    def load_documents(self) -> List[Document]:
        """Load markdown documents from the knowledge base directory."""
        
        print(f"\n📚 Loading documents from {self.docs_directory}")
        
        if not self.docs_directory.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_directory}")
        
        # Load markdown files
        loader = DirectoryLoader(
            str(self.docs_directory),
            glob="**/*.md",
            show_progress=True
        )
        
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} raw documents")
        
        return documents

    def enhance_document_metadata(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with structured metadata for better retrieval."""
        
        print("\n🏷️  Enhancing document metadata...")
        
        enhanced_docs = []
        
        for doc in documents:
            # Extract filename and path info
            file_path = Path(doc.metadata.get('source', ''))
            filename = file_path.stem
            
            # Initialize enhanced metadata
            enhanced_metadata = {
                **doc.metadata,
                'filename': filename,
                'file_path': str(file_path),
                'ingestion_date': datetime.now().isoformat(),
                'bank': 'GX Bank',
                'document_type': 'product_information'
            }
            
            # Extract metadata from document content
            content_lines = doc.page_content.split('\n')
            
            # Look for metadata section
            in_metadata = False
            for line in content_lines:
                if line.strip() == '## Metadata':
                    in_metadata = True
                    continue
                elif line.startswith('## ') and in_metadata:
                    break
                elif in_metadata and line.strip():
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip('- ').strip()
                        value = value.strip()
                        enhanced_metadata[key.lower().replace(' ', '_')] = value
            
            # Add product-specific metadata based on filename
            if 'home_loan' in filename.lower():
                enhanced_metadata.update({
                    'product_category': 'loans',
                    'product_type': 'home_loans',
                    'target_audience': 'home_buyers',
                    'keywords': ['mortgage', 'home loan', 'property', 'real estate', 'down payment', 'GX Premier', 'FHA', 'VA']
                })
            elif 'auto_loan' in filename.lower():
                enhanced_metadata.update({
                    'product_category': 'loans', 
                    'product_type': 'auto_loans',
                    'target_audience': 'vehicle_buyers',
                    'keywords': ['car loan', 'auto financing', 'vehicle', 'transportation', 'GX Premier Auto', 'refinancing']
                })
            elif 'credit_card' in filename.lower():
                enhanced_metadata.update({
                    'product_category': 'financial_products',
                    'product_type': 'credit_cards',
                    'target_audience': 'all_customers',
                    'keywords': ['credit card', 'rewards', 'cash back', 'points', 'dining', 'grocery', 'GX Silver', 'GX Gold', 'GX Elite', 'GX World']
                })
            
            # Create enhanced document
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            enhanced_docs.append(enhanced_doc)
        
        print(f"✅ Enhanced metadata for {len(enhanced_docs)} documents")
        return enhanced_docs

    def create_document_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into searchable chunks while preserving metadata."""
        
        print(f"\n✂️  Splitting documents into chunks...")
        
        all_chunks = []
        
        for doc in documents:
            # Split the document
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_id': f"{doc.metadata.get('filename', 'unknown')}_{i}",
                    'content_hash': hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
                })
                all_chunks.append(chunk)
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Show chunk statistics
        chunk_sizes = [len(chunk.page_content) for chunk in all_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        print(f"📏 Average chunk size: {avg_size:.0f} characters")
        print(f"📏 Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")
        
        return all_chunks

    def initialize_vector_store(self) -> PineconeVectorStore:
        """Initialize Pinecone vector store."""
        
        print(f"\n🌲 Initializing Pinecone vector store...")
        
        try:
            vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            
            print(f"✅ Initialized Pinecone vector store: {self.index_name}")
            return vectorstore
            
        except Exception as e:
            print(f"❌ Failed to initialize vector store: {e}")
            raise

    def check_for_updates(self, documents: List[Document]) -> tuple[List[Document], bool]:
        """Check if documents have changed since last ingestion."""
        
        print(f"\n🔍 Checking for document updates...")
        
        # Load existing metadata if available
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = {}
        
        new_or_updated_docs = []
        needs_update = False
        
        for doc in documents:
            filename = doc.metadata.get('filename', 'unknown')
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            # Check if document is new or updated
            if (filename not in existing_metadata or 
                existing_metadata[filename].get('content_hash') != content_hash):
                
                new_or_updated_docs.append(doc)
                needs_update = True
                
                # Update metadata tracking
                existing_metadata[filename] = {
                    'content_hash': content_hash,
                    'last_updated': datetime.now().isoformat(),
                    'file_path': doc.metadata.get('file_path')
                }
                
                print(f"📄 Updated: {filename}")
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        if not needs_update:
            print("✅ No documents need updating")
        else:
            print(f"📝 {len(new_or_updated_docs)} documents need processing")
        
        return new_or_updated_docs, needs_update

    def delete_and_recreate_index(self):
        """Delete existing index and recreate with correct dimensions."""
        
        print("🗑️  Deleting existing index to fix dimension mismatch...")
        
        try:
            # Delete the existing index
            self.pinecone_client.delete_index(self.index_name)
            print(f"✅ Deleted index: {self.index_name}")
            
            # Wait for deletion to complete
            print("⏳ Waiting for deletion to complete...")
            time.sleep(10)
            
            # Recreate with correct dimensions
            print(f"📝 Creating new index with {self.embedding_dimension} dimensions...")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for new index to be ready...")
            while not self.pinecone_client.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            print(f"✅ Recreated index {self.index_name} with {self.embedding_dimension} dimensions")
            
        except Exception as e:
            print(f"❌ Failed to recreate index: {e}")
            raise

    def clear_index(self):
        """Clear all vectors from the Pinecone index."""
        
        print("🗑️  Clearing Pinecone index...")
        
        try:
            index = self.pinecone_client.Index(self.index_name)
            
            # Delete all vectors (Pinecone deletes all if no IDs specified)
            index.delete(delete_all=True)
            
            print("✅ Pinecone index cleared successfully")
            
        except Exception as e:
            print(f"⚠️  Could not clear index: {e}")

    def ingest_documents(self, force_reload: bool = False) -> Dict[str, Any]:
        """Complete document ingestion pipeline."""
        
        print("🚀 Starting GX Bank document ingestion pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Load documents
            documents = self.load_documents()
            
            # Step 2: Check for updates (unless forcing reload)
            if not force_reload:
                documents_to_process, needs_update = self.check_for_updates(documents)
                if not needs_update:
                    return {
                        'success': True,
                        'message': 'No updates needed',
                        'documents_processed': 0,
                        'chunks_created': 0,
                        'processing_time': 0
                    }
            else:
                documents_to_process = documents
                print("🔄 Force reload requested - processing all documents")
            
            # Step 3: Enhance metadata
            enhanced_docs = self.enhance_document_metadata(documents_to_process)
            
            # Step 4: Create chunks
            chunks = self.create_document_chunks(enhanced_docs)
            
            # Step 5: Initialize vector store
            vectorstore = self.initialize_vector_store()
            
            # Step 6: Clear existing data if force reload
            if force_reload:
                self.clear_index()
                # Wait a bit for deletion to complete
                time.sleep(2)
            
            # Step 7: Add documents to vector store
            print(f"\n💾 Adding {len(chunks)} chunks to Pinecone...")
            
            # Process in batches to avoid API limits
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # Add batch to Pinecone
                    vectorstore.add_documents(batch)
                    total_added += len(batch)
                    print(f"✅ Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)")
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
                    continue
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create summary
            summary = {
                'success': True,
                'message': 'Document ingestion completed successfully',
                'documents_processed': len(documents_to_process),
                'chunks_created': len(chunks),
                'chunks_stored': total_added,
                'processing_time': processing_time,
                'pinecone_index': self.index_name,
                'embedding_model': 'OpenAI text-embedding-3-small (1024d)',
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
            
            print(f"\n🎉 Ingestion Complete!")
            print(f"📊 Processed: {summary['documents_processed']} documents")
            print(f"📊 Created: {summary['chunks_created']} chunks")
            print(f"📊 Stored: {summary['chunks_stored']} chunks")
            print(f"⏱️  Time: {processing_time:.2f} seconds")
            print(f"🌲 Pinecone Index: {self.index_name}")
            print(f"🤖 Embedding Model: OpenAI text-embedding-3-small (1024d)")
            
            return summary
            
        except Exception as e:
            error_summary = {
                'success': False,
                'error': str(e),
                'message': 'Document ingestion failed',
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            print(f"❌ Ingestion failed: {e}")
            return error_summary

    def test_vector_store(self, query: str = "GX credit cards for grocery shopping") -> Dict[str, Any]:
        """Test the vector store with a sample query."""
        
        print(f"\n🧪 Testing Pinecone vector store with query: '{query}'")
        
        try:
            # Initialize vector store for testing
            vectorstore = self.initialize_vector_store()
            
            # Perform similarity search
            results = vectorstore.similarity_search(
                query,
                k=3  # Return top 3 most similar chunks
            )
            
            print(f"✅ Found {len(results)} relevant chunks")
            
            test_results = {
                'success': True,
                'query': query,
                'results_count': len(results),
                'results': []
            }
            
            for i, doc in enumerate(results, 1):
                result_info = {
                    'rank': i,
                    'content_preview': doc.page_content[:200] + "...",
                    'metadata': {
                        'filename': doc.metadata.get('filename', 'unknown'),
                        'product_type': doc.metadata.get('product_type', 'unknown'),
                        'chunk_id': doc.metadata.get('chunk_id', 'unknown')
                    }
                }
                test_results['results'].append(result_info)
                
                print(f"\n📄 Result {i}:")
                print(f"   File: {result_info['metadata']['filename']}")
                print(f"   Type: {result_info['metadata']['product_type']}")
                print(f"   Preview: {result_info['content_preview']}")
            
            return test_results
            
        except Exception as e:
            print(f"❌ Vector store test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }

def main():
    """Main function to run document ingestion."""
    
    print("=" * 60)
    print("🏦 GX BANK DOCUMENT INGESTION SYSTEM")
    print("📍 OpenAI Embeddings + Pinecone Vector Store")
    print("📍 Using text-embedding-3-small (1024 dimensions)")
    print("=" * 60)
    
    # Configuration
    docs_dir = "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/rag_knwledge_base"
    index_name = "banking-agent"
    
    # Check if documents directory exists
    if not Path(docs_dir).exists():
        print(f"❌ Documents directory not found: {docs_dir}")
        print("Please create the directory and add your markdown files:")
        print(f"  mkdir -p {docs_dir}/financial_products")
        print(f"  # Add home_loans.md, auto_loans.md, credit_cards.md")
        return
    
    # Initialize ingestion system
    try:
        ingestion = GXBankDocumentIngestion(
            docs_directory=docs_dir,
            index_name=index_name,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Run ingestion
        result = ingestion.ingest_documents(force_reload=False)
        
        if result['success']:
            print("\n" + "=" * 60)
            print("✅ INGESTION SUCCESSFUL")
            print("=" * 60)
            
            # Test the vector store
            test_result = ingestion.test_vector_store()
            
            if test_result['success']:
                print("\n✅ Pinecone vector store test passed!")
            else:
                print(f"\n❌ Vector store test failed: {test_result.get('error')}")
        else:
            print("\n" + "=" * 60)
            print("❌ INGESTION FAILED")
            print("=" * 60)
            print(f"Error: {result.get('error')}")
    
    except Exception as e:
        print(f"❌ System initialization failed: {e}")

if __name__ == "__main__":
    main()