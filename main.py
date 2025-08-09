import os
import asyncio
import logging
import io
import traceback
import re
import time
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import requests
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.messages import SystemMessage, HumanMessage
from urllib.parse import urlparse
import docx2txt

# QdrantDB and hybrid search imports
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, SearchRequest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
from collections import deque
from time import monotonic

# Ollama import for local Gemma model
from langchain_community.chat_models import ChatOllama

# Optional import for MIME type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available. MIME type detection will use fallback methods.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

async def to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

# Enhanced Configuration for Legal Document Analysis
class Config:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    BEARER_TOKEN = "legal_doc_analyzer_token_2024"
    
    # Vector configurations - Updated for Legal Documents
    EMBEDDING_MODEL = "amixh/sentence-embedding-model-InLegalBERT-2"
    COLLECTION_NAME = "indian-legal-documents-hybrid"
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    
    # Hybrid search parameters
    DENSE_WEIGHT = 0.7  # Higher weight for dense vectors in legal context
    SPARSE_WEIGHT = 0.3
    
    # Performance optimizations for legal documents
    MAX_CHUNK_SIZE = 1536  # Larger chunks for legal context
    CHUNK_OVERLAP = 256   # More overlap for legal continuity
    SIMILARITY_THRESHOLD = 0.6  # Higher threshold for legal precision
    TOP_K = 15
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB for legal documents
    REQUEST_TIMEOUT = 45  # Longer timeout for complex legal docs
    MAX_RETRIES = 3
    BATCH_SIZE = 20
    MAX_WORKERS = 4

config = Config()

# Initialize embeddings with Legal BERT model
try:
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info("InLegalBERT-2 embedding model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Legal BERT model: {str(e)}")
    raise

security = HTTPBearer()

# Pydantic Models for Legal Analysis
class LegalQueryRequest(BaseModel):
    documents: str = Field(..., description="Comma-separated URLs to legal document blobs", min_length=1)
    questions: List[str] = Field(..., description="List of legal questions to analyze", min_items=1, max_items=20)
    
    @validator('questions')
    def validate_questions(cls, v):
        if not all(question.strip() for question in v):
            raise ValueError("All questions must be non-empty strings")
        return [question.strip() for question in v]
    
    @validator('documents')
    def validate_documents(cls, v):
        urls = [url.strip() for url in v.split(',') if url.strip()]
        if not urls:
            raise ValueError("At least one valid legal document URL must be provided")
        for url in urls:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")
        return v

class LegalQueryResponse(BaseModel):
    legal_analysis: List[str] = Field(..., description="List of legal analysis responses")

# Legal Sparse Vector Generator for Indian Law
class LegalSparseVectorGenerator:
    def __init__(self, method="tfidf"):
        self.method = method
        
        if method == "tfidf":
            # Indian Legal-optimized TF-IDF
            self.vectorizer = TfidfVectorizer(
                vocabulary=self._build_legal_vocabulary(),
                ngram_range=(1, 4),  # Capture legal phrases
                max_features=15000,
                stop_words='english'
            )
            self._init_tfidf()
    
    def _build_legal_vocabulary(self):
        """Build comprehensive Indian legal vocabulary"""
        vocabulary_terms = [
            # Constitutional terms
            'constitution', 'article', 'amendment', 'fundamental', 'rights', 'directive', 'principles',
            'supreme', 'court', 'high', 'subordinate', 'jurisdiction', 'writ', 'petition',
            'habeas', 'corpus', 'mandamus', 'certiorari', 'prohibition', 'quo', 'warranto',
            
            # Criminal law terms
            'indian', 'penal', 'code', 'ipc', 'crpc', 'evidence', 'act', 'section',
            'criminal', 'procedure', 'bail', 'cognizable', 'non-cognizable', 'bailable',
            'murder', 'culpable', 'homicide', 'assault', 'battery', 'theft', 'robbery',
            'dacoity', 'cheating', 'fraud', 'forgery', 'defamation', 'sedition',
            
            # Civil law terms
            'civil', 'procedure', 'contract', 'tort', 'negligence', 'damages', 'injunction',
            'specific', 'performance', 'breach', 'consideration', 'offer', 'acceptance',
            'void', 'voidable', 'illegal', 'unenforceable', 'limitation', 'period',
            
            # Property law
            'property', 'ownership', 'possession', 'title', 'deed', 'sale', 'mortgage',
            'lease', 'rent', 'tenancy', 'easement', 'inheritance', 'succession',
            'registration', 'stamp', 'duty', 'transfer',
            
            # Company and commercial law
            'company', 'companies', 'corporate', 'director', 'shareholder', 'partnership',
            'limited', 'liability', 'memorandum', 'articles', 'association', 'merger',
            'acquisition', 'insolvency', 'bankruptcy', 'liquidation', 'securities',
            
            # Family law
            'marriage', 'divorce', 'maintenance', 'alimony', 'custody', 'adoption',
            'guardianship', 'personal', 'law', 'hindu', 'muslim', 'christian', 'parsi',
            
            # Administrative law
            'administrative', 'tribunal', 'natural', 'justice', 'bias', 'hearing',
            'reasoned', 'order', 'judicial', 'review', 'ultra', 'vires', 'delegated',
            'legislation', 'subordinate', 'rule', 'regulation', 'notification',
            
            # Tax law
            'income', 'tax', 'gst', 'customs', 'excise', 'service', 'assessment',
            'penalty', 'prosecution', 'appeal', 'tribunal', 'commissioner',
            
            # Labor law
            'labour', 'labor', 'employment', 'workman', 'industrial', 'dispute',
            'trade', 'union', 'strike', 'lockout', 'gratuity', 'provident', 'fund',
            'bonus', 'minimum', 'wages', 'factories', 'shops', 'establishments',
            
            # Legal procedures
            'plaintiff', 'defendant', 'petitioner', 'respondent', 'appellant', 'appellee',
            'suit', 'plaint', 'written', 'statement', 'issues', 'evidence', 'witness',
            'examination', 'cross-examination', 'judgment', 'decree', 'order', 'appeal',
            'revision', 'review', 'execution', 'attachment', 'garnishee',
            
            # Legal concepts
            'precedent', 'ratio', 'decidendi', 'obiter', 'dicta', 'stare', 'decisis',
            'res', 'judicata', 'subjudice', 'forum', 'non', 'conveniens', 'locus', 'standi',
            'mens', 'rea', 'actus', 'reus', 'burden', 'proof', 'prima', 'facie',
            
            # Indian specific terms
            'bharatiya', 'nyaya', 'sanhita', 'bhartiya', 'nagarik', 'suraksha', 'sakshya',
            'adhiniyam', 'kanoon', 'vidhan', 'sabha', 'lok', 'sabha', 'rajya',
            'governor', 'president', 'parliament', 'legislature', 'executive', 'judiciary'
        ]

        # Remove duplicates and return unique terms
        return list(set(vocabulary_terms))
    
    def _init_tfidf(self):
        """Initialize TF-IDF with sample legal texts"""
        sample_texts = [
            "constitution article fundamental rights directive principles state policy supreme court high court",
            "indian penal code criminal procedure evidence act section murder culpable homicide theft robbery",
            "civil procedure contract tort negligence damages injunction specific performance breach consideration",
            "property ownership possession title deed sale mortgage lease rent tenancy easement inheritance",
            "company companies act director shareholder partnership limited liability memorandum articles association",
            "marriage divorce maintenance alimony custody adoption guardianship personal law hindu muslim christian",
            "administrative law tribunal natural justice bias hearing reasoned order judicial review ultra vires",
            "income tax gst customs excise service tax assessment penalty prosecution appeal tribunal",
            "labour law employment workman industrial dispute trade union strike lockout gratuity provident fund",
            "plaintiff defendant petitioner respondent suit plaint written statement issues evidence witness"
        ]
        self.vectorizer.fit(sample_texts)
    
    def generate_sparse_vector(self, text: str) -> dict:
        """Generate sparse vector from legal text"""
        if self.method == "tfidf":
            return self._generate_tfidf_sparse(text)
        else:
            return {"indices": [], "values": []}
    
    def _generate_tfidf_sparse(self, text: str) -> dict:
        """Generate TF-IDF sparse vector for legal text"""
        try:
            tfidf_matrix = self.vectorizer.transform([text])
            sparse_array = tfidf_matrix.toarray()[0]
            
            # Convert to sparse format
            indices = []
            values = []
            
            for i, value in enumerate(sparse_array):
                if value > 0:
                    indices.append(i)
                    values.append(float(value))
            
            return {"indices": indices, "values": values}
        except Exception as e:
            logger.error(f"Legal TF-IDF sparse vector generation failed: {str(e)}")
            return {"indices": [], "values": []}

# Enhanced QdrantDB Vector Store for Legal Documents
class LegalHybridVectorStore:
    def __init__(self, url: str, collection_name: str, api_key: Optional[str] = None):
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.collection_name = collection_name
            self.sparse_generator = LegalSparseVectorGenerator(method="tfidf")
            
            # Get embedding dimensions
            test_embedding = embedding_model.encode("test legal document")
            self.dense_dimension = len(test_embedding)
            self.sparse_dimension = 15000  # Legal TF-IDF vocab size
            
            self._create_hybrid_collection()
            self.processed_docs = set()
            logger.info(f"Legal hybrid vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize legal hybrid QdrantDB: {str(e)}")
            raise

    def _create_hybrid_collection(self):
        """Create collection with both dense and sparse vectors for legal docs"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        config.DENSE_VECTOR_NAME: models.VectorParams(
                            size=self.dense_dimension,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }
                )
                logger.info(f"Created legal hybrid collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error creating legal hybrid collection: {str(e)}")
            raise

    def document_exists(self, doc_hash: str) -> bool:
        """Check if legal document exists"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="doc_hash", match=models.MatchValue(value=doc_hash))]
                ),
                limit=1
            )
            return len(result[0]) > 0
        except:
            return doc_hash in self.processed_docs

    async def legal_hybrid_search(self, query: str, top_k: int = config.TOP_K) -> List[Tuple[Document, float]]:
        """Perform legal-optimized hybrid search with sparse + dense vectors"""
        try:
            start_time = time.time()
            
            # Generate query vectors
            dense_query = embedding_model.encode(query).tolist()
            sparse_query_data = self.sparse_generator.generate_sparse_vector(query)
            sparse_query = SparseVector(
                indices=sparse_query_data["indices"],
                values=sparse_query_data["values"]
            )
            
            logger.info(f"🔍 Legal Query: {query}")
            logger.info(f"📊 Dense vector dim: {len(dense_query)}")
            logger.info(f"🏷️  Sparse vector legal terms: {len(sparse_query_data['indices'])}")
            
            # Perform batch search (dense + sparse) for legal documents
            search_requests = [
                SearchRequest(
                    vector=NamedVector(name=config.DENSE_VECTOR_NAME, vector=dense_query),
                    limit=top_k * 2,
                    with_payload=True
                ),
                SearchRequest(
                    vector=NamedSparseVector(name=config.SPARSE_VECTOR_NAME, vector=sparse_query),
                    limit=top_k * 2,
                    with_payload=True
                )
            ]
            
            batch_results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=search_requests
            )
            
            logger.info(f"🔍 Legal batch search returned: {len(batch_results) if batch_results else 0} result sets")
            
            dense_results = batch_results[0] if len(batch_results) > 0 else []
            sparse_results = batch_results[1] if len(batch_results) > 1 else []
            
            # Handle case where results are wrapped in additional structure
            if hasattr(dense_results, 'points'):
                dense_results = dense_results.points
            if hasattr(sparse_results, 'points'):
                sparse_results = sparse_results.points
                
            logger.info(f"📊 Legal dense results: {len(dense_results)}")
            logger.info(f"🏷️  Legal sparse results: {len(sparse_results)}")
            
            # Apply Reciprocal Rank Fusion (RRF) for legal context
            fused_results = self._legal_reciprocal_rank_fusion(dense_results, sparse_results, top_k)
            
            search_time = time.time() - start_time
            logger.info(f"⚡ Legal hybrid search completed in {search_time:.3f}s - {len(fused_results)} results")
            
            # Log top legal results for debugging
            if fused_results:
                logger.info("🎯 Top legal results:")
                for i, (doc, score) in enumerate(fused_results[:3]):
                    logger.info(f"   {i+1}. Legal RRF Score: {score:.3f}, Text preview: {doc.page_content[:150]}...")
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Legal hybrid search failed: {str(e)}")
            return []

    def _legal_reciprocal_rank_fusion(self, dense_results, sparse_results, top_k: int) -> List[Tuple[Document, float]]:
        """Apply Legal-optimized Reciprocal Rank Fusion"""
        try:
            logger.info(f"🔗 Legal RRF fusion: dense={len(dense_results)}, sparse={len(sparse_results)}")
            
            rrf_scores = {}
            k = 60  # RRF parameter
            
            # Process dense results with legal weight
            for rank, result in enumerate(dense_results, 1):
                doc_id = result.id
                score_contribution = config.DENSE_WEIGHT / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score_contribution
                
            # Process sparse results with legal weight
            for rank, result in enumerate(sparse_results, 1):
                doc_id = result.id
                score_contribution = config.SPARSE_WEIGHT / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score_contribution
            
            logger.info(f"🎯 Legal RRF calculated scores for {len(rrf_scores)} documents")
            
            # Sort by RRF score
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Create legal document results
            final_results = []
            all_results = {r.id: r for r in dense_results + sparse_results}
            
            for doc_id, rrf_score in sorted_results:
                if doc_id in all_results:
                    result = all_results[doc_id]
                    text_content = result.payload.get("text", "") if hasattr(result, 'payload') else ""
                    
                    if not text_content:
                        continue
                        
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": result.payload.get("source", "") if hasattr(result, 'payload') else "",
                            "chunk_id": result.payload.get("chunk_id", 0) if hasattr(result, 'payload') else 0,
                            "doc_hash": result.payload.get("doc_hash", "") if hasattr(result, 'payload') else "",
                            "legal_rrf_score": rrf_score
                        }
                    )
                    final_results.append((doc, rrf_score))
                    
            logger.info(f"🏆 Final legal RRF results: {len(final_results)} documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Legal RRF fusion failed: {str(e)}")
            return []

    async def add_legal_documents_async(self, documents: List[Document], batch_size: int = config.BATCH_SIZE):
        """Add legal documents with both dense and sparse vectors - async wrapper"""
        return await to_thread(self._add_legal_documents_sync, documents, batch_size)

    def _add_legal_documents_sync(self, documents: List[Document], batch_size: int = config.BATCH_SIZE):
        """Synchronous version of add_legal_documents for thread execution"""
        try:
            doc_hash = documents[0].metadata.get('doc_hash')
            
            if self.document_exists(doc_hash):
                logger.info(f"Legal document {doc_hash} already indexed")
                self.processed_docs.add(doc_hash)
                return

            points = []
            
            def create_legal_hybrid_point(doc):
                try:
                    # Generate dense vector using Legal BERT
                    dense_embedding = embedding_model.encode(doc.page_content).tolist()
                    
                    # Generate sparse vector with legal terms
                    sparse_data = self.sparse_generator.generate_sparse_vector(doc.page_content)
                    sparse_vector = SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"]
                    )
                    
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            config.DENSE_VECTOR_NAME: dense_embedding,
                            config.SPARSE_VECTOR_NAME: sparse_vector
                        },
                        payload={
                            "text": doc.page_content,
                            "source": doc.metadata['source'],
                            "chunk_id": doc.metadata['chunk_id'],
                            "doc_hash": doc_hash,
                            "text_length": len(doc.page_content),
                            "document_type": "legal"
                        }
                    )
                    return point
                    
                except Exception as e:
                    logger.error(f"Failed to create legal hybrid point: {str(e)}")
                    return None

            # Process legal documents in parallel
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = [executor.submit(create_legal_hybrid_point, doc) for doc in documents]
                for future in as_completed(futures):
                    point = future.result()
                    if point:
                        points.append(point)

            # Batch upsert legal points
            if points:
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    logger.info(f"✅ Upserted batch of {len(batch)} legal hybrid points")

            self.processed_docs.add(doc_hash)
            logger.info(f"🎉 Successfully indexed {len(documents)} legal chunks with hybrid vectors")
            
        except Exception as e:
            logger.error(f"Failed to add legal documents: {str(e)}")
            raise

    async def delete_legal_documents(self, doc_hashes: List[str]):
        """Delete legal documents from QdrantDB"""
        try:
            for doc_hash in doc_hashes:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[FieldCondition(key="doc_hash", match=MatchValue(value=doc_hash))]
                        )
                    )
                )
                logger.info(f"Deleted legal vectors for document {doc_hash}")
                self.processed_docs.discard(doc_hash)
        except Exception as e:
            logger.error(f"Failed to delete legal documents: {str(e)}")

# Enhanced Legal Query Processor for Indian Law
class IndianLegalQueryEnhancer:
    def __init__(self):
        self.legal_terms = {
            'constitution': ['article', 'amendment', 'fundamental rights', 'directive principles'],
            'criminal law': ['ipc', 'indian penal code', 'crpc', 'criminal procedure', 'evidence act'],
            'civil law': ['contract', 'tort', 'civil procedure code', 'specific performance'],
            'property': ['ownership', 'title', 'possession', 'registration', 'transfer'],
            'company law': ['companies act', 'director', 'shareholder', 'corporate governance'],
            'family law': ['marriage', 'divorce', 'maintenance', 'custody', 'personal law'],
            'labour law': ['employment', 'workman', 'industrial dispute', 'trade union'],
            'tax law': ['income tax', 'gst', 'customs', 'excise', 'service tax'],
            'procedure': ['suit', 'plaint', 'written statement', 'evidence', 'judgment'],
            'appeal': ['high court', 'supreme court', 'revision', 'review']
        }

    def expand_legal_query(self, query: str) -> str:
        """Enhanced query expansion for Indian legal domain"""
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Specific legal mappings
        if "article" in query_lower and ("constitution" in query_lower or any(str(i) in query_lower for i in range(1, 400))):
            expanded_terms.extend(["constitutional", "fundamental rights", "directive principles", "amendment"])
        
        if "section" in query_lower and any(str(i) in query_lower for i in range(1, 1000)):
            expanded_terms.extend(["provision", "clause", "subsection", "explanation", "proviso"])
        
        if "ipc" in query_lower or "penal code" in query_lower:
            expanded_terms.extend(["criminal", "offense", "punishment", "mens rea", "actus reus"])
            
        if "crpc" in query_lower or "criminal procedure" in query_lower:
            expanded_terms.extend(["investigation", "trial", "bail", "cognizable", "warrant"])
        
        if "contract" in query_lower:
            expanded_terms.extend(["agreement", "consideration", "offer", "acceptance", "breach"])
        
        if "property" in query_lower:
            expanded_terms.extend(["ownership", "title", "possession", "transfer", "registration"])
            
        if "company" in query_lower or "corporate" in query_lower:
            expanded_terms.extend(["director", "shareholder", "board", "resolution", "compliance"])
        
        if "marriage" in query_lower or "divorce" in query_lower:
            expanded_terms.extend(["matrimonial", "maintenance", "alimony", "custody", "personal law"])
        
        # Add legal context terms
        legal_context = "law legal act section article provision rule regulation statute"
        expanded_query = f"{query} {' '.join(set(expanded_terms))} {legal_context}"
        
        return expanded_query

# Legal Document Processor
class LegalDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", " ", ""]  # Legal document appropriate separators
        )
        self.document_cache = {}
        self.supported_types = {
            'application/pdf': self._extract_pdf_text,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx_text,
            'text/plain': self._extract_text_content,
            'text/html': self._extract_text_content
        }

    def _get_document_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    async def download_document_async(self, url: str) -> Tuple[bytes, str]:
        return await to_thread(self.download_document, url)

    async def process_legal_document_async(self, url: str) -> List[Document]:
        return await to_thread(self.process_legal_document, url)

    def download_document(self, url: str) -> Tuple[bytes, str]:
        """Download legal document with timeout optimization"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
            
            headers = {'User-Agent': 'Legal-Document-Analyzer/1.0'}
            
            with requests.get(url, timeout=config.REQUEST_TIMEOUT, headers=headers, stream=True) as response:
                response.raise_for_status()
                
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > config.MAX_DOCUMENT_SIZE:
                    raise HTTPException(status_code=413, detail=f"Legal document too large. Max size: {config.MAX_DOCUMENT_SIZE} bytes")
                
                content = response.content
                
                # MIME type detection for legal documents
                mime_type = response.headers.get('content-type', '').split(';')[0]
                if not mime_type:
                    if url.lower().endswith('.pdf'):
                        mime_type = 'application/pdf'
                    elif url.lower().endswith('.docx'):
                        mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    else:
                        mime_type = 'text/plain'
                
                return content, mime_type
                
        except requests.RequestException as e:
            logger.error(f"Failed to download legal document {url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download legal document: {str(e)}")

    def _extract_pdf_text(self, content: bytes) -> str:
        """Optimized PDF text extraction for legal documents"""
        try:
            pdf_file = io.BytesIO(content)
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages[:100]):  # Limit pages for legal docs
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean legal document text
                            cleaned_text = self._clean_legal_text(page_text.strip())
                            text_parts.append(cleaned_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from legal page {page_num + 1}: {str(e)}")
                        continue
                
                return "\n".join(text_parts).strip()
        except Exception as e:
            logger.error(f"Legal PDF extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract legal PDF text: {str(e)}")

    def _clean_legal_text(self, text: str) -> str:
        """Clean and normalize legal document text"""
        # Remove excessive whitespace while preserving legal structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Preserve legal citations and references
        text = re.sub(r'\b(Section|Article|Rule|Regulation|Para|Paragraph)\s+(\d+)', r'\1 \2', text)
        text = re.sub(r'\b(Act,?\s+\d{4})', r'\1', text)
        
        return text.strip()

    def _extract_docx_text(self, content: bytes) -> str:
        try:
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            if text:
                return self._clean_legal_text(text.strip())
            return "No text content found in legal document"
        except Exception as e:
            logger.error(f"Legal DOCX extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract legal DOCX text: {str(e)}")

    def _extract_text_content(self, content: bytes) -> str:
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    text = content.decode(encoding, errors='ignore')
                    if text.strip():
                        return self._clean_legal_text(text.strip())
                except:
                    continue
            return "Unable to decode legal text content"
        except Exception as e:
            logger.error(f"Legal text extraction failed: {str(e)}")
            return "Failed to extract legal text content"

    def process_legal_document(self, url: str) -> List[Document]:
        """Process legal document with caching"""
        doc_hash = self._get_document_hash(url)
        
        if doc_hash in self.document_cache:
            logger.info(f"Using cached legal document for {url}")
            return self.document_cache[doc_hash]

        try:
            content, mime_type = self.download_document(url)
            logger.info(f"Downloaded legal document {url} with MIME type: {mime_type}")
            
            if mime_type in self.supported_types:
                text = self.supported_types[mime_type](content)
            else:
                logger.warning(f"Unsupported MIME type {mime_type}, treating as plain text")
                text = self._extract_text_content(content)

            if not text or len(text.strip()) < 50:  # Higher threshold for legal docs
                raise HTTPException(status_code=400, detail="Legal document appears to be empty or contains insufficient text content")

            chunks = self.text_splitter.split_text(text)
            meaningful_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]  # Higher threshold for legal
            
            if not meaningful_chunks:
                raise HTTPException(status_code=400, detail="No meaningful legal text chunks could be extracted from the document")

            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "chunk_id": i,
                        "mime_type": mime_type,
                        "doc_hash": doc_hash,
                        "document_type": "legal"
                    }
                )
                for i, chunk in enumerate(meaningful_chunks)
            ]

            self.document_cache[doc_hash] = documents
            logger.info(f"Processed {len(documents)} legal chunks for {url}")
            return documents

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing legal document {url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error processing legal document: {str(e)}")

# FastAPI App for Legal Document Analysis
app = FastAPI(
    title="Indian Legal Document Analyzer with InLegalBERT-2",
    description="Advanced Legal RAG system with hybrid search for Indian law documents using InLegalBERT-2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize legal components
legal_processor = LegalDocumentProcessor()
legal_vector_store = LegalHybridVectorStore(config.QDRANT_URL, config.COLLECTION_NAME, config.QDRANT_API_KEY)
legal_query_enhancer = IndianLegalQueryEnhancer()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Enhanced legal response cleaning
def clean_legal_response(response: str) -> str:
    """Clean legal response to remove context references"""
    patterns_to_remove = [
        r"according to the (?:provided )?(?:legal )?(?:context|document|information)[,.]?\s*",
        r"based on the (?:provided )?(?:legal )?(?:context|document|information)[,.]?\s*",
        r"as per (?:the )?(?:legal )?(?:context|document|section \d+)[,.]?\s*",
        r"from (?:the )?(?:provided )?(?:legal )?(?:context|information|document)[,.]?\s*",
        r"(?:the )?(?:legal )?(?:context|document) (?:shows|states|indicates|mentions)[,.]?\s*",
        r"referring to (?:the )?(?:provided )?(?:legal )?(?:context|information)[,.]?\s*",
        r"section \d+[,.]?\s*",
        r"chunk \d+[,.]?\s*",
    ]
    
    cleaned = response
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned

# Main Legal Analysis API endpoint
@app.post("/legal/analyze", response_model=LegalQueryResponse)
async def analyze_legal_documents(request: LegalQueryRequest, background_tasks: BackgroundTasks, token: str = Depends(verify_token)):
    
    async def _handle_legal_analysis():
        start_time = time.time()
        
        # Initialize legal LLM with specialized prompting
        llm = ChatOllama(
            model="gemma3:1b",  # Updated to available Gemma model
            temperature=0.1,    # Lower temperature for legal precision
            max_tokens=2048,    # More tokens for detailed legal analysis
        )
        
        try:
            doc_urls = [url.strip() for url in request.documents.split(',') if url.strip()]
            logger.info(f"Processing {len(doc_urls)} legal documents and {len(request.questions)} legal questions")

            # Process legal documents
            doc_hashes = []
            failed_docs = []
            
            # Process legal docs sequentially for accuracy
            for url in doc_urls:
                try:
                    doc_hash = legal_processor._get_document_hash(url)
                    
                    if not legal_vector_store.document_exists(doc_hash):
                        logger.info(f"Processing new legal document: {url}")
                        documents = await legal_processor.process_legal_document_async(url)
                        await legal_vector_store.add_legal_documents_async(documents)
                    else:
                        logger.info(f"Legal document already processed: {url}")
                    
                    doc_hashes.append(doc_hash)
                    
                except Exception as e:
                    logger.error(f"Error processing legal document {url}: {str(e)}")
                    failed_docs.append(f"{url}: {str(e)}")
                    continue

            if not doc_hashes:
                raise HTTPException(status_code=400, detail="No legal documents could be processed successfully.")

            # Legal analysis with controlled concurrency
            legal_sem = asyncio.Semaphore(3)

            # Process legal questions with specialized legal prompting
            async def process_legal_question(question: str) -> str:
                async with legal_sem:
                    try:
                        expanded_query = legal_query_enhancer.expand_legal_query(question)
                        retrieved_docs = await legal_vector_store.legal_hybrid_search(expanded_query)
                        
                        if not retrieved_docs:
                            return "No relevant legal information found in the provided documents for this question."

                        # Build legal context from fused results
                        context_parts = []
                        total_length = 0
                        max_context_length = 4000  # More context for legal analysis
                        
                        for doc, rrf_score in retrieved_docs:
                            if total_length + len(doc.page_content) > max_context_length:
                                remaining_space = max_context_length - total_length
                                if remaining_space > 200:
                                    context_parts.append(doc.page_content[:remaining_space] + "...")
                                break
                            context_parts.append(doc.page_content)
                            total_length += len(doc.page_content)
                        
                        context = "\n\n".join(context_parts)
                        
                        # Specialized legal system prompt
                        system_prompt = """You are a senior Indian legal expert. Answer in max 1 line, 20 words max, very short and concise. 
Use precise legal terms, cite relevant laws/articles, focus on key issues and remedies. 
If info insufficient, say "The provided legal documents do not contain sufficient information." 
Maintain professional tone.



Provide detailed, accurate legal analysis as a qualified legal practitioner would."""

                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"""Legal Question: {question}

Legal Document Context:
{context}

Provide comprehensive legal analysis addressing all aspects of this question based on the legal provisions and information available in the documents.""")
                        ]
                        
                        response = await llm.ainvoke(messages)
                        return clean_legal_response(response.content)
                        
                    except Exception as e:
                        logger.error(f"Error processing legal question '{question}': {str(e)}")
                        return f"An error occurred during legal analysis of this question: {str(e)}"

            # Process all legal questions concurrently
            logger.info("Processing legal questions with hybrid search...")
            legal_analyses = await asyncio.gather(*[process_legal_question(q) for q in request.questions])
            
            processing_time = time.time() - start_time
            logger.info(f"Completed legal analysis in {processing_time:.2f} seconds")
            
            # Schedule cleanup
            background_tasks.add_task(legal_vector_store.delete_legal_documents, doc_hashes)
            
            return {"legal_analysis": legal_analyses}
            
        except Exception as e:
            logger.error(f"Legal analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Legal analysis failed: {str(e)}")

    return await _handle_legal_analysis()

# Health check endpoint for legal system
@app.get("/health")
async def health_check():
    try:
        # Test InLegalBERT-2 model
        test_embedding = embedding_model.encode("test legal document analysis")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "system": "Indian Legal Document Analyzer",
            "components": {
                "inlegalbert_model": "operational",
                "qdrant_legal_vector_store": "operational",
                "legal_llm": "operational",
                "legal_hybrid_search": "enabled",
                "legal_sparse_vectors": "tfidf_legal_optimized",
                "legal_dense_vectors": "inlegalbert-2"
            }
        }
    except Exception as e:
        logger.error(f"Legal system health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Legal analysis service unhealthy")

# Legal system metrics endpoint
@app.get("/legal/metrics")
async def get_legal_metrics(token: str = Depends(verify_token)):
    return {
        "status": "operational",
        "system": "Indian Legal Document Analyzer",
        "configuration": {
            "qdrant_collection": config.COLLECTION_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "max_chunk_size": config.MAX_CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "similarity_threshold": config.SIMILARITY_THRESHOLD,
            "top_k": config.TOP_K,
            "legal_hybrid_search": {
                "dense_weight": config.DENSE_WEIGHT,
                "sparse_weight": config.SPARSE_WEIGHT,
                "sparse_method": "tfidf_legal_optimized",
                "fusion_algorithm": "legal_reciprocal_rank_fusion",
                "enabled": True
            }
        },
        "legal_specializations": [
            "Constitutional Law",
            "Criminal Law (IPC, CrPC, Evidence Act)",
            "Civil Law (Contract, Tort, Property)",
            "Commercial & Company Law",
            "Administrative & Service Law",
            "Tax Law (Income Tax, GST)",
            "Labour & Employment Law",
            "Family & Personal Law"
        ],
        "version": "1.0.0",
        "features": [
            "inlegalbert_2_embeddings",
            "legal_hybrid_search",
            "indian_law_optimization",
            "legal_term_expansion",
            "statutory_provision_analysis",
            "case_law_integration",
            "legal_citation_support"
        ]
    }

# Legal document types endpoint
@app.get("/legal/document-types")
async def get_supported_legal_document_types():
    return {
        "supported_legal_documents": [
            "Constitutional Documents",
            "Acts and Statutes",
            "Rules and Regulations",
            "Legal Judgments and Orders",
            "Legal Contracts and Agreements",
            "Legal Opinions and Briefs",
            "Government Notifications",
            "Legal Research Papers",
            "Case Law Reports",
            "Legal Commentaries"
        ],
        "supported_formats": [
            "PDF (Portable Document Format)",
            "DOCX (Microsoft Word)",
            "TXT (Plain Text)",
            "HTML (Web Documents)"
        ],
        "processing_capabilities": [
            "Legal text extraction and cleaning",
            "Legal citation recognition",
            "Statutory provision identification",
            "Legal term normalization",
            "Hierarchical legal structure preservation"
        ]
    }

# Legal search endpoint for direct queries
@app.post("/legal/search")
async def search_legal_documents(
    query: str,
    document_urls: str,
    top_k: int = 10,
    token: str = Depends(verify_token)
):
    """Direct legal document search endpoint"""
    try:
        doc_urls = [url.strip() for url in document_urls.split(',') if url.strip()]
        
        # Process documents if needed
        for url in doc_urls:
            doc_hash = legal_processor._get_document_hash(url)
            if not legal_vector_store.document_exists(doc_hash):
                documents = await legal_processor.process_legal_document_async(url)
                await legal_vector_store.add_legal_documents_async(documents)
        
        # Perform legal search
        expanded_query = legal_query_enhancer.expand_legal_query(query)
        results = await legal_vector_store.legal_hybrid_search(expanded_query, top_k)
        
        search_results = []
        for doc, score in results:
            search_results.append({
                "text": doc.page_content,
                "score": score,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", 0)
            })
        
        return {
            "query": query,
            "expanded_query": expanded_query,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Legal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Legal search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info", 
        access_log=True,
        timeout_keep_alive=60
    )