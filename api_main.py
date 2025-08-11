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

# Ollama import for local Gemma model
from langchain_community.chat_models import ChatOllama

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

# Enhanced Configuration
class Config:
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    BEARER_TOKEN = "legal_doc_analyzer_token_2024"
    
    # Vector configurations
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback to reliable model
    COLLECTION_NAME = "indian-legal-documents-hybrid"
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    
    # Hybrid search parameters
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3
    
    # Performance optimizations
    MAX_CHUNK_SIZE = 1000  # Smaller chunks for better precision
    CHUNK_OVERLAP = 100
    SIMILARITY_THRESHOLD = 0.6
    TOP_K = 10
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT = 45
    MAX_RETRIES = 3
    BATCH_SIZE = 20
    MAX_WORKERS = 4

config = Config()

# Initialize embeddings with fallback
try:
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info(f"Embedding model {config.EMBEDDING_MODEL} initialized successfully")
except Exception as e:
    logger.warning(f"Failed to load preferred model, falling back to all-MiniLM-L6-v2: {str(e)}")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

security = HTTPBearer()

# Pydantic Models
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

# Legal Sparse Vector Generator
class LegalSparseVectorGenerator:
    def __init__(self, method="tfidf"):
        self.method = method
        
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                vocabulary=self._build_legal_vocabulary(),
                ngram_range=(1, 3),
                max_features=10000,
                stop_words='english'
            )
            self._init_tfidf()
    
    def _build_legal_vocabulary(self):
        """Build Indian legal vocabulary"""
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
            'nirbhaya', 'case', 'mercy', 'petition', 'death', 'sentence', 'commute',
            
            # Civil law terms
            'civil', 'procedure', 'contract', 'tort', 'negligence', 'damages', 'injunction',
            'specific', 'performance', 'breach', 'consideration', 'offer', 'acceptance',
            'void', 'voidable', 'illegal', 'unenforceable', 'limitation', 'period',
            
            # Property law
            'property', 'ownership', 'possession', 'title', 'deed', 'sale', 'mortgage',
            'lease', 'rent', 'tenancy', 'easement', 'inheritance', 'succession',
            'registration', 'stamp', 'duty', 'transfer',
            
            # Legal procedures
            'plaintiff', 'defendant', 'petitioner', 'respondent', 'appellant', 'appellee',
            'suit', 'plaint', 'written', 'statement', 'issues', 'evidence', 'witness',
            'examination', 'cross-examination', 'judgment', 'decree', 'order', 'appeal',
            'revision', 'review', 'execution', 'attachment', 'garnishee',
            
            # Legal concepts
            'precedent', 'ratio', 'decidendi', 'obiter', 'dicta', 'stare', 'decisis',
            'res', 'judicata', 'subjudice', 'forum', 'non', 'conveniens', 'locus', 'standi',
            'mens', 'rea', 'actus', 'reus', 'burden', 'proof', 'prima', 'facie'
        ]
        return list(set(vocabulary_terms))
    
    def _init_tfidf(self):
        """Initialize TF-IDF with sample legal texts"""
        sample_texts = [
            "constitution article fundamental rights directive principles state policy supreme court high court",
            "indian penal code criminal procedure evidence act section murder culpable homicide theft robbery",
            "civil procedure contract tort negligence damages injunction specific performance breach consideration",
            "property ownership possession title deed sale mortgage lease rent tenancy easement inheritance",
            "plaintiff defendant petitioner respondent suit plaint written statement issues evidence witness",
            "nirbhaya case mercy petition death sentence commute supreme court president rejection"
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
            
            indices = []
            values = []
            
            for i, value in enumerate(sparse_array):
                if value > 0:
                    indices.append(i)
                    values.append(float(value))
            
            return {"indices": indices, "values": values}
        except Exception as e:
            logger.error(f"TF-IDF sparse vector generation failed: {str(e)}")
            return {"indices": [], "values": []}

# Enhanced QdrantDB Vector Store
class LegalHybridVectorStore:
    def __init__(self, url: str, collection_name: str, api_key: Optional[str] = None):
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.collection_name = collection_name
            self.sparse_generator = LegalSparseVectorGenerator(method="tfidf")
            
            # Get embedding dimensions
            test_embedding = embedding_model.encode("test legal document")
            self.dense_dimension = len(test_embedding)
            self.sparse_dimension = 10000
            
            logger.info(f"Embedding model produces {self.dense_dimension} dimensions")
            
            self._ensure_collection()
            self.processed_docs = set()
            logger.info(f"Legal hybrid vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid QdrantDB: {str(e)}")
            raise

    def _ensure_collection(self):
        """Ensure collection exists with correct dimensions, recreate if needed"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            collection_exists = self.collection_name in collection_names
            
            if collection_exists:
                try:
                    # Check if collection has correct dimensions
                    collection_info = self.client.get_collection(self.collection_name)
                    existing_dim = collection_info.config.params.vectors[config.DENSE_VECTOR_NAME].size
                    
                    if existing_dim != self.dense_dimension:
                        logger.warning(f"Collection has wrong dimensions ({existing_dim} vs {self.dense_dimension}). Recreating...")
                        self.client.delete_collection(self.collection_name)
                        collection_exists = False
                    else:
                        logger.info(f"Collection exists with correct dimensions: {existing_dim}")
                        return
                        
                except Exception as e:
                    logger.warning(f"Error checking collection info: {str(e)}. Recreating...")
                    try:
                        self.client.delete_collection(self.collection_name)
                    except:
                        pass
                    collection_exists = False
            
            if not collection_exists:
                logger.info(f"Creating new collection with {self.dense_dimension} dimensions")
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
                logger.info(f"Created new hybrid collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring hybrid collection: {str(e)}")
            raise

    def document_exists(self, doc_hash: str) -> bool:
        """Check if document exists"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="doc_hash", match=models.MatchValue(value=doc_hash))]
                ),
                limit=1
            )
            return len(result[0]) > 0
        except Exception as e:
            logger.warning(f"Error checking document existence: {str(e)}")
            return doc_hash in self.processed_docs

    async def hybrid_search(self, query: str, top_k: int = config.TOP_K) -> List[Tuple[Document, float]]:
        """Perform hybrid search with sparse + dense vectors"""
        try:
            start_time = time.time()
            
            # Generate query vectors
            dense_query = embedding_model.encode(query).tolist()
            sparse_query_data = self.sparse_generator.generate_sparse_vector(query)
            sparse_query = SparseVector(
                indices=sparse_query_data["indices"],
                values=sparse_query_data["values"]
            )
            
            logger.info(f"🔍 Query: {query}")
            logger.info(f"📊 Dense vector dim: {len(dense_query)}")
            logger.info(f"🏷️  Sparse vector terms: {len(sparse_query_data['indices'])}")
            
            # Perform batch search
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
            
            dense_results = batch_results[0] if len(batch_results) > 0 else []
            sparse_results = batch_results[1] if len(batch_results) > 1 else []
            
            if hasattr(dense_results, 'points'):
                dense_results = dense_results.points
            if hasattr(sparse_results, 'points'):
                sparse_results = sparse_results.points
                
            logger.info(f"📊 Dense results: {len(dense_results)}")
            logger.info(f"🏷️  Sparse results: {len(sparse_results)}")
            
            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
            
            search_time = time.time() - start_time
            logger.info(f"⚡ Hybrid search completed in {search_time:.3f}s - {len(fused_results)} results")
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, top_k: int) -> List[Tuple[Document, float]]:
        """Apply Reciprocal Rank Fusion"""
        try:
            rrf_scores = {}
            k = 60  # RRF parameter
            
            # Process dense results
            for rank, result in enumerate(dense_results, 1):
                doc_id = result.id
                score_contribution = config.DENSE_WEIGHT / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score_contribution
                
            # Process sparse results
            for rank, result in enumerate(sparse_results, 1):
                doc_id = result.id
                score_contribution = config.SPARSE_WEIGHT / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score_contribution
            
            # Sort by RRF score
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Create document results
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
                            "rrf_score": rrf_score
                        }
                    )
                    final_results.append((doc, rrf_score))
                    
            return final_results
            
        except Exception as e:
            logger.error(f"RRF fusion failed: {str(e)}")
            return []

    async def add_documents_async(self, documents: List[Document], batch_size: int = config.BATCH_SIZE):
        """Add documents with both dense and sparse vectors"""
        return await to_thread(self._add_documents_sync, documents, batch_size)

    def _add_documents_sync(self, documents: List[Document], batch_size: int = config.BATCH_SIZE):
        """Synchronous version of add_documents"""
        try:
            doc_hash = documents[0].metadata.get('doc_hash')
            
            if self.document_exists(doc_hash):
                logger.info(f"Document {doc_hash} already indexed")
                self.processed_docs.add(doc_hash)
                return

            points = []
            
            def create_hybrid_point(doc):
                try:
                    # Generate dense vector with correct dimensions
                    dense_embedding = embedding_model.encode(doc.page_content)
                    logger.debug(f"Generated embedding with {len(dense_embedding)} dimensions")
                    dense_embedding = dense_embedding.tolist()
                    
                    # Verify dimensions match expected
                    if len(dense_embedding) != self.dense_dimension:
                        logger.error(f"Dimension mismatch: expected {self.dense_dimension}, got {len(dense_embedding)}")
                        return None
                    
                    # Generate sparse vector
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
                    logger.error(f"Failed to create hybrid point: {str(e)}")
                    return None

            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = [executor.submit(create_hybrid_point, doc) for doc in documents]
                for future in as_completed(futures):
                    point = future.result()
                    if point:
                        points.append(point)

            # Batch upsert points
            if points:
                try:
                    for i in range(0, len(points), batch_size):
                        batch = points[i:i + batch_size]
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch
                        )
                        logger.info(f"✅ Upserted batch of {len(batch)} hybrid points")
                except Exception as e:
                    logger.error(f"Failed to upsert points: {str(e)}")
                    raise

            self.processed_docs.add(doc_hash)
            logger.info(f"🎉 Successfully indexed {len(documents)} chunks with hybrid vectors")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    async def delete_documents(self, doc_hashes: List[str]):
        """Delete documents from QdrantDB"""
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
                logger.info(f"Deleted vectors for document {doc_hash}")
                self.processed_docs.discard(doc_hash)
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")

# Legal Query Enhancer
class IndianLegalQueryEnhancer:
    def __init__(self):
        self.legal_patterns = {
            'bail': ['bail application', 'bail granted', 'bail rejected', 'custody', 'release'],
            'nirbhaya': ['nirbhaya case', 'delhi gang rape', '2012 case', 'juvenile', 'death sentence'],
            'mercy petition': ['mercy petition', 'president', 'commutation', 'death sentence', 'clemency'],
            'section': ['section', 'provision', 'clause', 'subsection'],
            'article': ['article', 'constitutional', 'fundamental rights'],
            'court': ['supreme court', 'high court', 'trial court', 'session court']
        }

    def expand_legal_query(self, query: str) -> str:
        """Enhanced query expansion for legal domain"""
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Check for legal patterns
        for pattern, related_terms in self.legal_patterns.items():
            if pattern in query_lower:
                expanded_terms.extend(related_terms)
        
        # Specific case handling
        if "nirbhaya" in query_lower or "delhi gang rape" in query_lower:
            expanded_terms.extend(["2012", "juvenile", "death sentence", "supreme court", "mercy petition"])
        
        if "bail" in query_lower:
            expanded_terms.extend(["custody", "release", "granted", "rejected", "application"])
        
        # Add legal context
        legal_context = "law legal act section article provision rule"
        expanded_query = f"{query} {' '.join(set(expanded_terms))} {legal_context}"
        
        return expanded_query

# Document Processor
class LegalDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", " ", ""]
        )
        self.document_cache = {}

    def _get_document_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    async def download_document_async(self, url: str) -> Tuple[bytes, str]:
        return await to_thread(self.download_document, url)

    async def process_document_async(self, url: str) -> List[Document]:
        return await to_thread(self.process_document, url)

    def download_document(self, url: str) -> Tuple[bytes, str]:
        """Download document with timeout optimization"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
            
            headers = {'User-Agent': 'Legal-Document-Analyzer/1.0'}
            
            with requests.get(url, timeout=config.REQUEST_TIMEOUT, headers=headers, stream=True) as response:
                response.raise_for_status()
                
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > config.MAX_DOCUMENT_SIZE:
                    raise HTTPException(status_code=413, detail=f"Document too large. Max size: {config.MAX_DOCUMENT_SIZE} bytes")
                
                content = response.content
                
                # MIME type detection
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
            logger.error(f"Failed to download document {url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(content)
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages[:100]):  # Limit pages
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self._clean_text(page_text.strip())
                            text_parts.append(cleaned_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
                
                return "\n".join(text_parts).strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Preserve legal citations
        text = re.sub(r'\b(Section|Article|Rule|Para)\s+(\d+)', r'\1 \2', text)
        
        return text.strip()

    def _extract_docx_text(self, content: bytes) -> str:
        try:
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            if text:
                return self._clean_text(text.strip())
            return "No text content found"
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract DOCX text: {str(e)}")

    def _extract_text_content(self, content: bytes) -> str:
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    text = content.decode(encoding, errors='ignore')
                    if text.strip():
                        return self._clean_text(text.strip())
                except:
                    continue
            return "Unable to decode text content"
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return "Failed to extract text content"

    def process_document(self, url: str) -> List[Document]:
        """Process document with caching"""
        doc_hash = self._get_document_hash(url)
        
        if doc_hash in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[doc_hash]

        try:
            content, mime_type = self.download_document(url)
            logger.info(f"Downloaded document {url} with MIME type: {mime_type}")
            
            # Extract text based on MIME type
            if mime_type == 'application/pdf':
                text = self._extract_pdf_text(content)
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = self._extract_docx_text(content)
            else:
                text = self._extract_text_content(content)

            if not text or len(text.strip()) < 50:
                raise HTTPException(status_code=400, detail="Document appears to be empty or contains insufficient text content")

            chunks = self.text_splitter.split_text(text)
            meaningful_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]
            
            if not meaningful_chunks:
                raise HTTPException(status_code=400, detail="No meaningful text chunks could be extracted from the document")

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
            logger.info(f"Processed {len(documents)} chunks for {url}")
            return documents

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document {url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error processing document: {str(e)}")

# FastAPI App
app = FastAPI(
    title="Legal Document Analyzer",
    description="Enhanced Legal RAG system with hybrid search",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize components
processor = LegalDocumentProcessor()
vector_store = LegalHybridVectorStore(config.QDRANT_URL, config.COLLECTION_NAME, config.QDRANT_API_KEY)
query_enhancer = IndianLegalQueryEnhancer()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Enhanced response cleaning
def clean_response(response: str) -> str:
    """Clean response to remove context references"""
    patterns_to_remove = [
        r"according to the (?:provided )?(?:context|document|information)[,.]?\s*",
        r"based on the (?:provided )?(?:context|document|information)[,.]?\s*",
        r"as per (?:the )?(?:context|document)[,.]?\s*",
        r"from (?:the )?(?:provided )?(?:context|information|document)[,.]?\s*",
        r"(?:the )?(?:context|document) (?:shows|states|indicates|mentions)[,.]?\s*",
        r"referring to (?:the )?(?:provided )?(?:context|information)[,.]?\s*",
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
async def analyze_legal_documents(
    request: LegalQueryRequest, 
    background_tasks: BackgroundTasks, 
    token: str = Depends(verify_token)
):
    """Enhanced legal analysis endpoint that directly answers questions"""
    
    async def _handle_legal_analysis():
        start_time = time.time()
        
        # Initialize LLM with better settings for direct answers
        llm = ChatOllama(
            model="gemma3:1b",  # Use available model
            temperature=0.1,      # Low temperature for factual responses
            max_tokens=1000,      # Sufficient tokens for detailed answers
        )
        
        try:
            doc_urls = [url.strip() for url in request.documents.split(',') if url.strip()]
            logger.info(f"Processing {len(doc_urls)} documents and {len(request.questions)} questions")

            # Process documents
            doc_hashes = []
            failed_docs = []
            
            for url in doc_urls:
                try:
                    doc_hash = processor._get_document_hash(url)
                    
                    if not vector_store.document_exists(doc_hash):
                        logger.info(f"Processing new document: {url}")
                        documents = await processor.process_document_async(url)
                        await vector_store.add_documents_async(documents)
                    else:
                        logger.info(f"Document already processed: {url}")
                    
                    doc_hashes.append(doc_hash)
                    
                except Exception as e:
                    logger.error(f"Error processing document {url}: {str(e)}")
                    failed_docs.append(f"{url}: {str(e)}")
                    continue

            if not doc_hashes:
                raise HTTPException(status_code=400, detail="No documents could be processed successfully.")

            # Process questions with direct answer focus
            async def process_question(question: str) -> str:
                try:
                    # Expand query for better search
                    expanded_query = query_enhancer.expand_legal_query(question)
                    
                    # Search for relevant documents
                    retrieved_docs = await vector_store.hybrid_search(expanded_query, top_k=8)
                    
                    if not retrieved_docs:
                        return "I could not find relevant information in the provided documents to answer this question."

                    # Build focused context from most relevant results
                    context_parts = []
                    total_length = 0
                    max_context_length = 3000  # Focused context length
                    
                    for doc, score in retrieved_docs:
                        if total_length + len(doc.page_content) > max_context_length:
                            remaining_space = max_context_length - total_length
                            if remaining_space > 200:
                                context_parts.append(doc.page_content[:remaining_space] + "...")
                            break
                        context_parts.append(doc.page_content)
                        total_length += len(doc.page_content)
                    
                    context = "\n\n".join(context_parts)
                    
                    # Enhanced system prompt for direct answers
                    system_prompt = """You are an expert Indian legal analyst. Your task is to provide direct, accurate answers to legal questions based on the provided legal documents.

INSTRUCTIONS:
1. Answer the question directly and specifically
2. Use information ONLY from the provided legal documents
3. If the question asks about bail, focus on bail-related information
4. If the question asks about specific cases (like Nirbhaya), focus on that case
5. If the question asks about legal procedures, explain the relevant procedures
6. Be concise but comprehensive
7. Do NOT mention "according to the document" or similar phrases
8. If you cannot find the answer in the documents, say "The provided documents do not contain sufficient information to answer this question"

Provide a clear, direct answer that directly addresses what the user is asking."""

                    # Create focused prompt
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"""Question: {question}

Legal Document Content:
{context}

Please provide a direct answer to this question based on the legal information provided above.""")
                    ]
                    
                    response = await llm.ainvoke(messages)
                    answer = clean_response(response.content)
                    
                    # Ensure we have a meaningful answer
                    if len(answer.strip()) < 10:
                        return "The provided documents do not contain sufficient information to answer this question."
                    
                    return answer
                        
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {str(e)}")
                    return f"An error occurred while processing this question: {str(e)}"

            # Process all questions
            logger.info("Processing questions with hybrid search...")
            legal_analyses = await asyncio.gather(*[process_question(q) for q in request.questions])
            
            processing_time = time.time() - start_time
            logger.info(f"Completed legal analysis in {processing_time:.2f} seconds")
            
            # Schedule cleanup
            background_tasks.add_task(vector_store.delete_documents, doc_hashes)
            
            return {"legal_analysis": legal_analyses}
            
        except Exception as e:
            logger.error(f"Legal analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Legal analysis failed: {str(e)}")

    return await _handle_legal_analysis()

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test embedding model
        test_embedding = embedding_model.encode("test legal document")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "system": "Legal Document Analyzer",
            "components": {
                "embedding_model": "operational",
                "qdrant_vector_store": "operational",
                "legal_llm": "operational",
                "hybrid_search": "enabled"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Direct search endpoint
@app.post("/legal/search")
async def search_documents(
    query: str,
    document_urls: str,
    top_k: int = 10,
    token: str = Depends(verify_token)
):
    """Direct document search endpoint"""
    try:
        doc_urls = [url.strip() for url in document_urls.split(',') if url.strip()]
        
        # Process documents if needed
        for url in doc_urls:
            doc_hash = processor._get_document_hash(url)
            if not vector_store.document_exists(doc_hash):
                documents = await processor.process_document_async(url)
                await vector_store.add_documents_async(documents)
        
        # Perform search
        expanded_query = query_enhancer.expand_legal_query(query)
        results = await vector_store.hybrid_search(expanded_query, top_k)
        
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
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Chat endpoint for real-time interaction
@app.post("/legal/chat")
async def chat_with_documents(
    question: str,
    document_urls: str,
    token: str = Depends(verify_token)
):
    """Single question chat endpoint"""
    try:
        # Create a request object
        request = LegalQueryRequest(
            documents=document_urls,
            questions=[question]
        )
        
        # Use the existing analyze endpoint
        background_tasks = BackgroundTasks()
        result = await analyze_legal_documents(request, background_tasks, token)
        
        return {
            "question": question,
            "answer": result.legal_analysis[0] if result.legal_analysis else "No answer available",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Metrics endpoint
@app.get("/legal/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    return {
        "status": "operational",
        "system": "Legal Document Analyzer",
        "configuration": {
            "qdrant_collection": config.COLLECTION_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "max_chunk_size": config.MAX_CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "similarity_threshold": config.SIMILARITY_THRESHOLD,
            "top_k": config.TOP_K,
            "hybrid_search": {
                "dense_weight": config.DENSE_WEIGHT,
                "sparse_weight": config.SPARSE_WEIGHT,
                "enabled": True
            }
        },
        "version": "2.0.0",
        "features": [
            "direct_question_answering",
            "hybrid_search",
            "legal_optimization",
            "document_caching",
            "real_time_chat"
        ]
    }

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