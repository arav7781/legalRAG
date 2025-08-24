import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from docling.document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_hub_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Create cache directories
for cache_dir in ["/tmp/transformers_cache", "/tmp/hf_home", "/tmp/hf_hub_cache", "/tmp/sentence_transformers"]:
    os.makedirs(cache_dir, exist_ok=True)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

qdrant_client = None
embeddings_model = None
llm = None
document_converter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize global resources on startup"""
    global qdrant_client, embeddings_model, llm, document_converter
    
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
            verify=False  
        )
        collections = qdrant_client.get_collections()
        logger.info("Qdrant client initialized and connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
            verify=False
        )
    
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'cache_folder': '/tmp/sentence_transformers'
            },
            cache_folder='/tmp/sentence_transformers'
        )
        logger.info("HuggingFace embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
        try:
            embeddings_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  
                model_kwargs={'device': 'cpu'},
                cache_folder='/tmp/sentence_transformers'
            )
            logger.info("HuggingFace embeddings initialized with fallback model")
        except Exception as e2:
            logger.error(f"Failed to initialize fallback embeddings: {e2}")
            embeddings_model = None
            logger.warning("No embeddings model available - embedding operations will fail")
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0
    )
    
    try:
        document_converter = DocumentConverter()
        logger.info("DocumentConverter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DocumentConverter: {e}")
        raise e
    
    logger.info("Application initialized successfully")
    yield
    
    # Cleanup
    logger.info("Application shutting down")