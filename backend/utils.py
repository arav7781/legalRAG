import requests
import tempfile
import os
import logging
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastapi import HTTPException
from config import document_converter, embeddings_model, qdrant_client, logger

async def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    try:
        response = requests.get(str(url), timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

async def extract_pdf_content(pdf_content: bytes) -> List[Document]:
    """Extract content from PDF using multiple fallback methods"""
    temp_file_path = None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        full_text = ""
        
        # Method 1: Try Docling (most robust)
        try:
            logger.info("Trying Docling extraction...")
            result = document_converter.convert(temp_file_path)
            full_text = result.document.export_to_markdown()
            
            if not full_text or len(full_text.strip()) < 10:
                full_text = result.document.export_to_text() or ""
            
            if full_text and len(full_text.strip()) > 10:
                logger.info("Docling extraction successful")
            else:
                raise ValueError("Docling extracted minimal text")
                
        except Exception as e:
            logger.warning(f"Docling extraction failed: {e}")
            full_text = ""
        
        # Method 2: Fallback to PyPDF2
        if not full_text or len(full_text.strip()) < 10:
            try:
                logger.info("Trying PyPDF2 extraction...")
                full_text = await extract_pdf_fallback(temp_file_path)
                if full_text and len(full_text.strip()) > 10:
                    logger.info("PyPDF2 extraction successful")
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 3: Try pdfplumber as another fallback
        if not full_text or len(full_text.strip()) < 10:
            try:
                logger.info("Trying pdfplumber extraction...")
                full_text = await extract_pdf_pdfplumber(temp_file_path)
                if full_text and len(full_text.strip()) > 10:
                    logger.info("pdfplumber extraction successful")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        if not full_text or len(full_text.strip()) < 10:
            raise ValueError("No readable text found in PDF. The PDF might be image-based and requires OCR, or it might be corrupted.")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "pdf",
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        
        if not documents:
            raise ValueError("No text content could be extracted from the PDF")
        
        logger.info(f"Extracted {len(documents)} document chunks")
        return documents
        
    except Exception as e:
        # Clean up temporary file in case of error
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        logger.error(f"Failed to extract PDF content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF content: {e}")

async def extract_pdf_fallback(pdf_path: str) -> str:
    """Fallback PDF extraction using PyPDF2"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except ImportError:
        logger.warning("PyPDF2 not available for fallback extraction")
        return ""
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
        return ""

async def extract_pdf_pdfplumber(pdf_path: str) -> str:
    """Another fallback using pdfplumber"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        logger.warning("pdfplumber not available")
        return ""
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
        return ""

async def store_in_qdrant(documents: List[Document], collection_name: str):
    """Store documents in Qdrant vector database"""
    try:
        if embeddings_model is None:
            raise ValueError("Embeddings model not available")
        
        # Create collection if it doesn't exist
        try:
            qdrant_client.get_collection(collection_name)
        except Exception:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        # Generate embeddings and store documents
        points = []
        for i, doc in enumerate(documents):
            embedding = embeddings_model.embed_query(doc.page_content)
            
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            )
            points.append(point)
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        logger.info(f"Stored {len(documents)} documents in Qdrant collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to store documents in Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store documents: {e}")