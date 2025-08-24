import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from models import PDFUploadRequest, QuestionRequest, ChatResponse
from utils import download_pdf, extract_pdf_content, store_in_qdrant
from agent import create_workflow
from config import qdrant_client, logger
from typing import List

app = FastAPI(
    title="Agentic RAG with PDF Processing",
    description="Production-ready RAG system with agentic workflow for PDF Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-pdf", response_model=dict)
async def upload_pdf(request: PDFUploadRequest, background_tasks: BackgroundTasks):
    """Upload and process PDF from URL"""
    try:
        collection_name = request.collection_name or f"pdf_{uuid.uuid4().hex[:8]}"
        
        pdf_content = await download_pdf(request.pdf_url)
        
        documents = await extract_pdf_content(pdf_content)
        
        await store_in_qdrant(documents, collection_name)
        
        return {
            "status": "success",
            "message": f"PDF processed successfully",
            "collection_name": collection_name,
            "document_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    """Chat with the documents using agentic RAG"""
    try:
        # Check if collection exists
        try:
            qdrant_client.get_collection(request.collection_name)
        except Exception:
            raise HTTPException(
                status_code=404, 
                detail=f"Collection '{request.collection_name}' not found. Please upload a PDF first."
            )
        
        # Create workflow
        workflow = create_workflow()
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=request.question)],
            "collection_name": request.collection_name
        }
        
        # Run the workflow
        result = workflow.invoke(initial_state)
        
        # Extract final answer
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        return ChatResponse(
            answer=answer,
            sources=[request.collection_name],
            metadata={
                "collection_name": request.collection_name,
                "message_count": len(result["messages"])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.get("/collections", response_model=List[str])
async def list_collections():
    """List all available collections"""
    try:
        collections = qdrant_client.get_collections()
        return [collection.name for collection in collections.collections]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Agentic RAG service is running"}

if __name__ == "__main__":
    run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False
    )