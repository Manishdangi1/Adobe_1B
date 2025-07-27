"""
FastAPI web application for Challenge 1b PDF Analysis
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from ..core.analysis_engine import AnalysisEngine
from ...config.config import app_config

# Pydantic models for API
class ChallengeInfo(BaseModel):
    challenge_id: str = Field(..., description="Challenge identifier")
    test_case_name: str = Field(..., description="Test case name")

class Document(BaseModel):
    filename: str = Field(..., description="Document filename")
    title: str = Field(..., description="Document title")

class Persona(BaseModel):
    role: str = Field(..., description="User persona role")

class JobToBeDone(BaseModel):
    task: str = Field(..., description="Task description")

class AnalysisRequest(BaseModel):
    challenge_info: ChallengeInfo
    documents: List[Document]
    persona: Persona
    job_to_be_done: JobToBeDone

class AnalysisResponse(BaseModel):
    metadata: Dict[str, Any]
    extracted_sections: List[Dict[str, Any]]
    subsection_analysis: List[Dict[str, Any]]
    processing_time: float
    status: str = "success"

# Initialize FastAPI app
app = FastAPI(
    title="Challenge 1b PDF Analysis API",
    description="Advanced PDF analysis solution for multi-collection document processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global analysis engine
analysis_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global analysis_engine
    analysis_engine = AnalysisEngine()
    await analysis_engine.initialize()
    logger.info("API server started and analysis engine initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global analysis_engine
    if analysis_engine:
        await analysis_engine.cleanup()
    logger.info("API server shutdown")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Challenge 1b PDF Analysis API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_initialized": analysis_engine is not None
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_collection(request: AnalysisRequest):
    """
    Analyze a collection of PDFs based on persona and task
    
    Args:
        request: Analysis request with challenge info, persona, and task
        
    Returns:
        Analysis results with extracted sections and insights
    """
    try:
        if not analysis_engine:
            raise HTTPException(status_code=500, detail="Analysis engine not initialized")
        
        # Get collection path based on challenge ID
        collection_name = f"Collection {request.challenge_info.challenge_id.split('_')[-1]}"
        collection_path = app_config.base_dir / collection_name
        
        if not collection_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Collection not found: {collection_name}"
            )
        
        # Run analysis
        result = await analysis_engine.analyze_collection(
            collection_path=collection_path,
            challenge_id=request.challenge_info.challenge_id,
            persona=request.persona.role,
            job_to_be_done=request.job_to_be_done.task
        )
        
        return AnalysisResponse(
            metadata=result.metadata,
            extracted_sections=[
                {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": section["importance_rank"],
                    "page_number": section["page_number"]
                }
                for section in result.extracted_sections
            ],
            subsection_analysis=[
                {
                    "document": analysis["document"],
                    "refined_text": analysis["refined_text"],
                    "page_number": analysis["page_number"]
                }
                for analysis in result.subsection_analysis
            ],
            processing_time=result.processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/collection/{collection_name}")
async def analyze_specific_collection(
    collection_name: str,
    background_tasks: BackgroundTasks
):
    """
    Analyze a specific collection by name
    
    Args:
        collection_name: Name of the collection to analyze
        background_tasks: FastAPI background tasks
        
    Returns:
        Analysis status and job ID
    """
    try:
        collection_path = app_config.base_dir / collection_name
        if not collection_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Collection not found: {collection_name}"
            )
        
        # Look for input file
        input_file = collection_path / "challenge1b_input.json"
        if not input_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Input file not found for {collection_name}"
            )
        
        # Read input configuration
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        challenge_id = input_data.get("challenge_info", {}).get("challenge_id", "")
        persona = input_data.get("persona", {}).get("role", "")
        job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
        
        if not all([challenge_id, persona, job_to_be_done]):
            raise HTTPException(
                status_code=400,
                detail="Invalid input configuration"
            )
        
        # Add background task for analysis
        output_file = collection_path / "challenge1b_output.json"
        background_tasks.add_task(
            run_background_analysis,
            collection_name,
            challenge_id,
            persona,
            job_to_be_done,
            output_file
        )
        
        return {
            "status": "analysis_started",
            "collection": collection_name,
            "challenge_id": challenge_id,
            "message": "Analysis started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_background_analysis(
    collection_name: str,
    challenge_id: str,
    persona: str,
    job_to_be_done: str,
    output_file: Path
):
    """Run analysis in background"""
    try:
        collection_path = app_config.base_dir / collection_name
        
        result = await analysis_engine.analyze_collection(
            collection_path=collection_path,
            challenge_id=challenge_id,
            persona=persona,
            job_to_be_done=job_to_be_done
        )
        
        analysis_engine.save_analysis_result(result, output_file)
        logger.info(f"Background analysis completed for {collection_name}")
        
    except Exception as e:
        logger.error(f"Background analysis failed for {collection_name}: {e}")

@app.get("/collections")
async def list_collections():
    """List available collections"""
    try:
        collection_dirs = [
            d.name for d in app_config.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("Collection")
        ]
        
        collections = []
        for collection_name in collection_dirs:
            collection_path = app_config.base_dir / collection_name
            input_file = collection_path / "challenge1b_input.json"
            
            collection_info = {
                "name": collection_name,
                "has_input": input_file.exists(),
                "has_output": (collection_path / "challenge1b_output.json").exists(),
                "pdf_count": len(list((collection_path / "PDFs").glob("*.pdf"))) if (collection_path / "PDFs").exists() else 0
            }
            
            if input_file.exists():
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                    collection_info["challenge_id"] = input_data.get("challenge_info", {}).get("challenge_id", "")
                    collection_info["persona"] = input_data.get("persona", {}).get("role", "")
                except Exception:
                    pass
            
            collections.append(collection_info)
        
        return {"collections": collections}
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/status")
async def get_collection_status(collection_name: str):
    """Get status of a specific collection"""
    try:
        collection_path = app_config.base_dir / collection_name
        if not collection_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Collection not found: {collection_name}"
            )
        
        input_file = collection_path / "challenge1b_input.json"
        output_file = collection_path / "challenge1b_output.json"
        pdf_dir = collection_path / "PDFs"
        
        status = {
            "collection": collection_name,
            "exists": True,
            "has_input": input_file.exists(),
            "has_output": output_file.exists(),
            "has_pdfs": pdf_dir.exists(),
            "pdf_count": len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
        }
        
        if input_file.exists():
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                status["challenge_id"] = input_data.get("challenge_info", {}).get("challenge_id", "")
                status["persona"] = input_data.get("persona", {}).get("role", "")
            except Exception:
                pass
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                status["last_analysis"] = output_data.get("metadata", {}).get("analysis_timestamp", "")
                status["processing_time"] = output_data.get("metadata", {}).get("processing_time", 0)
            except Exception:
                pass
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting collection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about the current ML model"""
    try:
        if not analysis_engine or not analysis_engine.embedding_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        return analysis_engine.embedding_model.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/clear-cache")
async def clear_model_cache():
    """Clear the model cache"""
    try:
        if not analysis_engine or not analysis_engine.embedding_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        analysis_engine.embedding_model.clear_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def run_api_server():
    """Run the API server"""
    uvicorn.run(
        "src.api.app:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
        log_level=app_config.log_level.lower()
    )

if __name__ == "__main__":
    run_api_server() 