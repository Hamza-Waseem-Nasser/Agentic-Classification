"""
Simple API Endpoint for ITSM Classification
===========================================

This provides a single endpoint that your CTO can call to use your classification system.
Just run this file and it will expose your project at http://localhost:8000/classify

Usage:
    python simple_api.py

Then POST to: http://localhost:8000/classify
With JSON: {"text": "مرحبا، أحتاج مساعدة في الطابعة", "ticket_id": "optional-id"}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import Optional
import logging

# Import your existing classification system
from main import initialize_classification_system, classify_ticket

# Simple request/response models
class ClassifyRequest(BaseModel):
    text: str
    ticket_id: Optional[str] = None

class ClassifyResponse(BaseModel):
    ticket_id: str
    category: str
    subcategory: str
    category_confidence: float
    subcategory_confidence: float
    success: bool
    processing_time: float

# Create FastAPI app
app = FastAPI(title="ITSM Classification API", version="1.0.0")

# Global pipeline
pipeline = None

@app.on_event("startup")
async def startup():
    """Initialize the classification system on startup"""
    global pipeline
    print("🚀 Initializing ITSM Classification System...")
    try:
        pipeline = await initialize_classification_system(strict_mode=True)
        print("✅ Classification system ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "ITSM Classification API", 
        "status": "running",
        "endpoint": "/classify"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if pipeline else "unhealthy",
        "pipeline_ready": pipeline is not None
    }

@app.post("/classify", response_model=ClassifyResponse)
async def classify_ticket_endpoint(request: ClassifyRequest):
    """
    Main classification endpoint
    
    This is the single endpoint your CTO needs to call your system.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Classification system not ready")
    
    try:
        # Use your existing classification function
        result = await classify_ticket(
            pipeline, 
            request.text, 
            request.ticket_id or "auto-generated"
        )
        
        # Extract the important information
        classification = result.get('classification', {})
        processing = result.get('processing', {})
        
        response = ClassifyResponse(
            ticket_id=result.get('ticket_id', 'unknown'),
            category=classification.get('category', 'غير محدد'),
            subcategory=classification.get('subcategory', 'غير محدد'),
            category_confidence=classification.get('category_confidence', 0.0),
            subcategory_confidence=classification.get('subcategory_confidence', 0.0),
            success=result.get('success', False),
            processing_time=processing.get('processing_time', 0.0)
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/test-accuracy")
async def run_accuracy_test(max_cases: Optional[int] = 10):
    """
    Run accuracy test against your test data
    
    This endpoint runs your test_classification_accuracy.py functionality
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Classification system not ready")
    
    try:
        # Import your accuracy tester
        from test_classification_accuracy import ClassificationAccuracyTester
        
        # Create tester instance
        tester = ClassificationAccuracyTester()
        tester.pipeline = pipeline  # Use the already initialized pipeline
        
        # Load test data
        test_cases = tester.load_test_data()
        if not test_cases:
            raise HTTPException(status_code=400, detail="No test data found")
        
        # Limit test cases
        if max_cases and max_cases < len(test_cases):
            test_cases = test_cases[:max_cases]
        
        # Run tests
        results = []
        successful = 0
        failed = 0
        
        for test_case in test_cases:
            try:
                result = await tester.test_single_case(test_case)
                results.append(result)
                
                if result.get('success', False):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                results.append({
                    'incident_id': test_case.get('incident_id', 'unknown'),
                    'error': str(e),
                    'success': False
                })
        
        # Calculate basic accuracy
        if successful > 0:
            category_correct = sum(1 for r in results if r.get('category_correct', False))
            subcategory_correct = sum(1 for r in results if r.get('subcategory_correct', False))
            overall_correct = sum(1 for r in results if r.get('overall_correct', False))
            
            accuracy_summary = {
                "category_accuracy": f"{(category_correct/successful)*100:.1f}%",
                "subcategory_accuracy": f"{(subcategory_correct/successful)*100:.1f}%", 
                "overall_accuracy": f"{(overall_correct/successful)*100:.1f}%"
            }
        else:
            accuracy_summary = {
                "category_accuracy": "0%",
                "subcategory_accuracy": "0%",
                "overall_accuracy": "0%"
            }
        
        return {
            "test_summary": {
                "total_cases": len(test_cases),
                "successful": successful,
                "failed": failed,
                "accuracy": accuracy_summary
            },
            "sample_results": results[:5],  # First 5 results as examples
            "full_results": results  # All results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Accuracy test failed: {str(e)}")

if __name__ == "__main__":
    print("🎯 Starting Simple ITSM Classification API")
    print("📡 Your CTO can call:")
    print("   • POST http://localhost:8000/classify - Classify single ticket")
    print("   • POST http://localhost:8000/test-accuracy?max_cases=10 - Run accuracy test")
    print("📋 Example: {'text': 'Arabic ticket description', 'ticket_id': 'optional'}")
    print("🔗 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
