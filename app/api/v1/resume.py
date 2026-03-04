from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.dependencies import get_current_user
from app.models.resume_schemas import ResumeAnalysisRequest, ResumeBuilderResponse
from app.services.resume_enhancer import resume_enhancer_service

router = APIRouter()

@router.post("/resume/final-enhanced", response_model=ResumeBuilderResponse)
async def generate_final_enhanced_resume(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """
    AI-enhanced resume generator using Google Gemini.
    """
    try:
        body = await request.json()
        validated_data = ResumeAnalysisRequest.model_validate(body)
        return await resume_enhancer_service.enhance_resume(validated_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI resume generation failed: {str(e)}"
        )
