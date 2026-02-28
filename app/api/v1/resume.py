from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_current_user
from app.models.resume_schemas import ResumeAnalysisRequest, ResumeBuilderResponse
from app.services.resume_enhancer import resume_enhancer_service

router = APIRouter()

@router.post("/resume/final-enhanced", response_model=ResumeBuilderResponse)
async def generate_final_enhanced_resume(
    request: ResumeAnalysisRequest,
    user: dict = Depends(get_current_user)
):
    """
    AI-enhanced resume generator using Google Gemini 2.5 Pro.
    Expects Authorization Bearer Header.
    """
    try:
        return await resume_enhancer_service.enhance_resume(request)
        
    except Exception as e:
        import traceback
        error_detail = f"AI resume generation failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )
