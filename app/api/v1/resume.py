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
        import requests, tempfile, os
        from app.api.v1.upload import extract_text
        
        body = await request.json()
        validated_data = ResumeAnalysisRequest.model_validate(body)
        
        # 1. Fetch the raw resume data
        raw_text = ""
        resume_url = ""
        if validated_data.resume and validated_data.resume.url:
            resume_url = validated_data.resume.url
        elif validated_data.url:
            resume_url = validated_data.url
            
        if resume_url:
            try:
                local_url = resume_url.replace("localhost", "127.0.0.1")
                resp = requests.get(local_url, timeout=15)
                if resp.status_code == 200:
                    # try to get extension
                    ext = ".pdf"
                    if "." in local_url.split("/")[-1]:
                        ext = "." + local_url.split("/")[-1].split(".")[-1]
                        
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name
                    try:
                        raw_text = extract_text(tmp_path)
                    finally:
                        os.remove(tmp_path)
            except Exception as e:
                pass

        # 2. Invoke generator
        return await resume_enhancer_service.enhance_resume(validated_data, user, raw_text)
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"AI resume generation failed: {str(e)}"
        )
