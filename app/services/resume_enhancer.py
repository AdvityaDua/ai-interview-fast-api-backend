import json
from typing import Optional, List, Any
from google import genai
from google.genai import types
from app.core.config import settings
from app.models.resume_schemas import ResumeAnalysisRequest, ResumeBuilderResponse

class ResumeEnhancerService:
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        
    async def enhance_resume(self, request: ResumeAnalysisRequest) -> ResumeBuilderResponse:
        if not self.client:
            raise ValueError("GOOGLE_API_KEY is not configured.")
            
        # Extract data directly or from nested resume object
        rd = request.resume if request.resume else request
        an = getattr(rd, 'analytics', None)
        en = getattr(rd, 'enhancement', None)
        
        # Build optimized prompt context
        context = {
            "cv_quality": {
                "score": an.cv_quality.overall_score if an and an.cv_quality else 0,
                "details": [s.model_dump() for s in an.cv_quality.subscores] if an and an.cv_quality and an.cv_quality.subscores else []
            },
            "jd_match": {
                "score": an.jd_match.overall_score if an and an.jd_match else 0,
                "details": [s.model_dump() for s in an.jd_match.subscores] if an and an.jd_match and an.jd_match.subscores else []
            },
            "tailored_data": en.model_dump() if en else {}
        }
        
        prompt = f"""You are an expert resume writer. Using the analysis data below, generate a professional, high-impact resume.
        
        ANALYSIS DATA:
        {json.dumps(context, indent=2)}
        
        TASK:
        Return a JSON object matching the `ResumeBuilderResponse` schema.
        - Structure `personal_info` accurately.
        - Categorize `skills` into `frontend`, `backend`, and `tools_cloud`.
        - Provide `experience` descriptions as impactful bullet point lists.
        - Ensure `projects` have `technologies` (list) and `highlights` (bullets).
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeBuilderResponse,
                temperature=0.1,
            )
        )
        
        # Parse and validate the response
        return ResumeBuilderResponse.model_validate_json(response.text)

resume_enhancer_service = ResumeEnhancerService()
