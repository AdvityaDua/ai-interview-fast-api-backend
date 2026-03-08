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
        
    async def enhance_resume(self, request: ResumeAnalysisRequest, user: dict = None, raw_text: str = "") -> ResumeBuilderResponse:
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
            "tailored_data": en.model_dump() if en else {},
            "user_info": user if user else {}
        }
        
        prompt = f"""You are an expert resume writer. Using the analysis data and the raw resume text below, generate a professional, high-impact resume.
        
        RAW RESUME TEXT:
        {raw_text}
        
        ANALYSIS DATA:
        {json.dumps(context, indent=2)}
        
        TASK:
        Return a JSON object matching the `ResumeBuilderResponse` schema.
        - Structure `personal_info` accurately. Use the `user_info` email if provided. DO NOT use generic or dummy names (like "Akash Bargoti" or "John Doe") if personal info is not provided; leave the name as empty string or "[Your Name]" instead. Extract correct phone, GitHub, LinkedIn, and education directly from RAW RESUME TEXT.
        - Categorize `skills` into `frontend`, `backend`, and `tools_cloud`.
        - Provide `experience` descriptions as impactful bullet point lists. Use EXACT history from RAW RESUME TEXT.
        - Ensure `projects` have `technologies` (list) and `highlights` (bullets).
        - DO NOT hallucinate fake education, certifications, or experience. If the data is missing from ANALYSIS DATA or RAW RESUME TEXT, return empty lists for those fields.
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
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
