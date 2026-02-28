import json
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
            
        if request.resume:
            resume_data = request.resume
        else:
            resume_data = request
            
        analytics = resume_data.analytics
        enhancement = resume_data.enhancement
        
        # Prepare context for Gemini
        context_data = {
            "cv_quality": {
                "overall_score": analytics.cv_quality.overall_score,
                "subscores": [
                    {
                        "dimension": sub.dimension,
                        "score": sub.score,
                        "max_score": sub.max_score,
                        "evidence": sub.evidence
                    }
                    for sub in analytics.cv_quality.subscores
                ]
            },
            "jd_match": {
                "overall_score": analytics.jd_match.overall_score if analytics.jd_match else 0,
                "subscores": [
                    {
                        "dimension": sub.dimension,
                        "score": sub.score,
                        "max_score": sub.max_score,
                        "evidence": sub.evidence
                    }
                    for sub in (analytics.jd_match.subscores if analytics.jd_match else [])
                ]
            },
            "key_takeaways": {
                "green_flags": analytics.key_takeaways.green_flags,
                "red_flags": analytics.key_takeaways.red_flags
            },
            "tailored_resume": {
                "summary": enhancement.tailored_resume.summary,
                "skills": enhancement.tailored_resume.skills,
                "experience": enhancement.tailored_resume.experience,
                "projects": enhancement.tailored_resume.projects
            },
            "top_1_percent_gap": {
                "strengths": enhancement.top_1_percent_gap.strengths,
                "gaps": enhancement.top_1_percent_gap.gaps,
                "actionable_next_steps": enhancement.top_1_percent_gap.actionable_next_steps
            }
        }
        
        prompt = f"""You are an expert resume writer and career coach. Based on the comprehensive resume analysis data provided, generate the best possible professional resume content.

ANALYSIS DATA:
{json.dumps(context_data, indent=2)}

YOUR TASK:
Generate a complete, professional resume strictly answering as a JSON object matching the `ResumeBuilderResponse` schema formatting.
Extract and enhance all information intelligently from the evidence and data provided. Ensure personal_info is completely formulated (even if blank infer placeholders).
"""
        
        response = self.client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResumeBuilderResponse,
                temperature=0.2,
            )
        )
        
        result_dict = json.loads(response.text)
        return ResumeBuilderResponse(**result_dict)

resume_enhancer_service = ResumeEnhancerService()
