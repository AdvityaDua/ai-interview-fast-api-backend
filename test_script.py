import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.environ.get("GEMINI_API_KEY", "")

from app.services.resume_enhancer import resume_enhancer_service
from app.models.resume_schemas import ResumeAnalysisRequest

async def test():
    req_data = {
        "message": "test",
        "resume": {
            "id": "123",
            "filename": "mock.pdf",
            "analytics": {
                "cv_quality": {"overall_score": 80, "subscores": []},
                "jd_match": {"overall_score": 80, "subscores": []}
            },
            "enhancement": {
                "tailored_resume": {
                    "summary": "M",
                    "experience": [],
                    "skills": [],
                    "projects": []
                }
            }
        }
    }
    
    req = ResumeAnalysisRequest(**req_data)
    try:
        res = await resume_enhancer_service.enhance_resume(req)
        print("Success!")
        print(res)
    except Exception as e:
        print("Error!")
        print(repr(e))

if __name__ == "__main__":
    asyncio.run(test())
