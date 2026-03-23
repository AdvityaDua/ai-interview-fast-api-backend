from fastapi import APIRouter
from app.api.v1.test_route import router as test_router
from app.api.v1.resume import router as resume_router
from app.api.v1.cv import router as cv_router
from app.api.v1.upload import router as upload_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.code_execution import router as code_router
from app.api.v1.tts import router as tts_router

api_router = APIRouter()
api_router.include_router(test_router, tags=["Test"])
api_router.include_router(resume_router, tags=["Resume"])
api_router.include_router(cv_router, tags=["CV Evaluation"])
api_router.include_router(upload_router, prefix="/upload", tags=["Uploads"])
api_router.include_router(evaluation_router, tags=["Evaluation"])
api_router.include_router(code_router)
api_router.include_router(tts_router, tags=["TTS"])
