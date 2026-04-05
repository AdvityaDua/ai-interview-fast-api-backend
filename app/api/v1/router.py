from fastapi import APIRouter
from app.api.v1.test_route import router as test_router
from app.api.v1.resume import router as resume_router
from app.api.v1.cv import router as cv_router
from app.api.v1.upload import router as upload_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.code_execution import router as code_router
from app.api.v1.tts import router as tts_router
from app.api.v1.semantic_match import router as semantic_match_router
from app.api.v1.docs import router as docs_router
from app.api.v1.interview_engine import router as interview_engine_router
from app.api.v1.endpoints.rag import router as rag_router
from app.api.v1.endpoints.question_bank import router as question_bank_router
from app.api.v1.endpoints.cover_letter import router as cover_letter_router

api_router = APIRouter()
api_router.include_router(test_router, tags=["Test"])
api_router.include_router(resume_router, tags=["Resume"])
api_router.include_router(cv_router, tags=["CV Evaluation"])
api_router.include_router(upload_router, prefix="/upload", tags=["Uploads"])
api_router.include_router(evaluation_router, tags=["Evaluation"])
api_router.include_router(code_router)
api_router.include_router(tts_router, tags=["TTS"])
api_router.include_router(semantic_match_router, tags=["Matching"])
api_router.include_router(docs_router, prefix="/docs", tags=["Documentation"])
api_router.include_router(interview_engine_router, tags=["Interview Engine"])
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
api_router.include_router(question_bank_router, prefix="/question-bank", tags=["Question Bank"])
api_router.include_router(cover_letter_router, prefix="/cover-letter", tags=["Cover Letter"])
