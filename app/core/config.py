from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Prod-Grade FastAPI Backend"
    JWT_ACCESS_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    PORT: int = 8001
    HOST: str = "0.0.0.0"
    # Optional — keys are fetched from NestJS admin DB at runtime; env vars are fallback
    GOOGLE_API_KEY: Optional[str] = None
    DEEPGRAM_KEY: str
    GROQ_API_KEY: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"
    PISTON_URL: str = "http://localhost:2000"
    BACKEND_URL: str = "https://api.aiforjob.ai"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002"

    class Config:
        env_file = ".env"

settings = Settings()

API_DESCRIPTION = """
🚀 **AI Interview Coach API**

This API powers the core AI models, WebSockets for the Interview module, and Resume Evaluation engine. 
It replaces the legacy backends with a robust, production-grade FastAPI architecture.

### Key Features:
* **Interview WebSockets**: Handles full duplex transcription and conversation with Gemini.
* **Resume Enhancer**: Powered by Groq for ultra-fast CV/JD analysis and scoring.
* **Authentication**: Native JWT validation syncing perfectly with the frontend & NestJS.
"""

TAGS_METADATA = [
    {"name": "Resume", "description": "Core Resume Enhancer using Gemini 2.5 Pro for full generation."},
    {"name": "CV Evaluation", "description": "Legacy routes for scoring and fit-index using Groq."},
    {"name": "Uploads", "description": "Extract text from DOCX, PDF, txt, and images using Tesseract/pdfplumber."},
    {"name": "Evaluation", "description": "Direct text evaluation bridging CV and JD."},
]
