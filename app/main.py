import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, API_DESCRIPTION, TAGS_METADATA
from app.core.middleware import AuthMiddleware
from app.api.v1.router import api_router
from app.api.v1.ws_interview import ws_router as interview_ws_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Interview Coach Backend",
        description=API_DESCRIPTION,
        version="2.0.0",
        openapi_tags=TAGS_METADATA
    )

    # Add custom JWT Auth Middleware
    app.add_middleware(AuthMiddleware)

    # Configure CORS - added after Auth so it wraps it as the outermost layer
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "https://aiforjob.ai", "https://www.aiforjob.ai", "https://api.aiforjob.ai", "https://www.api.aiforjob.ai"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(interview_ws_router, prefix="/api/v1/interview")

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
