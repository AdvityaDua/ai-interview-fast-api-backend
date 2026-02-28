from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.security import verify_jwt_token

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow openapi docs, healthchecks, and WebSockets (auth handled via Depends)
        unprotected_paths = ["/docs", "/openapi.json", "/redoc", "/api/v1/health", "/api/v1/interview/"]
        
        if any(request.url.path.startswith(path) for path in unprotected_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"}
            )
            
        token = auth_header.split(" ")[1]
        try:
            payload = verify_jwt_token(token)
            request.state.user = payload
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e:
            return JSONResponse(status_code=401, content={"detail": "Authentication failed"})

        response = await call_next(request)
        return response
