from fastapi import Request, HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from app.core.security import verify_jwt_token


class AuthMiddleware:
    """
    Pure ASGI middleware for JWT authentication.
    Uses raw ASGI instead of BaseHTTPMiddleware to avoid breaking WebSocket connections.
    BaseHTTPMiddleware is known to intercept WebSocket upgrade requests and return 403.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # Let WebSocket connections pass through — auth is handled via Depends() on the route
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return

        # Only apply auth logic to HTTP requests
        if scope["type"] == "http":
            request = Request(scope, receive)
            path = request.url.path

            unprotected_paths = ["/docs", "/openapi.json", "/redoc", "/api/v1/health", "/api/v1/interview/", "/api/v1/code/"]

            if any(path.startswith(p) for p in unprotected_paths):
                await self.app(scope, receive, send)
                return

            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid Authorization header"}
                )
                await response(scope, receive, send)
                return

            token = auth_header.split(" ")[1]
            try:
                payload = verify_jwt_token(token)
                scope["state"] = {**scope.get("state", {}), "user": payload}
            except HTTPException as e:
                response = JSONResponse(status_code=e.status_code, content={"detail": e.detail})
                await response(scope, receive, send)
                return
            except Exception:
                response = JSONResponse(status_code=401, content={"detail": "Authentication failed"})
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
