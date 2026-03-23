from fastapi import Request

async def get_current_user(request: Request):
    """
    Dependency to get the authenticated user from the request state.
    This assumes that AuthMiddleware has already verified the token and set request.state.user.
    """
    return getattr(request.state, "user", None)
