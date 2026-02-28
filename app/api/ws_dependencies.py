from fastapi import WebSocket, HTTPException, Query, WebSocketException, status
from app.core.security import verify_jwt_token

async def get_ws_current_user(websocket: WebSocket, token: str = Query(None)):
    """
    Dependency to authenticate WebSocket connections using a query parameter token.
    Browser WebSockets do not support adding custom headers like 'Authorization'.
    """
    if not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token query parameter")
    
    try:
        user_payload = verify_jwt_token(token)
        return user_payload
    except HTTPException as e:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(e.detail))
    except Exception:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
