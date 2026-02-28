from fastapi import WebSocket
from app.services.ai.streaming_session import StreamingInterviewSession
from app.services.ai.gemini_client import GeminiClient

class ConnectionManager:
    def __init__(self):
        # Maps user ID (sub) to their active WebSocket connection
        self.active_connections: dict[str, WebSocket] = {}
        # Maps user ID to their active StreamingInterviewSession
        # This acts as the persistent session storage to allow reconnects
        self.sessions: dict[str, StreamingInterviewSession] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_json(self, data: dict, user_id: str):
        ws = self.active_connections.get(user_id)
        if ws:
            await ws.send_json(data)
            
    def get_or_create_session(self, user_id: str) -> StreamingInterviewSession:
        if user_id not in self.sessions:
            client = GeminiClient()
            self.sessions[user_id] = StreamingInterviewSession(client)
        return self.sessions[user_id]

    def clear_session(self, user_id: str):
        if user_id in self.sessions:
            del self.sessions[user_id]

manager = ConnectionManager()
