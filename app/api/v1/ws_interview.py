import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.api.ws_manager import manager
from app.api.ws_dependencies import get_ws_current_user
from app.services.ai.stt_service import stt_service

ws_router = APIRouter()

answer_counts = {}



@ws_router.websocket("/transcribe")
async def stt_stream_endpoint(
    websocket: WebSocket, 
    user_payload: dict = Depends(get_ws_current_user)
):
    """
    **Real-time Speech-To-Text WebSocket**
    Connect here with `?token=YOUR_JWT` to stream raw audio bytes.
    Receives JSON with transcription updates (`"type": "partial"` or `"type": "final"`).
    """
    user_id = user_payload.get("sub")
    await websocket.accept()
    print(f"[STT WS] Client connected (user: {user_id})")

    if user_id not in answer_counts:
        answer_counts[user_id] = 0
    answer_counts[user_id] += 1
    answer_idx = answer_counts[user_id]

    dg_stream = None

    try:
        async def on_partial(text):
            try:
                await websocket.send_json({"type": "partial", "text": text})
            except Exception:
                pass

        async def on_final(text):
            try:
                await websocket.send_json({"type": "final", "text": text})
            except Exception:
                pass

        dg_stream = await stt_service.create_stream(on_partial, on_final)

        while True:
            message = await websocket.receive()
            if "bytes" in message:
                await dg_stream.send(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    await dg_stream.finish()
                    audio_path = dg_stream.save_audio(user_id, answer_idx)
                    if audio_path:
                        print(f"[STT] Answer audio saved: {audio_path}")
                    dg_stream = None
                    break 

    except WebSocketDisconnect:
        print(f"[STT WS] Client disconnected (user: {user_id})")
    except Exception as e:
        print(f"[STT WS] Error: {e}")
    finally:
        if dg_stream:
            try:
                await dg_stream.finish()
                dg_stream.save_audio(user_id, answer_idx)
            except Exception:
                pass


@ws_router.websocket("/chat")
async def stream_interview_endpoint(
    websocket: WebSocket, 
    user_payload: dict = Depends(get_ws_current_user)
):
    """
    **Full-Duplex Interview Chat WebSocket**
    Connect here with `?token=YOUR_JWT`.
    Handles full bidirectional conversational state and feedback generation using Gemini.
    """
    user_id = user_payload.get("sub")
    await manager.connect(websocket, user_id)
    
    try:
        session = manager.get_or_create_session(user_id)
        client = session.client
        
        # Determine if we need to initialize or just resume session
        if not session.history:
            data = await websocket.receive_text()
            init_payload = json.loads(data)
            
            if init_payload.get("type") == "init":
                resume_text = init_payload.get("resume_text", "")
                jd_text = init_payload.get("jd_text", "")
                interview_type = init_payload.get("interview_type", "technical")
                role = init_payload.get("role", "")
                company = init_payload.get("company", "")
                
                await session.initialize_session(resume_text, jd_text, interview_type, role, company)
                await manager.send_json({"type": "info", "content": "Context initialized."}, user_id)
                
                await manager.send_json({"type": "stream_start"}, user_id)
                async for chunk in session.stream_response(None):
                    await manager.send_json({"type": "text", "content": chunk}, user_id)
                await manager.send_json({"type": "stream_end"}, user_id)
        else:
            await manager.send_json({"type": "info", "content": "Session reconnected."}, user_id)
            
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "message":
                user_text = payload.get("content")
                
                await manager.send_json({"type": "stream_start"}, user_id)
                async for chunk in session.stream_response(user_text):
                     await manager.send_json({"type": "text", "content": chunk}, user_id)
                await manager.send_json({"type": "stream_end"}, user_id)
                
                if getattr(session, 'ended', False):
                    await manager.send_json({"type": "info", "content": "Interview complete. Generating feedback..."}, user_id)
                    feedback = await client.generate_feedback(session.history, session.context_summary)
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": feedback.model_dump()
                    }, user_id)
                    manager.clear_session(user_id)
                    break 
            
            elif payload.get("type") == "end_session":
                await manager.send_json({"type": "info", "content": "Ending session. Generating your interview analysis..."}, user_id)
                try:
                    feedback = await client.generate_feedback(session.history, session.context_summary)
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": feedback.model_dump()
                    }, user_id)
                except Exception as fb_err:
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": None,
                        "error": str(fb_err)
                    }, user_id)
                manager.clear_session(user_id)
                break

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        print(f"Error in stream_endpoint: {e}")
        manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass
