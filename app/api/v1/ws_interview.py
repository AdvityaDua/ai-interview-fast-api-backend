import json
import asyncio
import os
from dataclasses import asdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.api.ws_manager import manager
from app.api.ws_dependencies import get_ws_current_user
from app.services.ai.stt_service import stt_service
from app.services.ai.audio_analyzer import AudioAnalyzer

ws_router = APIRouter()

answer_counts = {}

# Shared analyzer instance
audio_analyzer = AudioAnalyzer()


async def _analyze_and_cleanup(user_id: str, audio_path: str, answer_idx: int):
    """Background task: analyze audio, store metrics, delete file."""
    try:
        metrics = await audio_analyzer.analyze_audio_file(audio_path, answer_idx)
        metrics_dict = asdict(metrics)
        manager.add_audio_metrics(user_id, metrics_dict)

        # Persist updated metrics to Redis
        await manager.save_session_to_cache(user_id)

        pass
    except Exception as e:
        pass
    finally:
        # Delete the audio file to free disk space
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            pass


@ws_router.websocket("/transcribe/{client_id}")
async def stt_stream_endpoint(
    websocket: WebSocket, 
    client_id: str,
    user_payload: dict = Depends(get_ws_current_user)
):
    """
    **Real-time Speech-To-Text WebSocket**
    Connect here with `?token=YOUR_JWT` to stream raw audio bytes.
    Receives JSON with transcription updates (`"type": "partial"` or `"type": "final"`).
    After each answer, fires a background task to analyze audio quality.
    """
    user_id = user_payload.get("sub")
    await websocket.accept()

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
                        # Fire background analysis — don't await it
                        asyncio.create_task(
                            _analyze_and_cleanup(user_id, audio_path, answer_idx)
                        )
                    dg_stream = None
                    break 

    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass
    finally:
        if dg_stream:
            try:
                await dg_stream.finish()
                audio_path = dg_stream.save_audio(user_id, answer_idx)
                if audio_path:
                    asyncio.create_task(
                        _analyze_and_cleanup(user_id, audio_path, answer_idx)
                    )
            except Exception:
                pass


def _aggregate_audio_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate per-answer audio metrics into a session summary."""
    if not metrics_list:
        return {}

    valid = [m for m in metrics_list if m.get("word_count", 0) > 0]
    if not valid:
        return {"total_answers_analyzed": 0, "per_answer": metrics_list}

    n = len(valid)
    return {
        "total_answers_analyzed": len(metrics_list),
        "overall_confidence": round(sum(m["composite_confidence"] for m in valid) / n, 1),
        "avg_wpm": round(sum(m["words_per_minute"] for m in valid) / n, 1),
        "total_filler_count": sum(m["filler_word_count"] for m in valid),
        "avg_filler_rate": round(sum(m["filler_words_per_minute"] for m in valid) / n, 2),
        "avg_word_confidence": round(sum(m["avg_word_confidence"] for m in valid) / n, 4),
        "avg_speech_ratio": round(sum(m["speech_to_silence_ratio"] for m in valid) / n, 1),
        "per_answer": metrics_list,
    }


@ws_router.websocket("/chat/{client_id}")
async def stream_interview_endpoint(
    websocket: WebSocket, 
    client_id: str,
    user_payload: dict = Depends(get_ws_current_user)
):
    """
    **Full-Duplex Interview Chat WebSocket**
    """
    user_id = user_payload.get("sub")
    await manager.connect(websocket, user_id)
    print(f"[WS] Client {client_id} (User: {user_id}) connected.")
    
    try:
        session, restored = await manager.get_or_create_session(user_id)
        
        if restored and session.history:
            print(f"[WS] Restoring session for {user_id} with {len(session.history)} messages.")
            restored_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session.history
            ]
            await manager.send_json({
                "type": "restored",
                "messages": restored_messages,
                "interview_type": session.interview_type,
                "role": session.role,
                "company": session.company,
            }, user_id)
        
        elif not session.history:
            print(f"[WS] Waiting for init payload from {user_id}...")
            # We wrap this in a timeout to prevent hanging forever if client doesn't send init
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                init_payload = json.loads(data)
            except asyncio.TimeoutError:
                print(f"[WS] Timeout waiting for init from {user_id}")
                await manager.send_json({"type": "error", "content": "Initialization timeout. Please try again."}, user_id)
                await websocket.close()
                return

            if init_payload.get("type") == "init":
                resume_text = init_payload.get("resume_text", "")
                jd_text = init_payload.get("jd_text", "")
                interview_type = init_payload.get("interview_type", "technical")
                role = init_payload.get("role", "")
                company = init_payload.get("company", "")
                duration = init_payload.get("duration", 0)
                
                print(f"[WS] Initializing context for {user_id} ({interview_type} round)...")
                try:
                    await manager.send_json({"type": "info", "content": "Analyzing resume & role..."}, user_id)
                    await session.initialize_session(resume_text, jd_text, interview_type, role, company, duration)
                    await manager.send_json({"type": "info", "content": "Context initialized."}, user_id)
                    
                    await manager.save_session_to_cache(user_id)
                    
                    print(f"[WS] Starting first AI response for {user_id}")
                    await manager.send_json({"type": "stream_start"}, user_id)
                    async for chunk in session.stream_response(None):
                        await manager.send_json({"type": "text", "content": chunk}, user_id)
                    await manager.send_json({"type": "stream_end"}, user_id)
                    
                    await manager.save_session_to_cache(user_id)
                except Exception as e:
                    print(f"[WS] Initialization error for {user_id}: {str(e)}")
                    await manager.send_json({"type": "error", "content": f"AI Engine failed to initialize: {str(e)}"}, user_id)
                    # We don't close yet, maybe they can retry? But usually init failure is fatal for the session.
            
        # Main message loop
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "message":
                user_text = payload.get("content")
                print(f"[WS] Message from {user_id}: {user_text[:50]}...")
                
                try:
                    await manager.send_json({"type": "stream_start"}, user_id)
                    async for chunk in session.stream_response(user_text):
                         await manager.send_json({"type": "text", "content": chunk}, user_id)
                    await manager.send_json({"type": "stream_end"}, user_id)
                    
                    await manager.save_session_to_cache(user_id)
                    
                    if getattr(session, 'ended', False):
                        await manager.send_json({"type": "info", "content": "Interview complete. Generating feedback..."}, user_id)
                        feedback = await session.client.generate_feedback(session.history, session.context_summary)
                        
                        audio_summary = _aggregate_audio_metrics(manager.get_audio_metrics(user_id))
                        
                        await manager.send_json({
                            "type": "end_interview",
                            "feedback": feedback.model_dump(),
                            "audio_analysis": audio_summary,
                        }, user_id)
                        await manager.clear_session(user_id)
                        break 
                except Exception as e:
                    print(f"[WS] Error during response for {user_id}: {str(e)}")
                    await manager.send_json({"type": "error", "content": f"AI Engine error: {str(e)}"}, user_id)
                    await manager.send_json({"type": "stream_end"}, user_id)
            
            elif payload.get("type") == "end_session":
                print(f"[WS] Manual end session for {user_id}")
                await manager.send_json({"type": "info", "content": "Ending session. Generating analysis..."}, user_id)
                try:
                    feedback = await session.client.generate_feedback(session.history, session.context_summary)
                    audio_summary = _aggregate_audio_metrics(manager.get_audio_metrics(user_id))
                    
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": feedback.model_dump(),
                        "audio_analysis": audio_summary,
                    }, user_id)
                except Exception as fb_err:
                    print(f"[WS] Feedback generation error for {user_id}: {str(fb_err)}")
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": None,
                        "audio_analysis": _aggregate_audio_metrics(manager.get_audio_metrics(user_id)),
                        "error": str(fb_err)
                    }, user_id)
                await manager.clear_session(user_id)
                break

            elif payload.get("type") == "ping":
                await manager.send_json({"type": "pong"}, user_id)

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected.")
        manager.disconnect(user_id)
    except Exception as e:
        print(f"[WS] Unexpected error in stream endpoint: {str(e)}")
        manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass
