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

        print(f"[AudioAnalysis] Answer {answer_idx} done: confidence={metrics.composite_confidence}, wpm={metrics.words_per_minute}")
    except Exception as e:
        print(f"[AudioAnalysis] Error analyzing answer {answer_idx}: {e}")
    finally:
        # Delete the audio file to free disk space
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"[AudioAnalysis] Deleted {audio_path}")
        except Exception as e:
            print(f"[AudioAnalysis] Error deleting {audio_path}: {e}")


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
                        # Fire background analysis — don't await it
                        asyncio.create_task(
                            _analyze_and_cleanup(user_id, audio_path, answer_idx)
                        )
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
    Connect here with `?token=YOUR_JWT`.
    Handles full bidirectional conversational state and feedback generation using Gemini.
    Supports session restoration from Redis on reconnect.
    Includes aggregated audio analysis metrics in the end-of-interview feedback.
    """
    user_id = user_payload.get("sub")
    await manager.connect(websocket, user_id)
    
    try:
        session, restored = await manager.get_or_create_session(user_id)
        client = session.client
        
        if restored and session.history:
            # ── Session restored from cache — send history back to client ──
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
            print(f"[WS] Session restored for user {user_id} ({len(session.history)} messages)")
        
        elif not session.history:
            # ── Brand new session — wait for init payload ──
            data = await websocket.receive_text()
            init_payload = json.loads(data)
            
            if init_payload.get("type") == "init":
                resume_text = init_payload.get("resume_text", "")
                jd_text = init_payload.get("jd_text", "")
                interview_type = init_payload.get("interview_type", "technical")
                role = init_payload.get("role", "")
                company = init_payload.get("company", "")
                duration = init_payload.get("duration", 0)
                
                await session.initialize_session(resume_text, jd_text, interview_type, role, company, duration)
                await manager.send_json({"type": "info", "content": "Context initialized."}, user_id)
                
                # Save initial context to Redis
                await manager.save_session_to_cache(user_id)
                
                await manager.send_json({"type": "stream_start"}, user_id)
                async for chunk in session.stream_response(None):
                    await manager.send_json({"type": "text", "content": chunk}, user_id)
                await manager.send_json({"type": "stream_end"}, user_id)
                
                # Save after first AI response
                await manager.save_session_to_cache(user_id)
            
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "message":
                user_text = payload.get("content")
                
                await manager.send_json({"type": "stream_start"}, user_id)
                async for chunk in session.stream_response(user_text):
                     await manager.send_json({"type": "text", "content": chunk}, user_id)
                await manager.send_json({"type": "stream_end"}, user_id)
                
                # Persist after every exchange
                await manager.save_session_to_cache(user_id)
                
                if getattr(session, 'ended', False):
                    await manager.send_json({"type": "info", "content": "Interview complete. Generating feedback..."}, user_id)
                    feedback = await client.generate_feedback(session.history, session.context_summary)
                    
                    # Include aggregated audio metrics
                    audio_summary = _aggregate_audio_metrics(manager.get_audio_metrics(user_id))
                    
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": feedback.model_dump(),
                        "audio_analysis": audio_summary,
                    }, user_id)
                    await manager.clear_session(user_id)
                    break 
            
            elif payload.get("type") == "end_session":
                await manager.send_json({"type": "info", "content": "Ending session. Generating your interview analysis..."}, user_id)
                try:
                    feedback = await client.generate_feedback(session.history, session.context_summary)
                    
                    # Include aggregated audio metrics
                    audio_summary = _aggregate_audio_metrics(manager.get_audio_metrics(user_id))
                    
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": feedback.model_dump(),
                        "audio_analysis": audio_summary,
                    }, user_id)
                except Exception as fb_err:
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": None,
                        "audio_analysis": _aggregate_audio_metrics(manager.get_audio_metrics(user_id)),
                        "error": str(fb_err)
                    }, user_id)
                await manager.clear_session(user_id)
                break

    except WebSocketDisconnect:
        # Do NOT clear session on disconnect — Redis keeps it alive for reconnect
        manager.disconnect(user_id)
        print(f"[WS] User {user_id} disconnected (session preserved in Redis)")
    except Exception as e:
        print(f"Error in stream_endpoint: {e}")
        manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass
