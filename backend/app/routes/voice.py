"""Voice question endpoint: audio in, transcript + answer out.

Speech recognition uses Gemini multimodal input, so it requires
``GEMINI_API_KEY``. Without a key the endpoint returns HTTP 503 with a clear
message and the clients fall back to typed input; nothing is faked.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/api/ai", tags=["voice"])

MAX_AUDIO_BYTES = 8 * 1024 * 1024
ALLOWED_MIME = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm", "audio/ogg", "audio/mp4", "audio/m4a"}


def voice_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def transcribe_and_answer(audio: bytes, mime: str, language: str, context: Dict[str, Any]) -> Dict[str, str]:
    """Ask Gemini to transcribe the question and answer it. Raises on failure."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY", "").strip())
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are KisanSeva Agronomist AI helping an Indian farmer.\n"
        f"Farm context (may be empty): {json.dumps(context)[:1500]}\n"
        f"The farmer asked a spoken question. Reply language: {language}.\n"
        "Return strictly two lines:\n"
        "TRANSCRIPT: <what the farmer said, in the spoken language>\n"
        f"ANSWER: <short practical answer in {language}, max 120 words>"
    )
    resp = model.generate_content([{"mime_type": mime, "data": audio}, prompt])
    text = (resp.text or "").strip()
    transcript, answer = "", text
    if "ANSWER:" in text:
        head, _, answer = text.partition("ANSWER:")
        transcript = head.replace("TRANSCRIPT:", "").strip()
    return {"transcript": transcript, "reply": answer.strip()}


@router.get("/voice/status")
async def voice_status():
    """Tell clients whether server-side voice questions are available."""
    return {
        "voice_question_enabled": voice_available(),
        "requires": None if voice_available() else "GEMINI_API_KEY",
        "read_aloud": "client-side (browser speech synthesis / expo-speech)",
    }


@router.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    language: str = Form("Hindi"),
    context: Optional[str] = Form(None),
):
    """Transcribe a spoken question and answer it in the chosen language."""
    if not voice_available():
        raise HTTPException(
            status_code=503,
            detail="Voice questions need GEMINI_API_KEY on the server. Type your question instead.",
        )
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio larger than 8 MB.")
    mime = (audio.content_type or "audio/wav").lower()
    if mime not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type '{mime}'.")
    try:
        ctx = json.loads(context) if context else {}
    except json.JSONDecodeError:
        ctx = {}
    try:
        result = transcribe_and_answer(data, mime, language, ctx)
    except Exception as exc:  # network/model errors must not crash the API
        raise HTTPException(status_code=502, detail=f"Speech service error: {exc}")
    return {"language": language, "source": "gemini-1.5-flash", **result}
