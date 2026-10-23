"""FastAPI Router for Kisan Vaani Vernacular Agro-Voice Engine."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.services.vernacular_voice_service import (
    process_vernacular_query,
    VERNACULAR_INTENT_REGISTRY,
)

router = APIRouter(prefix="/api/vaani", tags=["kisan-vaani"])


class VaaniQueryRequest(BaseModel):
    query_text: str = Field("फसल की पत्तियों में पीलापन आ रहा है क्या खाद डालें?", description="Farmer voice or text query")
    preferred_dialect: str = Field("Hindi", description="Preferred output dialect: Hindi, Hinglish, Punjabi")


@router.get("/sample-prompts")
async def get_sample_prompts():
    """Returns catalog of frequent vernacular farmer queries."""
    return {
        "intents": [item["intent"] for item in VERNACULAR_INTENT_REGISTRY],
        "prompts": [
            {"label": "🍂 पीलापन व खाद", "query": "फसल की पत्तियों में पीलापन आ रहा है क्या खाद डालें?"},
            {"label": "🐛 कीट व इल्ली रोकथाम", "query": "पत्तियों में छेद हो रहे हैं और कीड़ा लगा है कौन सी दवा छिड़कें?"},
            {"label": "🌾 यूरिया डालने का सही समय", "query": "यूरिया और डीएपी खाद किस समय और कितनी मात्रा में देना चाहिए?"},
            {"label": "🌧️ बारिश पूर्व छिड़काव", "query": "कल बारिश होने वाली है क्या आज कीटनाशक का छिड़काव कर सकते हैं?"},
            {"label": "💰 मंडी में सही भाव", "query": "मंडी में फसल बेचने का सही समय क्या है और रेट कैसे मिलेगा?"},
        ],
    }


@router.post("/vernacular-query")
async def execute_vernacular_query(payload: VaaniQueryRequest):
    """Processes farmer query and returns structured vernacular cards and audio payload."""
    return process_vernacular_query(
        query_text=payload.query_text,
        preferred_dialect=payload.preferred_dialect,
    )


@router.get("/demo-vaani")
async def demo_vaani(topic: str = Query("chlorosis")):
    """Quick 1-click test endpoint for Kisan Vaani."""
    if topic == "pest":
        q = "खेत में कीड़ा लगा है कौन सी दवा डालें"
    elif topic == "fertilizer":
        q = "यूरिया और डीएपी कब डालना चाहिए"
    else:
        q = "पत्तियों में पीलापन आ रहा है क्या करें"
    return process_vernacular_query(query_text=q)
