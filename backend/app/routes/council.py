"""FastAPI Router for Autonomous Multi-Agent Agronomic Debate Council."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from backend.app.services.multi_agent_council import DebateRequest, run_council_debate

router = APIRouter(prefix="/api/council", tags=["multi-agent-council"])


@router.post("/debate")
async def debate(req: DebateRequest):
    """Trigger a multi-turn live agronomic debate between Pathologist, Soil Biologist, and Financial Actuary.

    Resolves treatment conflict with an Arbiter consensus verdict and Safety Integrity Signature.
    """
    try:
        verdict = run_council_debate(req)
        return verdict
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Debate Council error: {str(exc)}")
