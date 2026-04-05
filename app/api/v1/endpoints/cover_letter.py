"""
cover_letter.py — POST /api/v1/cover-letter/generate

Accepts resume text + job description and returns a fully-structured
CoverLetterData JSON object ready to drop into the frontend templates.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional

from app.api.dependencies import get_current_user
from app.core.key_manager import key_manager
from google import genai

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateCoverLetterRequest(BaseModel):
    jd: str
    resumeText: str
    applicantName: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    companyName: Optional[str] = None
    roleTitle: Optional[str] = None


class CoverLetterResponse(BaseModel):
    cover_letter: dict


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(req: GenerateCoverLetterRequest) -> str:
    today = datetime.now().strftime("%B %d, %Y")

    # Hints the user may have pre-filled from the dialog
    hints = []
    if req.applicantName: hints.append(f"- Applicant name (user-provided): {req.applicantName}")
    if req.email:         hints.append(f"- Applicant email (user-provided): {req.email}")
    if req.phone:         hints.append(f"- Applicant phone (user-provided): {req.phone}")
    if req.location:      hints.append(f"- Applicant location (user-provided): {req.location}")
    if req.companyName:   hints.append(f"- Company name (user-provided): {req.companyName}")
    if req.roleTitle:     hints.append(f"- Role title (user-provided): {req.roleTitle}")
    hints_block = "\n".join(hints) if hints else "(none)"

    return f"""You are an expert career coach and professional writer. Your task is to generate a complete, compelling, highly personalised cover letter in strict JSON format.

=== INPUTS ===

TODAY'S DATE: {today}

USER-PROVIDED HINTS (trust these over anything you extract):
{hints_block}

--- RESUME TEXT ---
{req.resumeText}

--- JOB DESCRIPTION ---
{req.jd}

=== EXTRACTION RULES ===

From the RESUME TEXT, extract:
1. applicantName  — Full name on the resume. If already in hints, use the hint.
2. email          — Email address found in resume header/contact section. If in hints, use hint.
3. phone          — Phone number. Format nicely (e.g. +1 (555) 123-4567). If in hints, use hint.
4. location       — City and country/state from resume header (e.g. "Toronto, ON" or "San Francisco, CA"). If in hints, use hint.
5. linkedin       — LinkedIn URL if present in resume, else omit.
6. website        — Portfolio/GitHub/website if present in resume, else omit.

From the JOB DESCRIPTION, extract:
7. hiringManagerName  — If a specific name is mentioned (e.g. "Hiring contact: Sarah Jones"), use it. Otherwise use "Hiring Manager".
8. hiringManagerTitle — If a specific title is mentioned, use it. Otherwise use "Hiring Manager".
9. companyName        — Company name. If already in hints, use hint. Extract from JD if not in hints.
10. companyAddress    — Company address if mentioned in JD. If not found, use "[Company Address]".
11. roleTitle         — Exact job title from JD. If already in hints, use hint.

=== COVER LETTER WRITING RULES ===

- Write 3+ strong paragraphs tailored to THIS specific role and THIS specific company.
- openingParagraph: Express genuine enthusiasm for the specific role. Mention 1-2 concrete achievements or skills from the resume that directly match the top requirements from the JD. Do NOT use generic phrases like "I am writing to express my interest".
- bodyParagraphs (array of 2): Each paragraph must reference SPECIFIC skills, technologies, or experiences from the resume that match SPECIFIC requirements in the JD. Use numbers and metrics where present in the resume. Second paragraph should address something about the company's mission/product/culture from the JD and why the candidate is drawn to it.
- closingParagraph: Confident, professional close. Mention availability for an interview. Thank them for their time.
- salutation: "Sincerely"
- Avoid filler phrases. Be specific, confident, and professional.
- Match tone to the company: startup → energetic; enterprise → formal; creative → engaging.

=== OUTPUT FORMAT ===

Respond with ONLY valid JSON. No markdown fences, no prose, no explanation. The JSON must exactly match this schema:

{{
  "applicantName": "string",
  "email": "string",
  "phone": "string",
  "location": "string",
  "linkedin": "string or omit",
  "website": "string or omit",
  "date": "{today}",
  "hiringManagerName": "string",
  "hiringManagerTitle": "string",
  "companyName": "string",
  "companyAddress": "string",
  "roleTitle": "string",
  "openingParagraph": "string",
  "bodyParagraphs": ["string", "string"],
  "closingParagraph": "string",
  "salutation": "Sincerely"
}}
"""


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=CoverLetterResponse)
async def generate_cover_letter(
    body: GenerateCoverLetterRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    api_key = key_manager.get_gemini_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")

    # ── Enforce cover letter limit via NestJS backend ─────────────────────────
    # Use the internal endpoint (no JWT re-validation required) — pass the
    # user ID that this service already verified from the JWT. This matches
    # the same pattern used by analytics/ai-usage (no @UseGuards on NestJS side).
    try:
        import httpx
        from app.core.config import settings

        user_id = (user or {}).get("sub") or (user or {}).get("id") or (user or {}).get("userId")
        if not user_id:
            raise HTTPException(status_code=401, detail="Could not determine user identity")

        async with httpx.AsyncClient(timeout=8.0) as http_client:
            limit_resp = await http_client.post(
                f"{settings.BACKEND_URL}/cover-letters/check-and-use-internal",
                json={"userId": str(user_id)},
            )
        if limit_resp.status_code == 400:
            # Explicit limit-exceeded response from NestJS — block the request
            detail = limit_resp.json().get("message", "Cover letter limit reached")
            raise HTTPException(status_code=429, detail=detail)
        elif limit_resp.status_code not in (200, 201):
            # Any other unexpected status — fail-open so infra issues don't block users
            import logging
            logging.getLogger(__name__).warning(
                f"Cover letter limit check returned unexpected status "
                f"{limit_resp.status_code} — proceeding (fail-open)"
            )
    except HTTPException:
        raise
    except Exception as e:
        # Network error / timeout — fail-open, avoids service outage
        import logging
        logging.getLogger(__name__).warning(f"Cover letter limit check failed: {e}")
    # ─────────────────────────────────────────────────────────────────────────

    model_name = key_manager.get_gemini_model()
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(body)

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_name,
            contents=prompt,
        )
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI generation failed: {e}")

    # Strip markdown fences if the model wrapped despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise HTTPException(status_code=502, detail="AI returned unparseable response")
        data = json.loads(match.group())

    # Ensure required keys have sensible defaults if the model omitted them
    today = datetime.now().strftime("%B %d, %Y")
    defaults = {
        "applicantName":     body.applicantName or "Applicant",
        "email":             body.email or "",
        "phone":             body.phone or "",
        "location":          body.location or "",
        "date":              today,
        "hiringManagerName": "Hiring Manager",
        "hiringManagerTitle":"Hiring Manager",
        "companyName":       body.companyName or "the company",
        "companyAddress":    "",
        "roleTitle":         body.roleTitle or "the position",
        "openingParagraph":  "",
        "bodyParagraphs":    ["", ""],
        "closingParagraph":  "",
        "salutation":        "Sincerely",
    }
    for k, v in defaults.items():
        if k not in data or not data[k]:
            data[k] = v

    # bodyParagraphs must be a list
    if not isinstance(data.get("bodyParagraphs"), list):
        data["bodyParagraphs"] = [str(data["bodyParagraphs"])]

    # Report token usage to analytics (best-effort, non-blocking)
    try:
        import httpx
        from app.core.config import settings
        from app.core.security import verify_jwt_token

        user_id = str(user.get("sub") or user.get("_id") or user.get("id", "anonymous"))
        usage_meta = getattr(response, "usage_metadata", None)
        in_tok  = getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
        out_tok = getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0

        async def _report():
            payload = {
                "userId": user_id,
                "sessionId": f"cover_letter_{user_id}_{uuid.uuid4().hex[:8]}",
                "model": model_name,
                "inputTokens": in_tok,
                "outputTokens": out_tok,
                "totalTokens": in_tok + out_tok,
                "costUsd": (in_tok / 1_000_000) * 0.10 + (out_tok / 1_000_000) * 0.40,
                "source": "cover_letter",
                "interviewType": "",
            }
            async with httpx.AsyncClient() as c:
                await c.post(f"{settings.BACKEND_URL}/analytics/ai-usage", json=payload, timeout=5.0)

        asyncio.create_task(_report())
    except Exception:
        pass  # Never block the response for analytics

    return {"cover_letter": data}
