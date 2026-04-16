# cv_eval/llm_scorer.py

import json, time, logging, os
from dotenv import load_dotenv
from .prompts import UNIFIED_EVALUATION_PROMPT, CV_ONLY_EVALUATION_PROMPT, IMPROVEMENT_PROMPT, CV_ONLY_IMPROVEMENT_PROMPT
from pydantic import BaseModel
from typing import List, Optional

from google import genai
from google.genai import types
from app.core.key_manager import key_manager


load_dotenv()
logger = logging.getLogger(__name__)

# --- Pydantic Schemas for Structured Output ---

class SubScore(BaseModel):
    dimension: str
    score: float
    max_score: float
    evidence: List[str]

class ScoreSection(BaseModel):
    overall_score: float
    subscores: List[SubScore]

class KeyTakeaways(BaseModel):
    red_flags: List[str]
    green_flags: List[str]

class EvaluateResponseSchema(BaseModel):
    cv_quality: ScoreSection
    jd_match: Optional[ScoreSection] = None
    key_takeaways: KeyTakeaways

class PersonalInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

class TailoredResume(BaseModel):
    personal_info: PersonalInfo
    summary: str
    experience: List[str]
    skills: List[str]
    projects: List[str]

class Top1PercentGap(BaseModel):
    strengths: List[str]
    gaps: List[str]
    actionable_next_steps: List[str]

class ImprovementResponseSchema(BaseModel):
    tailored_resume: TailoredResume
    top_1_percent_gap: Top1PercentGap
    cover_letter: str

# ----------------------------------------------------
class LLMScorer:
    def __init__(self, client=None, model=None, temperature=0.0, timeout=60):
        api_key = key_manager.get_gemini_key() or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")
        
        self.client = client or genai.Client(api_key=api_key)
        self.model = model or key_manager.get_gemini_model()
        self.temperature = temperature
        self.timeout = timeout
        # Cumulative token counters — reset per upload request
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def reset_usage(self) -> None:
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def get_usage(self) -> dict:
        return {
            "input_tokens":  self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    # ---------- CV vs JD (auto-switch) ----------
    def unified_evaluate(self, cv_text: str, jd_text: str = "") -> dict:
        if jd_text and jd_text.strip():
            prompt = UNIFIED_EVALUATION_PROMPT.format(cv_text=cv_text, jd_text=jd_text)
        else:
            prompt = CV_ONLY_EVALUATION_PROMPT.format(cv_text=cv_text)

        result = self._call_gemini(prompt, schema=EvaluateResponseSchema)
        return self._fix_overall_scores(result)

    # ---------- CV only (legacy alias) ----------
    def evaluate_cv_only(self, cv_text: str) -> dict:
        return self.unified_evaluate(cv_text=cv_text, jd_text="")

    # ---------- Improvement ----------
    def improvement(self, cv_text: str, jd_text: str = "") -> dict:
        if not cv_text.strip():
            raise ValueError("CV text is required for improvement")

        if jd_text and jd_text.strip():
            prompt = IMPROVEMENT_PROMPT.format(cv_text=cv_text, jd_text=jd_text)
        else:
            prompt = CV_ONLY_IMPROVEMENT_PROMPT.format(cv_text=cv_text)

        return self._call_gemini(prompt, schema=ImprovementResponseSchema)

    # ---------- Score consistency ----------
    @staticmethod
    def _fix_overall_scores(result: dict) -> dict:
        """
        Recompute overall_score as the exact sum of subscores.
        This prevents the LLM from hallucinating a different total
        and guarantees deterministic scores for the same subscore set.
        """
        for section_key in ("cv_quality", "jd_match"):
            section = result.get(section_key)
            if section and isinstance(section.get("subscores"), list) and section["subscores"]:
                computed = round(sum(s["score"] for s in section["subscores"]), 1)
                section["overall_score"] = computed
        return result

    # ---------- Core Gemini call with structured output ----------
    def _call_gemini(self, prompt: str, schema: type[BaseModel]) -> dict:
        """
        Call Gemini with structured output (response_schema + response_mime_type).
        JSON parsing happens inside the retry loop so malformed responses are retried.
        Uses seed=42 for deterministic output on the same input.
        """
        last_error = None

        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        seed=42,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )

                # Accumulate Gemini token usage
                um = getattr(resp, 'usage_metadata', None)
                if um:
                    in_tokens  = getattr(um, 'prompt_token_count',     0)
                    out_tokens = getattr(um, 'candidates_token_count', 0)
                    if out_tokens == 0:
                        out_tokens = getattr(um, 'total_token_count', 0) - in_tokens
                    self.total_input_tokens  += in_tokens
                    self.total_output_tokens += out_tokens
                    print(f"[CV-Eval] Gemini tokens: in={in_tokens}, out={out_tokens}  cumulative: in={self.total_input_tokens}, out={self.total_output_tokens}")

                # Parse response — inside retry loop so failures are retried
                raw_text = resp.text.strip()
                return json.loads(raw_text)

            except json.JSONDecodeError as jde:
                last_error = jde
                logger.warning(f"JSON parse failed (attempt {attempt+1}/3): {jde.msg} at char {jde.pos}")
                if attempt < 2:
                    time.sleep(1.5 ** attempt)

            except Exception as e:
                last_error = e
                logger.error(f"Gemini API call failed (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(1.5 ** attempt)

        # All retries exhausted
        raise last_error
