# cv_eval/llm_scorer.py


import json, re, time, logging, os
from dotenv import load_dotenv
from .prompts import UNIFIED_EVALUATION_PROMPT, CV_ONLY_EVALUATION_PROMPT, IMPROVEMENT_PROMPT, CV_ONLY_IMPROVEMENT_PROMPT


from google import genai
from google.genai import types
from app.core.key_manager import key_manager


load_dotenv()
logger = logging.getLogger(__name__)

# Schmeas for Gemini Response Structure
EVALUATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "cv_quality": {
            "type": "OBJECT",
            "properties": {
                "overall_score": {"type": "NUMBER"},
                "subscores": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "dimension": {"type": "STRING"},
                            "score": {"type": "NUMBER"},
                            "max_score": {"type": "NUMBER"},
                            "evidence": {"type": "ARRAY", "items": {"type": "STRING"}}
                        },
                        "required": ["dimension", "score", "max_score", "evidence"]
                    }
                }
            },
            "required": ["overall_score", "subscores"]
        },
        "jd_match": {
            "type": "OBJECT",
            "properties": {
                "overall_score": {"type": "NUMBER"},
                "subscores": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "dimension": {"type": "STRING"},
                            "score": {"type": "NUMBER"},
                            "max_score": {"type": "NUMBER"},
                            "evidence": {"type": "ARRAY", "items": {"type": "STRING"}}
                        },
                        "required": ["dimension", "score", "max_score", "evidence"]
                    }
                }
            },
            "required": ["overall_score", "subscores"]
        },
        "key_takeaways": {
            "type": "OBJECT",
            "properties": {
                "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
                "green_flags": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["red_flags", "green_flags"]
        }
    },
    "required": ["cv_quality", "key_takeaways"] # jd_match is optional for CV-only
}

IMPROVEMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tailored_resume": {
            "type": "OBJECT",
            "properties": {
                "personal_info": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "email": {"type": "STRING"},
                        "phone": {"type": "STRING"},
                        "location": {"type": "STRING"},
                        "linkedin": {"type": "STRING"},
                        "github": {"type": "STRING"},
                        "website": {"type": "STRING"}
                    }
                },
                "summary": {"type": "STRING"},
                "experience": {"type": "ARRAY", "items": {"type": "STRING"}},
                "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                "projects": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["personal_info", "summary", "experience", "skills", "projects"]
        },
        "top_1_percent_gap": {
            "type": "OBJECT",
            "properties": {
                "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
                "gaps": {"type": "ARRAY", "items": {"type": "STRING"}},
                "actionable_next_steps": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["strengths", "gaps", "actionable_next_steps"]
        },
        "cover_letter": {"type": "STRING"}
    },
    "required": ["tailored_resume", "top_1_percent_gap", "cover_letter"]
}

class LLMScorer:
    def __init__(self, client=None, model=None, temperature=0.0, timeout=60):
        api_key = key_manager.get_gemini_key() or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")
        
        self.client = client or genai.Client(api_key=api_key)
        self.model = model or key_manager.get_gemini_model()
        self.temperature = temperature
        self.timeout = timeout
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

    # ---------- Public API ----------
    def unified_evaluate(self, cv_text: str, jd_text: str = "") -> dict:
        if jd_text and jd_text.strip():
            prompt = UNIFIED_EVALUATION_PROMPT.format(cv_text=cv_text, jd_text=jd_text)
        else:
            prompt = CV_ONLY_EVALUATION_PROMPT.format(cv_text=cv_text)
        return self._call_and_parse(prompt, schema=EVALUATION_SCHEMA, label="evaluation")

    def evaluate_cv_only(self, cv_text: str) -> dict:
        return self.unified_evaluate(cv_text=cv_text, jd_text="")

    def improvement(self, cv_text: str, jd_text: str = "") -> dict:
        if not cv_text.strip():
            raise ValueError("CV text is required for improvement")
        if jd_text and jd_text.strip():
            prompt = IMPROVEMENT_PROMPT.format(cv_text=cv_text, jd_text=jd_text)
        else:
            prompt = CV_ONLY_IMPROVEMENT_PROMPT.format(cv_text=cv_text)
        return self._call_and_parse(prompt, schema=IMPROVEMENT_SCHEMA, label="improvement")

    # ---------- Core: call + parse with repair ----------
    def _call_and_parse(self, prompt: str, schema: dict = None, label: str = "llm") -> dict:
        raw = self._call_llm(prompt, schema=schema)
        cleaned = self._extract_json_from_response(raw)

        # Attempt 1: direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 2: repair common issues and parse
        repaired = self._repair_json(cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(f"[{label}] JSON parse failed after repair: {e}")
            logger.error(f"[{label}] Raw (first 500 chars): {raw[:500]}")
            raise ValueError(f"LLM {label} failed: {e}")

    # ---------- Gemini API call ----------
    def _call_llm(self, prompt: str, schema: dict = None) -> str:
        for attempt in range(3):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=(
                        "You are a strict JSON generator. "
                        "Rules: 1) Output ONLY valid JSON. "
                        "2) Keep evidence quotes under 15 words each. "
                        "3) Escape all special characters in strings. "
                        "4) Never use literal newlines inside string values."
                    ),
                    temperature=self.temperature,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                )
                if schema:
                    config.response_schema = schema

                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
                um = getattr(resp, 'usage_metadata', None)
                if um:
                    in_tokens  = getattr(um, 'prompt_token_count',     0)
                    out_tokens = getattr(um, 'candidates_token_count', 0)
                    if out_tokens == 0:
                        out_tokens = getattr(um, 'total_token_count', 0) - in_tokens
                    self.total_input_tokens  += in_tokens
                    self.total_output_tokens += out_tokens
                    print(f"[CV-Eval] Gemini call tokens: in={in_tokens}, out={out_tokens}  cumulative: in={self.total_input_tokens}, out={self.total_output_tokens}")

                text = resp.text.strip()
                if text and not text.endswith('}'):
                    logger.warning(f"[CV-Eval] Truncated JSON detected (attempt {attempt+1})")
                    if attempt < 2:
                        time.sleep(1)
                        continue
                return text
            except Exception as e:
                logger.error(f"Gemini API call failed (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    raise
                time.sleep(1.5 ** attempt)

    # ---------- JSON repair ----------
    @staticmethod
    def _repair_json(text: str) -> str:
        """Fix common JSON issues from LLM output."""
        # 1. Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # 2. Fix unescaped control characters inside strings
        #    Walk through and escape literal newlines/tabs inside string values
        result = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            if ch == '\\':
                result.append(ch)
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue
            if in_string:
                if ch == '\n':
                    result.append('\\n')
                    continue
                if ch == '\r':
                    result.append('\\r')
                    continue
                if ch == '\t':
                    result.append('\\t')
                    continue
            result.append(ch)
        text = ''.join(result)

        # 3. If JSON is truncated (missing closing brackets), try to close it
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        if open_brackets > 0:
            text += ']' * open_brackets
        if open_braces > 0:
            text += '}' * open_braces

        return text

    @staticmethod
    def _extract_json_from_response(text: str) -> str:
        if "```json" in text:
            s = text.find("```json") + 7
            e = text.find("```", s)
            return text[s:e].strip()
        elif "```" in text:
            s = text.find("```") + 3
            e = text.find("```", s)
            return text[s:e].strip()
        start, end = text.find("{"), text.rfind("}")
        return text[start:end+1] if start != -1 and end != -1 else text
