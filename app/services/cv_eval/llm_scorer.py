# cv_eval/llm_scorer.py


import json, time, logging, os
from dotenv import load_dotenv
from .prompts import UNIFIED_EVALUATION_PROMPT, CV_ONLY_EVALUATION_PROMPT, IMPROVEMENT_PROMPT, CV_ONLY_IMPROVEMENT_PROMPT


from google import genai
from google.genai import types
from app.core.key_manager import key_manager


load_dotenv()
logger = logging.getLogger(__name__)

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

        raw = self._call_llm(prompt)
        cleaned = self._extract_json_from_response(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as jde:
             # Last resort: try to fix common JSON issues if standard parse fails
             logger.error(f"JSON Parse Error: {jde} for text: {cleaned[:100]}...")
             raise json.JSONDecodeError(f"{jde.msg} (raw output was {len(cleaned)} chars)", jde.doc, jde.pos)

    # ---------- CV only (legacy alias) ----------
    def evaluate_cv_only(self, cv_text: str) -> dict:
        return self.unified_evaluate(cv_text=cv_text, jd_text="")

    # ---------- Internals ----------
    def _call_llm(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a strict JSON generator. Your output must be valid JSON only. Always escape double quotes and special characters within strings. Do not include trailing commas. Do not include markdown block tokens like ```json at the start or end of your response — return ONLY the raw JSON string.",
                        temperature=self.temperature,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    )
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
                    print(f"[CV-Eval] Gemini call tokens: in={in_tokens}, out={out_tokens}  cumulative: in={self.total_input_tokens}, out={self.total_output_tokens}")
                return resp.text.strip()
            except Exception as e:
                logger.error(f"Gemini API call failed (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    raise
                time.sleep(1.5 ** attempt)
    
    def improvement(self, cv_text: str, jd_text: str = "") -> dict:
        if not cv_text.strip():
            raise ValueError("CV text is required for improvement")

        if jd_text and jd_text.strip():
            prompt = IMPROVEMENT_PROMPT.format(cv_text=cv_text, jd_text=jd_text)
        else:
            prompt = CV_ONLY_IMPROVEMENT_PROMPT.format(cv_text=cv_text)

        raw = self._call_llm(prompt)
        cleaned = self._extract_json_from_response(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as jde:
            logger.error(f"JSON Parse Error (Improvement): {jde} for text: {cleaned[:100]}...")
            raise json.JSONDecodeError(f"{jde.msg} (raw output was {len(cleaned)} chars)", jde.doc, jde.pos)


    @staticmethod
    def _extract_json_from_response(text: str) -> str:
        # 1. Basic markdown stripping
        if "```json" in text:
            s = text.find("```json") + 7
            e = text.find("```", s)
            text = text[s:e].strip()
        elif "```" in text:
            s = text.find("```") + 3
            e = text.find("```", s)
            text = text[s:e].strip()
        
        # 2. Extract first level object/array
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        # 3. Robust Cleanup (common LLM failure modes)
        # a) Remove trailing commas before closing braces/brackets
        import re
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # b) Replace literal newlines inside values with \n if they are between quotes
        # (This is tricky but simple regex can help common cases)
        # text = re.sub(r'(".*?")', lambda m: m.group(1).replace('\n', '\\n'), text, flags=re.DOTALL)
        
        return text.strip()
