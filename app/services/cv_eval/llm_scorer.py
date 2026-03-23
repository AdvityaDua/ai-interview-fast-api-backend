# cv_eval/llm_scorer.py


import json, time, logging, os
from dotenv import load_dotenv
from .prompts import UNIFIED_EVALUATION_PROMPT, CV_ONLY_EVALUATION_PROMPT, IMPROVEMENT_PROMPT, CV_ONLY_IMPROVEMENT_PROMPT


load_dotenv()
logger = logging.getLogger(__name__)

class LLMScorer:
    def __init__(self, client=None, model=None, temperature=0.0, timeout=60):
        from groq import Groq
        from app.core.key_manager import key_manager
        groq_key = key_manager.get_groq_key() or os.getenv("GROQ_API_KEY")
        self.client = client or (Groq(api_key=groq_key) if groq_key else Groq())
        self.model = model or key_manager.get_groq_model()
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
        return json.loads(cleaned)

    # ---------- CV only (legacy alias) ----------
    def evaluate_cv_only(self, cv_text: str) -> dict:
        return self.unified_evaluate(cv_text=cv_text, jd_text="")

    # ---------- Internals ----------
    def _call_llm(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a strict JSON generator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=3500,
                )
                # Accumulate Groq token usage
                if hasattr(resp, 'usage') and resp.usage:
                    self.total_input_tokens  += getattr(resp.usage, 'prompt_tokens',     0)
                    self.total_output_tokens += getattr(resp.usage, 'completion_tokens', 0)
                    print(f"[CV-Eval] Groq call tokens: in={resp.usage.prompt_tokens}, out={resp.usage.completion_tokens}  cumulative: in={self.total_input_tokens}, out={self.total_output_tokens}")
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq API call failed (attempt {attempt+1}/3): {e}")
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
        return json.loads(cleaned)


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
