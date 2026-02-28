from app.services.cv_eval.llm_scorer import LLMScorer
import logging

logger = logging.getLogger(__name__)

class CVEvaluationEngine:
    def __init__(self):
        self.llm_scorer = LLMScorer()
    
    def evaluate(self, cv_text: str, jd_text: str = ""):
        """Evaluate CV using LLM scorer"""
        try:
            return self.llm_scorer.unified_evaluate(cv_text, jd_text)
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback to basic response
            return {
                "cv_quality": {
                    "overall_score": 0,
                    "band": "Error",
                    "subscores": []
                },
                "jd_match": {} if not jd_text else {"overall_score": 0, "band": "Error", "subscores": []},
                "fit_index": {} if not jd_text else {"score": 0, "band": "Error"}
            }

evaluation_engine = CVEvaluationEngine()