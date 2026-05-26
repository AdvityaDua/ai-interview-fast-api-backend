"""HTTP client for the fine-tuned LLaMA interview API.

Exposes the same interface as GeminiClient so it can be swapped in without
changing callers: summarize_context, generate_feedback, and the api_key /
model_name attributes used by StreamingInterviewSession.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import asyncio

import httpx

from .schemas import (
    BehavioralInsights,
    ConfidenceLevel,
    DimensionScores,
    EvaluationDetail,
    FinalEvaluation,
    ImprovementPlan,
    QuestionAnalysis,
    SkillGapAnalysis,
    Summary,
    SeniorityAssessment,
    Verdict,
)

LLAMA_BASE_URL: str = os.getenv(
    "LLAMA_API_URL", "https://nutlike-rants-earache.ngrok-free.dev"
).rstrip("/")

_HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}
_MAX_RETRIES = 2

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Python": ["python", "pandas", "numpy", "fastapi", "flask", "django"],
    "JavaScript": ["javascript", "typescript", "node.js", "react", "next.js", "vue"],
    "System Design": ["system design", "microservices", "scalability", "distributed", "architecture"],
    "Databases": ["sql", "postgres", "mysql", "mongodb", "redis", "database", "nosql"],
    "APIs": ["api", "rest", "graphql", "websocket", "grpc", "http"],
    "Testing": ["test", "pytest", "jest", "unit testing", "integration testing", "tdd"],
    "DevOps": ["docker", "kubernetes", "ci/cd", "aws", "gcp", "azure", "terraform"],
    "DSA": ["algorithm", "data structure", "leetcode", "dynamic programming", "graph", "tree", "linked list"],
    "Machine Learning": ["machine learning", "ml", "deep learning", "pytorch", "tensorflow", "transformer", "llm", "rag", "nlp"],
    "Behavioral": ["leadership", "teamwork", "conflict", "communication", "ownership", "collaboration"],
    "Problem Solving": ["problem solving", "analytical", "case study", "logical reasoning"],
    "Product Thinking": ["product", "trade-off", "user impact", "metrics", "stakeholder"],
}


def _extract_skills_from_text(text: str, jd_text: str = "", role: str = "") -> List[str]:
    combined = f"{text}\n{jd_text}\n{role}".lower()
    skills = [topic for topic, kws in _TOPIC_KEYWORDS.items() if any(kw in combined for kw in kws)]
    return skills[:12] if skills else ["Problem Solving", "Communication", "Technical Knowledge"]


class LlamaClient:
    """Drop-in replacement for GeminiClient backed by fine-tuned LLaMA models."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or LLAMA_BASE_URL).rstrip("/")
        # Compatibility stubs so existing code that reads these attributes works.
        self.api_key: str = "llama-finetuned"
        self.model_name: str = "llama-finetuned"
        # LlamaInterviewGraph stores accumulated evaluator scores here after each turn
        # so generate_feedback can aggregate them without an extra LLM call.
        self._llama_scores: List[dict] = []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0, headers=_HEADERS) as http:
                resp = await http.get(f"{self.base_url}/health")
                return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core API calls (used by LlamaInterviewGraph directly)
    # ------------------------------------------------------------------

    async def _post_with_retry(self, endpoint: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=35.0, headers=_HEADERS) as http:
                    resp = await http.post(f"{self.base_url}{endpoint}", json=payload)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(1.5)
        raise RuntimeError(f"[LlamaClient] {endpoint} failed: {last_exc}")

    async def get_question(
        self,
        jd: dict,
        conversation_history: List[dict],
        round_type: str,
        difficulty: str,
        current_topic: str,
    ) -> dict:
        return await self._post_with_retry("/interviewer/question", {
            "jd": jd,
            "conversation_history": conversation_history,
            "round_type": round_type,
            "difficulty": difficulty,
            "current_topic": current_topic,
        })

    async def evaluate_answer(
        self,
        jd: dict,
        question: str,
        answer: str,
        round_type: str,
        difficulty: str,
        previous_scores: List[dict],
    ) -> dict:
        return await self._post_with_retry("/evaluator/score", {
            "jd": jd,
            "question": question,
            "answer": answer,
            "round_type": round_type,
            "difficulty": difficulty,
            "previous_scores": previous_scores,
        })

    # ------------------------------------------------------------------
    # GeminiClient-compatible interface
    # ------------------------------------------------------------------

    async def summarize_context(
        self,
        resume_text: str,
        jd_text: str,
        interview_type: str = "technical",
        role: str = "",
        company: str = "",
        candidate_name: str = "",
    ) -> Tuple[str, dict]:
        """Template-based context builder — no LLM call needed for LLaMA flow."""
        name = candidate_name or "the candidate"
        has_jd = bool(jd_text and jd_text.strip())

        lines = [
            f"IDENTIFIED NAME: {name}",
            f"Role: {role or 'Not specified'}",
            f"Company: {company or 'Not specified'}",
            f"Interview Type: {interview_type}",
            "",
            "RESUME:",
            (resume_text[:2500] if resume_text else "Not provided"),
            "",
            "JOB DESCRIPTION:",
            (jd_text[:1500] if has_jd else "Not provided"),
        ]
        return "\n".join(lines), {"input_tokens": 0, "output_tokens": 0}

    def extract_skills(
        self,
        context_summary: str,
        interview_type: str,
        role: str,
        jd_text: str = "",
    ) -> List[str]:
        """Keyword-based skill extraction — replaces the Gemini skills call."""
        return _extract_skills_from_text(context_summary, jd_text, role)

    async def generate_feedback(
        self,
        history: List[dict],
        context_summary: str,
    ) -> Tuple[FinalEvaluation, dict]:
        """Aggregate accumulated evaluator scores into a FinalEvaluation report."""
        user_turns = [m for m in history if m.get("role") in ("user", "candidate")]
        if not user_turns:
            raise ValueError(
                "Insufficient Data: The candidate did not answer any questions. "
                "No meaningful evaluation can be generated."
            )

        scores: List[dict] = list(self._llama_scores)
        model_turns = [m for m in history if m.get("role") in ("model", "interviewer")]

        # Build per-question analysis
        q_analyses: List[QuestionAnalysis] = []
        for idx, (q_msg, u_msg) in enumerate(zip(model_turns, user_turns), start=1):
            entry = scores[idx - 1] if idx - 1 < len(scores) else {}
            raw = min(int(float(entry.get("final_score", 5) or 5)), 10)
            q_analyses.append(
                QuestionAnalysis(
                    question_id=idx,
                    question=q_msg.get("content", "")[:300],
                    user_answer_summary=u_msg.get("content", "")[:400],
                    score=raw,
                    evaluation=EvaluationDetail(
                        strengths=[entry.get("key_strength")] if entry.get("key_strength") else ["Attempted the question."],
                        weaknesses=[entry.get("key_gap")] if entry.get("key_gap") else ["Could improve depth and clarity."],
                        ideal_answer_outline=[entry.get("feedback")] if entry.get("feedback") else ["Review the topic thoroughly."],
                    ),
                )
            )

        # Overall score (0-100) from average final_score (0-10)
        if scores:
            avg = sum(min(float(s.get("final_score", 5) or 5), 10) for s in scores) / len(scores)
        else:
            avg = 5.0
        overall = min(100, int(avg * 10))

        # Hire recommendation
        yes_count = sum(1 for s in scores if str(s.get("hiring_signal", "")).lower() == "yes")
        hire_ratio = yes_count / len(scores) if scores else 0.5
        hire_text = "Strong Hire" if hire_ratio >= 0.75 else ("Hire" if hire_ratio >= 0.5 else "No Hire")

        # Seniority
        ctx = context_summary.lower()
        if any(k in ctx for k in ("senior", "lead", "principal", "staff")):
            seniority = SeniorityAssessment.SENIOR
        elif any(k in ctx for k in ("junior", "fresher", "intern", "entry")):
            seniority = SeniorityAssessment.JUNIOR
        else:
            seniority = SeniorityAssessment.MID

        gaps = [s["key_gap"] for s in scores if s.get("key_gap")]
        feedbacks = [s["feedback"] for s in scores if s.get("feedback")]
        follow_ups = [s["follow_up"] for s in scores if s.get("follow_up")]
        strengths = [s["key_strength"] for s in scores if s.get("key_strength")]

        dim = min(10, int(avg))
        evaluation = FinalEvaluation(
            summary=Summary(
                overall_score=overall,
                hire_recommendation=hire_text,
                seniority_assessment=seniority,
                confidence_assessment=ConfidenceLevel.MEDIUM,
            ),
            dimension_scores=DimensionScores(
                technical_depth=min(10, int(avg * 1.05)),
                problem_solving=dim,
                system_design=min(10, int(avg * 0.95)),
                communication=min(10, int(avg * 0.9)),
                role_fit=dim,
            ),
            question_wise_analysis=q_analyses,
            skill_gap_analysis=SkillGapAnalysis(
                critical_gaps=gaps[:3] or ["Review core technical concepts"],
                moderate_gaps=gaps[3:6] or ["Practice with additional examples"],
                minor_gaps=gaps[6:] or [],
            ),
            behavioral_insights=BehavioralInsights(
                communication_style="Articulate" if avg >= 6 else "Developing",
                thinking_pattern="Structured" if avg >= 6 else "Improving",
                pressure_handling="Composed" if avg >= 5 else "Needs practice",
            ),
            improvement_plan=ImprovementPlan(**{
                "immediate_actions": feedbacks[:2] or ["Review fundamentals and practice core problems."],
                "1_week_plan": follow_ups[:2] or ["Solve 5 problems on identified weak areas daily."],
                "1_month_plan": [
                    "Build a project applying the skills identified as gaps.",
                    "Schedule weekly mock interviews to track progress.",
                ],
            }),
            verdict=Verdict(
                strengths_to_highlight=strengths[:3] or ["Demonstrated willingness to engage."],
                areas_to_fix_before_next_interview=gaps[:2] or ["Deepen technical understanding."],
                final_recommendation_text=(
                    f"{hire_text}: Overall score {overall}/100. "
                    + ("Strong performance across evaluated areas." if overall >= 70
                       else "Key areas need improvement before re-interview.")
                ),
            ),
        )
        return evaluation, {"input_tokens": 0, "output_tokens": 0}
