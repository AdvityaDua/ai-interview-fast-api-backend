"""Token-optimized interview agent.

This is the replacement for the previous LangGraph-based flow. It performs one
compact LLM call per turn, using a compressed prompt and a single structured
output schema that combines answer evaluation and next-question generation.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from app.core.key_manager import key_manager
from .llama_client import LlamaClient
from .llama_graph import LlamaInterviewGraph
from .source_interview_planner import question_fingerprint
from .schemas import (
    Action,
    AnswerEvaluation,
    ClarityDepth,
    ConfidenceLevel,
    Correctness,
    Difficulty,
    Decision,
    NextStep,
    NextStepType,
    SignalStrength,
)


class InterviewState(TypedDict):
    history: List[dict]
    performance_summary: str
    context_summary: str
    has_jd: bool
    skills_remaining: List[str]
    skills_covered: List[str]
    questions_asked: List[str]
    last_user_input: Optional[str]
    current_question: str
    current_evaluation: Optional[dict]
    follow_up_hint: str
    interview_type: str
    time_context: str
    role: str
    company: str
    is_developer_role: bool
    coding_questions_asked: int
    last_question_was_coding: bool
    turn_number: int
    last_answer_type: str
    consecutive_non_answers: int
    ended: bool
    end_reason: str
    max_questions: int
    max_questions_per_topic: int
    topic_question_counts: Dict[str, int]
    max_confusion_retries: int
    consecutive_disengaged: int
    input_tokens: int
    output_tokens: int


class OptimizedTurnOutput(BaseModel):
    performance_summary: str = Field(..., description="Concise updated summary of the candidate's trajectory.")
    answer_type: str = Field(
        "genuine_answer",
        description="One of: genuine_answer, confused, refused, off_topic, incomplete, wait_requested, end_requested, not_applicable",
    )
    answer_quality: str = Field(..., description="One of: strong, adequate, weak, incorrect, not_applicable")
    should_follow_up: bool = False
    follow_up_hint: str = ""
    newly_covered_skills: List[str] = Field(default_factory=list)
    last_answer_evaluation: Optional[AnswerEvaluation] = None
    decision: Decision
    next_step: NextStep
    confidence_in_candidate: ConfidenceLevel = ConfidenceLevel.MEDIUM


def _max_questions_for_duration(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 8
    if duration_minutes <= 30:
        return 13
    if duration_minutes <= 45:
        return 16
    return 18


def _max_questions_per_topic(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 1
    if duration_minutes <= 30:
        return 2
    if duration_minutes <= 45:
        return 2
    return 3


def _max_confusion_retries_for_duration(duration_minutes: int) -> int:
    if duration_minutes <= 0:
        return 2
    if duration_minutes <= 15:
        return 1
    return 2


def _take_recent(items: List[str], limit: int) -> List[str]:
    if limit <= 0:
        return []
    return items[-limit:]


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_text_for_comparison(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _looks_like_greeting(text: str) -> bool:
    normalized = _normalize_text_for_comparison(text)
    return normalized.startswith(("hi", "hello", "good morning", "good afternoon", "good evening"))


def _looks_like_closing(text: str) -> bool:
    normalized = _normalize_text_for_comparison(text)
    return normalized.startswith(("thank you", "thanks", "that concludes", "we appreciate", "it was great"))


def _pick_coding_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    coding_candidates = [item for item in candidates if isinstance(item, dict) and item.get("is_coding_question")]
    if coding_candidates:
        return coding_candidates[0]
    return None


class InterviewGraph:
    """Interview engine — delegates to LlamaInterviewGraph backed by fine-tuned LLaMA."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: LlamaClient | None = None,
    ):
        llama_client = client if isinstance(client, LlamaClient) else LlamaClient()
        self._llama_graph = LlamaInterviewGraph(client=llama_client)
        self.client = llama_client
        self.model_name = "llama-finetuned"

    async def run_turn(self, state: InterviewState) -> InterviewState:
        return await self._llama_graph.run_turn(state)

    @staticmethod
    def _bump_topic_counts(state: InterviewState, target_skill: str) -> Dict[str, int]:
        return LlamaInterviewGraph._bump_topic_counts(state, target_skill)

    # Keep these so nothing that imports them breaks; they are no longer called.
    def _build_system_instruction(self, state: InterviewState) -> str:
        interview_type = state.get("interview_type", "technical")
        role = state.get("role", "")
        company = state.get("company", "")
        has_jd = state.get("has_jd", False)
        return f"""
You are a compact interview engine for a {interview_type} interview.
Role: {role or 'Not specified'}
Company: {company or 'Not specified'}
JD present: {has_jd}

Your job:
- Evaluate the user's answer briefly.
- Update the performance summary compactly.
- Decide whether to follow up, move on, or end.
- Generate exactly one next question.
- Keep prompts and outputs concise to reduce tokens.
- Ask a coding question only when it is actually required.
- Do not restate the entire history.
- Use the candidate's resume/JD context already supplied in the state.
- Use the provided source-backed question candidates when available.
- For technical interviews, balance company, resume, and JD evidence instead of leaning on only one source.
- Start with a brief greeting on the first turn.
- End with a polite closing statement when the interview is finished.
""".strip()

    def _build_prompt(self, state: InterviewState) -> str:
        skills_remaining = _take_recent(state.get("skills_remaining", []), 12)
        skills_covered = _take_recent(state.get("skills_covered", []), 12)
        questions_asked = _take_recent(state.get("questions_asked", []), 8)
        recent_history = state.get("history", [])[-6:]
        history_text = "\n".join(
            f"{item.get('role', 'unknown').upper()}: {item.get('content', '')}" for item in recent_history
        ) or "(No prior exchange)"
        last_user_input = state.get("last_user_input") or ""
        current_question = state.get("current_question") or ""
        time_context = state.get("time_context") or ""
        opening_line = state.get("opening_line") or ""
        closing_line = state.get("closing_line") or ""
        candidates = state.get("source_question_candidates", []) or []
        max_questions = int(state.get("max_questions", 0) or 0)
        questions_used = len(questions_asked)
        questions_left = max(0, max_questions - questions_used) if max_questions > 0 else 0

        return f"""
STATE SNAPSHOT
- turn_number: {state.get('turn_number', 0)}
- interview_type: {state.get('interview_type', '')}
- role: {state.get('role', '')}
- company: {state.get('company', '')}
- time_context: {time_context}
- ended: {state.get('ended', False)}
- end_reason: {state.get('end_reason', '')}
- last_answer_type: {state.get('last_answer_type', '')}
- consecutive_non_answers: {state.get('consecutive_non_answers', 0)}
- consecutive_disengaged: {state.get('consecutive_disengaged', 0)}
- questions_used: {questions_used}
- questions_left: {questions_left}
- max_questions_per_topic: {state.get('max_questions_per_topic', 0)}
- max_confusion_retries: {state.get('max_confusion_retries', 0)}
- is_developer_role: {state.get('is_developer_role', False)}
- question_source_mode: {state.get('question_source_mode', '')}

COMPRESSED CONTEXT
{state.get('context_summary', '')}

SKILLS TO COVER
{', '.join(skills_remaining) if skills_remaining else '(none)'}

SKILLS ALREADY COVERED
{', '.join(skills_covered) if skills_covered else '(none)'}

PREVIOUS QUESTIONS
{chr(10).join(f'- {q}' for q in questions_asked) if questions_asked else '(none)'}

RECENT HISTORY
{history_text}

CURRENT TURN
- current_question: {current_question}
- last_user_input: {last_user_input or '(none)'}
- opening_line: {opening_line or '(none)'}
- closing_line: {closing_line or '(none)'}

SOURCE-BACKED QUESTION CANDIDATES
{chr(10).join(f'- {item.get("question", "")} [{item.get("topic", "")}]' for item in candidates[:5]) if candidates else '(none)'}

RULES
- Produce one concise next question only.
- If the candidate is confused, rephrase simply and stay on the same topic.
- If the candidate asks for coding, return a coding question only when appropriate.
- Avoid repeating previously asked questions.
- For technical interviews, prefer a balanced mix of company, resume, and JD-backed candidates.
- Prefer the smallest prompt possible that still moves the interview forward.
- If time is running out or budget is exhausted, end cleanly.
- If source-backed candidates are available, prefer them over inventing a new question.

Return JSON matching the schema exactly.
""".strip()

