import time
import random
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from .gemini_client import GeminiClient
from .streaming_session import StreamingInterviewSession
from .interview_graph import InterviewGraph, InterviewState, _max_questions_for_duration, _max_questions_per_topic, _max_confusion_retries_for_duration
from .schemas import QuestionEvaluation
from .question_bank_service import load_question_bank

class CompanyGeminiClient(GeminiClient):
    """Specialized Gemini Client for Company-specific interviews."""
    pass

class CompanyInterviewSession(StreamingInterviewSession):
    """
    New Interviewer type that leverages RAG knowledge and advanced greeting logic.
    Shares the same base reporting and session management as the default interviewer.
    """
    def __init__(self, client: CompanyGeminiClient):
        super().__init__(client)
        # Use the specialized client
        self.client = client
        self.graph_engine = InterviewGraph(api_key=client.api_key)

    _QB_MAX_TURNS = 3  # force-rotate after this many turns on the same question

    async def _fetch_question_pool(self, company: str) -> List[str]:
        """
        Look up the topic by company name, load its question bank JSON,
        return a shuffled flat list of all questions.
        Returns [] if no question bank is found.
        """
        if not company or len(company) < 2:
            return []
        try:
            target_url = f"{settings.BACKEND_URL}/knowledge/topics/find-by-name?name={company}&all=true"
            async with httpx.AsyncClient() as http:
                res = await http.get(target_url, timeout=5.0)
                if res.status_code == 200:
                    topic = res.json()
                    if topic and "_id" in topic:
                        bank = load_question_bank(topic["_id"])
                        if bank:
                            # Entry questions first, then the rest — then shuffle each group
                            entry = list(bank.get("entry_questions", []))
                            rest = [q for q in bank.get("all_questions", []) if q not in entry]
                            random.shuffle(entry)
                            random.shuffle(rest)
                            pool = entry + rest
                            print(f"[CompanySession] ✅ QB pool: {len(pool)} questions for '{company}'")
                            return pool
        except Exception as e:
            print(f"[CompanySession] ⚠️ QB lookup error for '{company}': {e}")
        return []

    async def initialize_session(self, *args, **kwargs):
        # 1. Normal resume/JD summarization
        await super().initialize_session(*args, **kwargs)

        company = kwargs.get("company", args[6] if len(args) > 6 else "")

        # 2. Load question pool (does NOT inject all questions into context_summary)
        pool = await self._fetch_question_pool(company)
        if pool:
            self.state["qb_pool"] = pool
            self.state["qb_current"] = pool[0]
            self.state["qb_turns_on_current"] = 0
            # Store the base context_summary WITHOUT current question so we can
            # rebuild it cheaply on every turn.
            self.state["_qb_cs_base"] = self.state["context_summary"]
            self.state["has_jd"] = True
            print(f"[CompanySession] QB ready: {len(pool)} questions queued")

    def _inject_current_question(self) -> None:
        """Prepend only the current question into context_summary before each graph turn."""
        current = self.state.get("qb_current", "")
        base = self.state.get("_qb_cs_base", "")
        if not current or not base:
            return
        self.state["context_summary"] = (
            base
            + f"\n\n{'=' * 60}\n"
            + f"QUESTION BANK MODE — {self.state.get('company', '').upper()}\n"
            + f"{'=' * 60}\n"
            + f"CURRENT QUESTION (ask this now, verbatim or naturally rephrased):\n"
            + f"  → {current}\n\n"
            + f"INSTRUCTIONS:\n"
            + f"  • This turn: ask the CURRENT QUESTION above.\n"
            + f"  • Do 1-2 targeted follow-ups based on the candidate's answer.\n"
            + f"  • When follow-ups are exhausted, set next_step.type = 'new_topic'\n"
            + f"    so the system can serve you the next question from the bank.\n"
            + f"{'=' * 60}\n"
        )

    def _rotate_question(self, evaluation) -> None:
        """Advance to the next question when the LLM signals new_topic or after max turns."""
        if "qb_pool" not in self.state:
            return
        self.state["qb_turns_on_current"] = self.state.get("qb_turns_on_current", 0) + 1
        turns = self.state["qb_turns_on_current"]
        next_type = getattr(getattr(evaluation, "next_step", None), "type", None)
        should_rotate = (str(next_type) == "new_topic") or (turns >= self._QB_MAX_TURNS)
        if should_rotate:
            pool = self.state["qb_pool"]
            if len(pool) > 1:
                # Cycle: move current to end so we don't repeat early
                pool = pool[1:] + [pool[0]]
                self.state["qb_pool"] = pool
                self.state["qb_current"] = pool[0]
                self.state["qb_turns_on_current"] = 0
                print(f"[CompanySession] → Next QB question: {pool[0][:70]}")

    async def stream_response(self, user_input: str = None):
        """Override to inject/rotate the current question around each graph turn."""
        # Inject current question into context_summary BEFORE the graph runs
        self._inject_current_question()

        evaluation = None
        async for chunk in super().stream_response(user_input):
            yield chunk

        # After the turn completes, rotate if needed
        evaluation = self.state.get("current_evaluation")
        if evaluation:
            self._rotate_question(evaluation)
