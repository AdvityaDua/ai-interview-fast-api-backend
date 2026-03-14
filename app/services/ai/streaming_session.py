import time
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from .gemini_client import GeminiClient
from .interview_graph import InterviewGraph, InterviewState
from .schemas import Action

class StreamingInterviewSession:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.graph_engine = InterviewGraph(api_key=client.api_key) # Share API key
        self.state: InterviewState = {
            "history": [],
            "performance_summary": "The interview is just starting.",
            "context_summary": "",
            "last_user_input": None,
            "current_evaluation": None,
            "interview_type": "technical",
            "time_context": "",
            "role": "",
            "company": "",
            "ended": False,
            # New accuracy fields
            "questions_asked": [],
            "current_question": "",
            "follow_up_hint": "",
            "is_developer_role": False,
            "coding_questions_asked": 0,
            "turn_number": 0,
            "last_answer_type": "not_applicable",
            "consecutive_non_answers": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        self.start_time: float = 0.0
        self.duration_limit: int = 0 

    @property
    def history(self):
        return self.state["history"]

    @history.setter
    def history(self, value):
        self.state["history"] = value

    @property
    def ended(self):
        return self.state["ended"]

    @ended.setter
    def ended(self, value):
        self.state["ended"] = value

    @property
    def context_summary(self):
        return self.state["context_summary"]

    @context_summary.setter
    def context_summary(self, value):
        self.state["context_summary"] = value

    @property
    def interview_type(self):
        return self.state["interview_type"]

    @interview_type.setter
    def interview_type(self, value):
        self.state["interview_type"] = value

    @property
    def role(self):
        return self.state["role"]

    @role.setter
    def role(self, value):
        self.state["role"] = value

    @property
    def company(self):
        return self.state["company"]

    @company.setter
    def company(self, value):
        self.state["company"] = value

    @property
    def skills_remaining(self):
        return self.state.get("skills_remaining", [])

    @skills_remaining.setter
    def skills_remaining(self, value):
        self.state["skills_remaining"] = value

    @property
    def skills_covered(self):
        return self.state.get("skills_covered", [])

    @skills_covered.setter
    def skills_covered(self, value):
        self.state["skills_covered"] = value

    @property
    def performance_summary(self):
        return self.state.get("performance_summary", "")

    @performance_summary.setter
    def performance_summary(self, value):
        self.state["performance_summary"] = value

    @property
    def user_id(self):
        return self.state.get("user_id")

    @user_id.setter
    def user_id(self, value):
        self.state["user_id"] = value

    @property
    def session_id(self):
        return self.state.get("session_id")

    @session_id.setter
    def session_id(self, value):
        self.state["session_id"] = value

    @property
    def input_tokens(self):
        return self.state.get("input_tokens", 0)

    @input_tokens.setter
    def input_tokens(self, value):
        self.state["input_tokens"] = value

    @property
    def output_tokens(self):
        return self.state.get("output_tokens", 0)

    @output_tokens.setter
    def output_tokens(self, value):
        self.state["output_tokens"] = value

    async def initialize_session(self, user_id: str, session_id: str, resume_text: str, jd_text: str, interview_type: str = "technical", role: str = "", company: str = "", duration: int = 0, candidate_name: str = ""):
        self.start_time = time.time()
        self.duration_limit = duration
        
        # Static context summarization - now returns usage info too
        context, context_usage = await self.client.summarize_context(resume_text, jd_text, interview_type, role, company, candidate_name)
        
        # Track initialization tokens
        init_input_tokens = context_usage.get("input_tokens", 0)
        init_output_tokens = context_usage.get("output_tokens", 0)
        
        # Extract skills for structured tracking (one-time init call, saves tokens on every turn)
        # For developer/technical roles get 12 specific skills; for others, 10 broader ones
        _dev_hint = (
            "Include specific technologies, languages, frameworks, and algorithms relevant "
            "to the role. Be granular (e.g. 'React hooks', 'database indexing', 'REST API design')."
            if interview_type in ("technical", "problem")
            else "Include both technical and soft skills relevant to the role and interview type."
        )
        skills_prompt = (
            f"Based on this context summary, list the top 12 skills/topics to evaluate in a "
            f"{interview_type} interview for a '{role or 'the role'}' candidate. "
            f"{_dev_hint} "
            f"Return ONLY a comma-separated list, no numbering, no explanation.\n\n{context}"
        )
        skills_res = await self.client.client.aio.models.generate_content(model=self.client.model_name, contents=skills_prompt)
        initial_skills = [s.strip() for s in skills_res.text.split(',') if s.strip()][:12]
        
        # Track skills extraction tokens
        if hasattr(skills_res, 'usage_metadata'):
            init_input_tokens += getattr(skills_res.usage_metadata, 'prompt_token_count', 0)
            init_output_tokens += getattr(skills_res.usage_metadata, 'candidates_token_count', 0)
            print(f"[Session] skills extraction usage: in={getattr(skills_res.usage_metadata, 'prompt_token_count', 0)}, out={getattr(skills_res.usage_metadata, 'candidates_token_count', 0)}")
        
        print(f"[Session] Total init tokens: in={init_input_tokens}, out={init_output_tokens}")

        # Detect developer role for coding question enforcement
        _developer_keywords = (
            "developer", "engineer", "programmer", "software", "backend", "frontend",
            "full stack", "fullstack", "sde", "swe", "coder", "architect", "devops",
            "data scientist", "ml engineer", "machine learning", "android", "ios",
        )
        role_lower = (role or "").lower()
        is_developer = (
            interview_type in ("technical", "problem")
            and any(kw in role_lower for kw in _developer_keywords)
        )
        print(f"[Session] is_developer_role={is_developer} for role={role!r} type={interview_type!r}")

        self.state.update({
            "user_id": user_id,
            "session_id": session_id,
            "context_summary": context,
            "interview_type": interview_type,
            "role": role,
            "company": company,
            "history": [],
            "performance_summary": (
                f"Interview starting. Candidate: {candidate_name or 'unknown'}. "
                f"Role: {role or 'not specified'}. Round: {interview_type}. "
                f"No answers evaluated yet."
            ),
            "skills_remaining": initial_skills,
            "skills_covered": [],
            "ended": False,
            # Accuracy fields
            "questions_asked": [],
            "current_question": "",
            "follow_up_hint": "",
            "is_developer_role": is_developer,
            "coding_questions_asked": 0,
            "turn_number": 0,
            "last_answer_type": "not_applicable",
            "consecutive_non_answers": 0,
            "input_tokens": init_input_tokens,
            "output_tokens": init_output_tokens,
        })

    def get_time_context(self) -> str:
        if self.duration_limit <= 0 or self.start_time <= 0:
            return ""
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60.0
        remaining = self.duration_limit - elapsed_min
        elapsed_str = f"{int(elapsed_min)}m {int(elapsed % 60)}s"
        
        if remaining <= 0:
            return f"TIME: {elapsed_str} elapsed. Target met. WRAP UP."
        return f"TIME: {elapsed_str} elapsed. ~{int(remaining)}m remaining."

    async def stream_response(self, user_input: str = None):
        # Update local state
        self.state["last_user_input"] = user_input
        self.state["time_context"] = self.get_time_context()
        
        # Run the Optimized Graph
        # This uses the running summary instead of full history for the LLM call
        updated_state = await self.graph_engine.run_turn(self.state)
        self.state = updated_state
        
        evaluation = self.state["current_evaluation"]
        question_text = evaluation.next_step.question
        is_coding = evaluation.next_step.is_coding_question

        if self.state["ended"]:
            if len(question_text) < 5:
                question_text = "Thank you for the conversation. We have covered the key areas. I'll pass my feedback to the team."
            is_coding = False

        # Yield metadata
        yield {"type": "metadata", "is_coding": is_coding}

        # Streaming Effect
        chunk_size = 12
        for i in range(0, len(question_text), chunk_size):
            yield {"type": "text", "content": question_text[i:i+chunk_size]}
            await asyncio.sleep(0.01)

        # Update history with the model's question
        self.state["history"].append({"role": "model", "content": question_text})

        # Real-time Reporting: Send current usage to backend after each turn
        # This allows the admin dashboard to see cost/token growth in real-time
        asyncio.create_task(self.report_usage(self.state.get("user_id") or "anonymous", self.state.get("session_id") or "demo"))

    async def report_usage(self, user_id: str, session_id: str):
        """Send token usage to the NestJS backend for analytics."""
        # GEMINI 2.0 FLASH PRICING (Estimates)
        # Input: $0.10 / 1M tokens
        # Output: $0.40 / 1M tokens
        in_cost = (self.input_tokens / 1_000_000) * 0.10
        out_cost = (self.output_tokens / 1_000_000) * 0.40
        total_cost = in_cost + out_cost

        usage_data = {
            "userId": user_id,
            "sessionId": session_id,
            "model": "gemini-2.0-flash",
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.input_tokens + self.output_tokens,
            "costUsd": total_cost,
            "subscriptionStatus": "free",  # Backend will refine from user DB
            "source": "interview",
            "interviewType": self.state.get("interview_type", ""),
            "role": self.state.get("role", ""),
            "company": self.state.get("company", ""),
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.BACKEND_URL}/analytics/ai-usage",
                    json=usage_data,
                    timeout=5.0
                )
                response.raise_for_status()
            print(f"[AI] Usage reported: in={self.input_tokens}, out={self.output_tokens}, total={self.input_tokens + self.output_tokens} tokens for user {user_id} ({self.state.get('interview_type', '')})")
        except Exception as e:
            print(f"[AI] Failed to report usage: {str(e)}")
