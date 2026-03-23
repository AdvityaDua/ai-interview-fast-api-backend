import time
from typing import List
from .gemini_client import GeminiClient
from .schemas import Action
import asyncio


class StreamingInterviewSession:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.context_summary = ""
        self.history: List[dict] = []
        self.ended = False
        self.interview_type = "technical"
        self.role = ""
        self.company = ""
        self.start_time: float = 0.0
        self.duration_limit: int = 0  # minutes, 0 = unlimited

    async def initialize_session(self, resume_text: str, jd_text: str, interview_type: str = "technical", role: str = "", company: str = "", duration: int = 0):
        self.interview_type = interview_type
        self.role = role
        self.company = company
        self.duration_limit = duration
        self.start_time = time.time()
        self.context_summary = await self.client.summarize_context(resume_text, jd_text, interview_type, role, company)
        self.ended = False

    def get_time_context(self) -> str:
        """Generate time context string for the Gemini prompt."""
        if self.duration_limit <= 0 or self.start_time <= 0:
            return ""

        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60.0
        total_min = self.duration_limit
        remaining = total_min - elapsed_min

        elapsed_str = f"{int(elapsed_min)}m {int(elapsed % 60)}s"

        if remaining <= 0:
            return f"TIME: Elapsed {elapsed_str} | Target was {total_min} min | OVER TIME — wrap up now."
        elif remaining <= 3:
            return f"TIME: Elapsed {elapsed_str} | Remaining ~{remaining:.0f} min | STATUS: WRAPPING UP — ask one final question or closing remark."
        else:
            return f"TIME: Elapsed {elapsed_str} | Remaining ~{remaining:.0f} min | Target: {total_min} min"

    async def stream_response(self, user_input: str = None):
        """
        Generates a structured response (Evaluation + Question) and yields the question text chunks.
        """
        if user_input:
            self.history.append({"role": "user", "content": user_input})
        
        # 1. Generate Structured Question/Evaluation with time context
        time_context = self.get_time_context()
        evaluation = await self.client.generate_question(
            self.history, self.context_summary, self.interview_type, time_context
        )
        
        question_text = evaluation.next_step.question

        # Check for ending condition
        if evaluation.decision.action == Action.END:
            self.ended = True
            if len(question_text) < 5: 
                question_text = "Thank you for your time. This concludes the interview."
        
        # 2. Simulate Streaming (Chunking) logic
        chunk_size = 10
        for i in range(0, len(question_text), chunk_size):
            chunk = question_text[i:i+chunk_size]
            yield chunk
            await asyncio.sleep(0.01) 
            
        self.history.append({"role": "model", "content": question_text})
