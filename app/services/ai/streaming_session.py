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

    async def initialize_session(self, resume_text: str, jd_text: str, interview_type: str = "technical", role: str = "", company: str = ""):
        self.interview_type = interview_type
        self.role = role
        self.company = company
        self.context_summary = await self.client.summarize_context(resume_text, jd_text, interview_type, role, company)
        self.ended = False

    async def stream_response(self, user_input: str = None):
        """
        Generates a structured response (Evaluation + Question) and yields the question text chunks.
        """
        if user_input:
            self.history.append({"role": "user", "content": user_input})
        
        # 1. Generate Structured Question/Evaluation
        # This is a blocking call (await), so there will be a pause before streaming starts.
        evaluation = await self.client.generate_question(self.history, self.context_summary, self.interview_type)
        
        question_text = evaluation.next_step.question

        # Check for ending condition
        if evaluation.decision.action == Action.END:
            self.ended = True
            # Build a polite closing if the model didn't provide one in 'question'
            # Usually the model should provide a closing statement in 'question' if it decided to end.
            if len(question_text) < 5: 
                question_text = "Thank you for your time. This concludes the interview."
        
        # 2. Simulate Streaming (Chunking) logic
        # Since we have the full text, we can yield it in small chunks to mimic typing/stream.
        chunk_size = 10
        for i in range(0, len(question_text), chunk_size):
            chunk = question_text[i:i+chunk_size]
            yield chunk
            # Small delay to throttle the stream slightly for realism/frontend processing
            await asyncio.sleep(0.01) 
            
        self.history.append({"role": "model", "content": question_text})
