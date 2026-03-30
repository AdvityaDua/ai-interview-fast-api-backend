import time
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from .gemini_client import GeminiClient
from .streaming_session import StreamingInterviewSession
from .interview_graph import InterviewGraph, InterviewState, _max_questions_for_duration, _max_questions_per_topic, _max_confusion_retries_for_duration
from .schemas import QuestionEvaluation

class CompanyGeminiClient(GeminiClient):
    """Specialized Gemini Client for Company-specific interviews (FAANG, etc.)"""
    
    async def generate_question(self, history: List[dict], context_summary: str, interview_type: str = "technical", time_context: str = "") -> QuestionEvaluation:
        # Use a more rigid, expert persona for company-specific rounds
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"

        time_section = f"TIME MANAGEMENT:\n{time_context}\n" if time_context else ""

        prompt = f"""
        You are an elite senior principal interviewer conducting a rigorous {interview_type} interview for a top-tier tech company.
        
        CONTEXT SUMMARY & COMPANY KNOWLEDGE:
        {context_summary}
        
        RULES:
        -   MANDATORY SEQUENCING:
            1. First 1-2 turns: Professional greeting and introductory questions about their motivation for joining this specific company.
            2. Remaining Session: Transition into the REAL-WORLD TECHNICAL QUESTIONS provided in the RAG context below. 
        -   RAG PRIORITY: You MUST use the actual questions and case studies found in the 'QUESTION BANK & INTEL (RAG)' section. Do not ask generic technical questions.
        -   IGNORE RESUME: In this specialized round, do NOT ask about the candidate's projects or past history from their resume. Your goal is purely a standardized knowledge assessment based on the company's hiring bar in the RAG context.
        -   Be extremely rigorous. Analyze the candidate's technical logic and problem-solving depth.
        -   Ask ONE question at a time.
        
        {time_section}
        
        {prompt_history}
        
        Evaluate the latest response and generate the next JSON QuestionEvaluation.
        """

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QuestionEvaluation,
            },
        )
        return QuestionEvaluation.model_validate_json(response.text)

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

    async def _fetch_company_knowledge(self, company: str) -> str:
        """Find a KnowledgeTopic by name and fetch its vector grounding."""
        if not company or len(company) < 2: return ""
        try:
            target_url = f"{settings.BACKEND_URL}/knowledge/topics/find-by-name?name={company}&all=true"
            print(f"[CompanySession] Checking for company knowledge: {target_url}")
            async with httpx.AsyncClient() as client:
                res = await client.get(target_url, timeout=5.0)
                if res.status_code == 200:
                    topic = res.json()
                    if topic and "_id" in topic:
                        topic_id = topic["_id"]
                        from .rag_service import rag_service
                        grounding = await rag_service.get_topic_grounding(topic_id)
                        if grounding:
                            print(f"[CompanySession] ✅ Found grounding for '{company}' ({len(grounding)} chars)")
                            return (
                                f"\n\n{'=' * 60}\n"
                                f"🚨 CRITICAL SYSTEM OVERRIDE: SPECIALIZED {company.upper()} INTERVIEW 🚨\n"
                                f"{'=' * 60}\n"
                                f"You are now acting as an Elite Senior Principal Interviewer specifically for **{company}**.\n"
                                f"YOUR MANDATE:\n"
                                f"1. In your opening turn, you MUST greet the candidate and state clearly that you are evaluating them for the '{company}' specific round.\n"
                                f"2. You MUST prioritize the '{company} QUESTION BANK' provided below over generic questions.\n"
                                f"3. Elevate your rigor and standards to match top-tier {company} hiring bars.\n"
                                f"4. If specific {company} questions or case studies are provided below, you MUST ask them, adapting them naturally to the conversation.\n\n"
                                f"--- {company.upper()} QUESTION BANK & INTEL (RAG) ---\n"
                                f"{grounding}\n"
                                f"{'=' * 60}\n"
                            )
        except Exception as e:
            print(f"[CompanySession] ⚠️  RAG lookup error for '{company}': {e}")
        return ""

    async def initialize_session(self, *args, **kwargs):
        # First run the normal init to get context summary and resume analysis
        await super().initialize_session(*args, **kwargs)
        
        # Now enhance it with company knowledge
        company = kwargs.get("company", args[6] if len(args) > 6 else "")
        rag_context = await self._fetch_company_knowledge(company)
        
        if rag_context:
            self.state["context_summary"] += rag_context
            self.state["has_jd"] = True
            
            # Estimate tokens: 4 characters per token
            extra_tokens = len(rag_context) // 4
            self.rag_tokens += extra_tokens
            # Also count these as initial input tokens
            self.input_tokens += extra_tokens
            
            print(f"[CompanySession] Injected specialized grounding for {company} (~{extra_tokens} RAG tokens)")
