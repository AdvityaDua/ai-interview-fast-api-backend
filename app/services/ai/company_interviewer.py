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
            1. First 1-2 turns: Professional greeting and an opening bridge into the company-specific JD/RAG topic bank.
            2. Remaining Session: Transition into the REAL-WORLD TECHNICAL QUESTIONS provided in the RAG context below.
        -   RAG PRIORITY: You MUST use the actual questions and case studies found in the 'QUESTION BANK & INTEL (RAG)' section. Do not ask generic technical questions.
        -   JD ALWAYS MATTERS: The JD summary in the context is mandatory and should constrain the depth, scope, and difficulty of every question.
        -   RESUME IS SECONDARY: Do not make the candidate's resume the primary source of topics in this company round. Use it only as a light calibration signal when the RAG bank is exhausted or when you need to confirm seniority.
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

    async def _fetch_company_knowledge(self, company: str, jd_text: str = "") -> str:
        """Find a KnowledgeTopic by name and fetch its vector grounding."""
        if not company or len(company) < 2: return ""
        try:
            jd_context = jd_text.strip()
            target_url = f"{settings.BACKEND_URL}/knowledge/topics/find-by-name?name={company}&all=true"
            print(f"[CompanySession] Checking for company knowledge: {target_url}")
            async with httpx.AsyncClient() as client:
                res = await client.get(target_url, timeout=5.0)
                if res.status_code == 200:
                    topic = res.json()
                    if topic and "_id" in topic:
                        topic_id = topic["_id"]
                        from .rag_service import rag_service
                        grounding = await rag_service.get_topic_grounding(topic_id, company_name=company)

                        intro_block = (
                            f"\n\n{'=' * 60}\n"
                            f"🚨 CRITICAL SYSTEM OVERRIDE: SPECIALIZED {company.upper()} INTERVIEW 🚨\n"
                            f"{'=' * 60}\n"
                            f"You are now acting as an Elite Senior Principal Interviewer specifically for **{company}**.\n"
                            f"YOUR MANDATE:\n"
                            f"1. Use the JD summary already present in the context as the hard constraint for role scope and difficulty.\n"
                            f"2. Use the RAG section below as the primary topic bank for what this company usually asks.\n"
                            f"3. Ask around the topics surfaced by RAG, not generic interview filler.\n"
                            f"4. Treat the candidate's resume as secondary calibration only if the RAG bank is exhausted or ambiguous.\n"
                        )

                        if grounding:
                            print(f"[CompanySession] ✅ Found grounding for '{company}' ({len(grounding)} chars)")
                            return (
                                intro_block
                                + f"\n--- {company.upper()} QUESTION BANK & INTEL (RAG) ---\n"
                                + grounding
                                + f"\n{'=' * 60}\n"
                            )

                        print(f"[CompanySession] ⚠️ No RAG grounding found for '{company}', falling back to JD-guided company mode")
                        if jd_context:
                            return (
                                intro_block
                                + f"\n--- {company.upper()} QUESTION BANK & INTEL (RAG) ---\n"
                                + "No RAG chunks were retrieved for this company. Use the JD summary above to keep the interview focused.\n"
                                + f"{'=' * 60}\n"
                            )
        except Exception as e:
            print(f"[CompanySession] ⚠️  RAG lookup error for '{company}': {e}")
        return (
            f"\n\n{'=' * 60}\n"
            f"🚨 SPECIALIZED {company.upper()} INTERVIEW MODE 🚨\n"
            f"{'=' * 60}\n"
            f"Use the JD summary already present in the context as the primary constraint.\n"
            f"If RAG content is available, prioritize it for topic selection and question depth.\n"
            f"Resume content is secondary and should only be used to calibrate seniority or clarify ambiguous topic choice.\n"
            f"{'=' * 60}\n"
        )

    async def initialize_session(self, *args, **kwargs):
        # First run the normal init to get context summary and resume analysis
        await super().initialize_session(*args, **kwargs)
        
        # Now enhance it with company knowledge
        company = kwargs.get("company", args[6] if len(args) > 6 else "")
        jd_text = kwargs.get("jd_text", args[3] if len(args) > 3 else "")

        if company and not (jd_text or "").strip():
            raise ValueError("Company-specific interviews require a JD so the RAG bank can be grounded correctly.")

        rag_context = await self._fetch_company_knowledge(company, jd_text=jd_text)
        
        if rag_context:
            self.state["context_summary"] += rag_context
            self.state["has_jd"] = True
            
            # Estimate tokens: 4 characters per token
            extra_tokens = len(rag_context) // 4
            self.rag_tokens += extra_tokens
            # Also count these as initial input tokens
            self.input_tokens += extra_tokens
            
            print(f"[CompanySession] Injected specialized grounding for {company} (~{extra_tokens} RAG tokens)")
