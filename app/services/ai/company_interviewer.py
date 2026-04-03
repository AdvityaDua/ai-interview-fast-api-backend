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
        You are an elite senior principal interviewer conducting a rigorous {interview_type} interview.

        CONTEXT SUMMARY & KNOWLEDGE BASE:
        {context_summary}

        ══════════════════════════════════════════════════════════
        PRIMARY QUESTION MANDATE — READ CAREFULLY:
        ══════════════════════════════════════════════════════════
        The KNOWLEDGE CHUNKS section in the context contains an interview transcript or question
        bank uploaded by the admin.  Your questions MUST be drawn predominantly from that material.

        RULES:
        1. KNOWLEDGE-FIRST: At least 70–80% of your questions must come directly from the topics,
           questions, or scenarios present in the KNOWLEDGE CHUNKS.  You may lightly rephrase or
           enhance wording for clarity, but the substance belongs to the transcript.
        2. ENHANCEMENT ALLOWED: You may add up to 1–2 probing follow-ups per topic (e.g., "Why?",
           "What trade-offs did you consider?", "Can you show the code?") to deepen the assessment.
           These do NOT count against the knowledge-first budget.
        3. CROSS-QUESTION WITH LLM: If the candidate's answer reveals a gap or an interesting
           tangent that is DIRECTLY RELATED to a topic already covered in the knowledge chunks,
           you are encouraged to cross-question using your own reasoning to probe depth.
        4. NO FREE-FORM INVENTION: Do NOT invent new topics outside the knowledge base unless
           the bank is fully exhausted AND a critical JD requirement has not been covered.
        5. SEQUENCING:
           - Turn 1: Professional greeting and one bridging question to set context.
           - Turns 2+: Work through the knowledge-base question bank systematically.
        6. Ask ONE question at a time.
        7. JD constraint (if provided) defines seniority bar and scope — respect it.
        8. Resume is a calibration signal only; do NOT make it the primary topic source.
        ══════════════════════════════════════════════════════════

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
                            f"KNOWLEDGE-BASE INTERVIEW MODE — {company.upper()}\n"
                            f"{'=' * 60}\n"
                            f"An admin has uploaded an interview knowledge base for this topic.\n"
                            f"The KNOWLEDGE CHUNKS below are your PRIMARY question source.\n"
                            f"Draw 70–80%+ of your questions directly from that material.\n"
                            f"You may enhance questions slightly and cross-question with LLM reasoning,\n"
                            f"but do NOT invent topics outside the knowledge base unless it is fully exhausted.\n"
                            f"JD (if provided) constrains scope and seniority bar.\n"
                            f"Resume is a secondary calibration signal only.\n"
                        )

                        if grounding:
                            print(f"[CompanySession] ✅ Found grounding for '{company}' ({len(grounding)} chars)")
                            return (
                                intro_block
                                + f"\n--- {company.upper()} KNOWLEDGE CHUNKS (PRIMARY QUESTION SOURCE) ---\n"
                                + grounding
                                + f"\n{'=' * 60}\n"
                            )

                        print(f"[CompanySession] ⚠️ No RAG grounding found for '{company}', falling back to JD-guided mode")
                        if jd_context:
                            return (
                                intro_block
                                + f"\n--- KNOWLEDGE CHUNKS ---\n"
                                + "No knowledge-base chunks were found for this topic. Use the JD summary to focus the interview.\n"
                                + f"{'=' * 60}\n"
                            )
        except Exception as e:
            print(f"[CompanySession] ⚠️  RAG lookup error for '{company}': {e}")
        return (
            f"\n\n{'=' * 60}\n"
            f"KNOWLEDGE-BASE INTERVIEW MODE — {company.upper()}\n"
            f"{'=' * 60}\n"
            f"No knowledge-base chunks found. Use the JD summary as the primary constraint.\n"
            f"Resume content is secondary and should only calibrate seniority.\n"
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
