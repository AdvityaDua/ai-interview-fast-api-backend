import asyncio
import os
from google import genai
from google.genai import types
from typing import List, Optional
from .schemas import QuestionEvaluation, FinalEvaluation, MidInterviewSnapshot

class GeminiClient:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def summarize_context(self, resume_text: str, jd_text: str, interview_type: str = "technical", role: str = "", company: str = "") -> str:
        type_focus = {
            "technical": "Focus on: technical skills, programming languages, frameworks, system design experience, algorithms knowledge.",
            "behavioral": "Focus on: teamwork experiences, leadership examples, conflict resolution, communication skills, STAR-method scenarios.",
            "problem": "Focus on: problem-solving abilities, analytical thinking, case study scenarios, logical reasoning, creative solutions.",
            "hr": "Focus on: cultural fit, career goals, salary expectations, work-life balance views, motivation and passion.",
        }.get(interview_type, "Focus on: overall candidate assessment.")

        prompt = f"""
        You are an expert recruiter preparing for a {interview_type.upper()} interview round.
        {f'The role is: {role}' if role else ''}
        {f'The company is: {company}' if company else ''}
        
        Analyze the following Resume and Job Description (JD). 
        Create a dense, information-rich summary of under 2000 tokens.
        
        {type_focus}
        
        Output in the following sections ONLY:
        Candidate Profile
        Matched Skills
        Missing Skills
        Seniority Estimate
        Key Projects
        Interview Focus Areas (tailored for {interview_type} round)

        RESUME:
        {resume_text}

        JOB DESCRIPTION:
        {jd_text}
        """
        
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt
        )
        return response.text

    async def generate_question(self, history: List[dict], context_summary: str, interview_type: str = "technical", time_context: str = "") -> QuestionEvaluation:
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"
        
        type_rules = {
            "technical": """
    INTERVIEW TYPE: TECHNICAL ROUND
    - Ask about data structures, algorithms, system design, coding patterns.
    - Present coding problems and ask the candidate to explain their approach.
    - You CAN ask the candidate to write actual code — they have a code editor available.
      When you want them to code, say something like "Please write the code for this in the editor and submit it."
    - Dig deep into technical concepts. Don't accept surface-level answers.
    - If the candidate shares code, review it for correctness, efficiency, and style.
    - Cover topics from the JD: specific languages, frameworks, and tools.
    """,
            "behavioral": """
    INTERVIEW TYPE: BEHAVIORAL ROUND
    - Use the STAR method (Situation, Task, Action, Result) framework.
    - Ask about teamwork, leadership, conflict resolution, and handling pressure.
    - Probe for specific real-world examples, not hypotheticals.
    - Evaluate communication clarity and self-awareness.
    - Ask follow-up questions to verify depth and authenticity of examples.
    """,
            "problem": """
    INTERVIEW TYPE: PROBLEM SOLVING ROUND
    - Present real-world case studies and logical reasoning challenges.
    - Assess analytical thinking and structured problem decomposition.
    - Evaluate how the candidate breaks down ambiguous problems.
    - Test estimation skills and ability to handle incomplete information.
    - Focus on the thought process, not just the final answer.
    - You CAN ask them to implement their solution in code — they have a code editor available.
    """,
            "hr": """
    INTERVIEW TYPE: HR ROUND
    - Discuss career goals, motivation, and cultural fit.
    - Ask about work-life balance expectations and team collaboration style.
    - Evaluate communication skills, professionalism, and enthusiasm.
    - Keep the tone warm and conversational but insightful.
    - Assess alignment between the candidate's values and the company's culture.
    """
        }.get(interview_type, "")

        # Build time management section if time context exists
        time_section = ""
        if time_context:
            time_section = f"""
        TIME MANAGEMENT:
        {time_context}
        - Pace your questions to fit the interview within the target duration.
        - If remaining time < 3 minutes: Begin wrapping up. Ask a final question or make closing remarks.
        - If remaining time < 1 minute or OVER TIME: Set action to END. Provide a brief closing summary.
        - You may exceed the target by ±2 minutes if you are mid-topic — do NOT cut off abruptly.
        - Prioritize the most important skills from the JD when time is limited.
        """

        prompt = f"""
        You are an expert interviewer conducting a realtime {interview_type} interview.
        
        You must internally follow these steps:
        1. Evaluate the candidate's last answer
        2. Decide whether to continue or end
        3. Decide next question strategy
        4. Generate ONE question

        Do NOT output intermediate reasoning.
        
        CONTEXT SUMMARY:
        {context_summary}
        {type_rules}
        {time_section}
        
        YOUR GOAL:
        1.  Evaluate the candidate's last answer (if any).
        2.  Decide whether to continue or end it.
        3.  Generate the next question if continuing.
        
        RULES:
        -   Start with a greeting and an initial question if the history is empty.
        -   Be professional, encouraging, but rigorous.
        -   If the candidate struggles, offer a small hint or move to a simpler related topic.
        -   If the candidate answers well, increase difficulty.
        -   Ensure coverage of key skills from the JD.
        
        OUTPUT FORMAT:
        You must return a JSON object strictly adhering to the schema.

        {prompt_history}
        
        Evaluate the last user response (if any) and generate the next step.
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

    async def generate_feedback(self, history: List[dict], context_summary: str) -> FinalEvaluation:
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"

        prompt = f"""
        You are an expert technical interviewer. The interview has ended.
        Provide a comprehensive final evaluation of the candidate.
        
        CONTEXT SUMMARY:
        {context_summary}
        
        {prompt_history}
        
        IMPORTANT INSTRUCTIONS:
        - You MUST include EVERY SINGLE question you asked during the interview as a separate entry in question_wise_analysis.
        - Do NOT consolidate or merge questions. Each question gets its own entry.
        - Number the question_ids sequentially starting from 1.
        - The overall_score in summary should be out of 100.
        - Each individual question score should be out of 10.
        - For the IMPROVEMENT PLAN, you MUST provide:
            1. immediate_actions: Things to do in the next 24-48 hours.
            2. 1_week_plan: Focused study and practice for the next 7 days.
            3. 1_month_plan: Longer-term skill development for the next 30 days.
        - You MUST populate EVERY SINGLE FIELD in the JSON schema. Do not leave any field as null or empty unless the schema explicitly allows it (and even then, prefer providing data).
        
        Generate the final detailed evaluation report.
        """
        
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FinalEvaluation,
            },
        )
        return FinalEvaluation.model_validate_json(response.text)

    def connect_live(self, system_instruction: str):
        # Configuration aligning with user snippet where possible, while maintaining our features
        config = {
            "response_modalities": ["AUDIO", "TEXT"],
            "system_instruction": system_instruction,
            # Keeping voice config for now as it's standard for Bidi
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            }
        }
        return self.client.aio.live.connect(model=self.model_name, config=config)

    async def generate_chat_stream(self, history: List[dict], context_summary: str):
        """
        Streams regular text response based on history and context.
        Bypasses JSON structure for conversation flow to enable standard streaming.
        """
        # Construct prompt similar to live session but designed for text-only stream
        prompt_history = "INTERVIEW HISTORY:\n"
        for turn in history:
            role = turn['role']
            content = turn['content']
            prompt_history += f"{role.upper()}: {content}\n"
            
        system_instruction = f"""
        You are an expert technical interviewer.
        
        CONTEXT SUMMARY:
        {context_summary}
        
        GOAL:
        Ask questions to evaluate the candidate based on the Job Description.
        One question at a time.
        Be professional but conversational.
        If the candidate answers well, go deeper.
        If they struggle, hint or move on.
        
        Start by introducing yourself and asking the first question if history is empty.
        """
        
        # We append the history to the contents or manage it via chat session.
        # Ideally, we used ephemeral stateless calls before.
        # Let's use generaate_content_stream with the history in the prompt 
        # (or list of content objects if we wanted to be fancy, but text prompt is fine).
        
        full_prompt = f"{system_instruction}\n\n{prompt_history}\n\nCandidate just replied. Respond to the candidate."

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=full_prompt,
        ):
            if chunk.text:
                yield chunk.text
