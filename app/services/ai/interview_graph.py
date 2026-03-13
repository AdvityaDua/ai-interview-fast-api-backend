from typing import Annotated, List, TypedDict, Dict, Any, Optional
import operator
import json
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from .schemas import QuestionEvaluation, Action

class InterviewState(TypedDict):
    # Full history (kept but not sent to turn-by-turn LLM)
    history: List[dict] 
    
    # State-based summaries (The MAIN source of info for turn generation)
    performance_summary: str   
    context_summary: str       # Full summary from start
    skills_remaining: List[str]
    skills_covered: List[str]
    
    # Current turn info
    last_user_input: Optional[str]
    current_evaluation: Optional[QuestionEvaluation]
    
    # Configuration
    interview_type: str
    time_context: str
    role: str
    company: str
    
    # Control flags
    ended: bool

    # Token Intelligence Tracking
    input_tokens: int
    output_tokens: int

class InterviewGraph:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )
        # Use structured LLM for turn logic — include_raw=True so we keep the AIMessage
        # and can read usage_metadata for accurate token counting
        self.structured_llm = self.llm.with_structured_output(QuestionEvaluation, include_raw=True)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(InterviewState)
        builder.add_node("evaluate_turn", self._evaluate_turn)
        builder.add_node("produce_question", self._produce_question)
        builder.set_entry_point("evaluate_turn")
        builder.add_edge("evaluate_turn", "produce_question")
        builder.add_edge("produce_question", END)
        return builder.compile()

    async def _evaluate_turn(self, state: InterviewState) -> Dict[str, Any]:
        """Node 1: Updates performance summary and tracks skill coverage."""
        if not state["last_user_input"]:
            # Initial state setup
            return {
                "performance_summary": state.get("performance_summary", "Starting the interview."),
                "skills_covered": [],
                "skills_remaining": state.get("skills_remaining", [])
            }

        # Optimized: Single small prompt to update state
        prompt = f"""
        Analyze the last turn of the interview.
        
        PREVIOUS SUMMARY: {state['performance_summary']}
        LAST USER INPUT: {state['last_user_input']}
        REMAINING SKILLS: {", ".join(state['skills_remaining'])}
        
        Update the state. Output a JSON with these keys:
        - "new_summary": updated performance summary (max 300 words)
        - "newly_covered": list of skills demonstrated in this turn
        """
        
        try:
            # We use the raw LLM with a simple prompt to save tokens (no full JSON schema overhead here)
            response = await self.llm.ainvoke([SystemMessage(content="You are an interview auditor."), HumanMessage(content=prompt)])
            # Basic parsing of semi-structured text if needed, but for reliability we can use the model's output
            # For simplicity in this demo, let's assume valid JSON or just update fields based on content
            content = response.content
            # In production, use structured output for this too, but for 'optimization' we keep prompts lean.
            # Let's extract what we can.
            
            # Simple heuristic since LLM is instructed to output JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                remaining = [s for s in state['skills_remaining'] if s not in data.get("newly_covered", [])]
                covered = state['skills_covered'] + data.get("newly_covered", [])
                
                # LangChain Google AI stores usage in response.usage_metadata as a dict
                # with keys: input_tokens, output_tokens, total_tokens
                usage_meta = getattr(response, 'usage_metadata', {}) or {}
                input_tokens  = usage_meta.get('input_tokens', 0)
                output_tokens = usage_meta.get('output_tokens', 0)
                
                print(f"[Graph] _update_summary tokens: in={input_tokens}, out={output_tokens}")
                
                return {
                    "performance_summary": data.get("new_summary", state['performance_summary']),
                    "skills_covered": list(set(covered)),
                    "skills_remaining": remaining,
                    "input_tokens": state.get("input_tokens", 0) + input_tokens,
                    "output_tokens": state.get("output_tokens", 0) + output_tokens
                }
        except Exception as e:
            print(f"[Graph] _update_summary error: {e}")
            pass
            
        return {
            "input_tokens": state.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0)
        }

    async def _produce_question(self, state: InterviewState) -> Dict[str, Any]:
        """Node 2: Generates question using the state + key candidate context."""
        
        # We bring back the context_summary to ensure the AI knows WHO it is talking to.
        # However, we keep it below the performance summary to prioritize current flow.
        prompt = f"""
        You are an expert interviewer conducting a {state['interview_type']} round.
        
        CANDIDATE PERSONA & JD CONTEXT:
        {state['context_summary']}
        
        PROGRESS TRACKING:
        - Covered: {", ".join(state['skills_covered']) if state['skills_covered'] else "None yet"}
        - To Cover: {", ".join(state['skills_remaining']) if state['skills_remaining'] else "Deep dive"}
        
        INTERVIEW PERFORMANCE SO FAR:
        {state['performance_summary']}
        
        TIME STATUS: {state['time_context']}
        
        LAST INTERACTION:
        {f"CANDIDATE: {state['last_user_input']}" if state['last_user_input'] else "Just starting."}
        
        GOAL:
        Evaluate the latest input (if any) and ask the NEXT strategic question.
        1. GREETING: If this is the start (no history), GREET the candidate using the "IDENTIFIED NAME" from the PERSONA summary.
        2. PERSONALIZATION: Tailor the question based on their specific experience in the CANDIDATE PERSONA.
        3. STRATEGY: If they are a senior, ask about architecture/trade-offs. If junior, focus on implementation.
        """
        
        # include_raw=True → returns {"raw": AIMessage, "parsed": QuestionEvaluation, "parsing_error": ...}
        result     = await self.structured_llm.ainvoke(prompt)
        evaluation = result["parsed"]
        raw_msg    = result.get("raw")
        
        # LangChain Google AI stores usage in AIMessage.usage_metadata
        usage_meta    = getattr(raw_msg, 'usage_metadata', {}) or {}
        input_tokens  = usage_meta.get('input_tokens', 0)
        output_tokens = usage_meta.get('output_tokens', 0)
        
        print(f"[Graph] _produce_question tokens: in={input_tokens}, out={output_tokens}, total_so_far={state.get('input_tokens', 0) + input_tokens + state.get('output_tokens', 0) + output_tokens}")
        
        new_history = state["history"]
        if state["last_user_input"]:
            new_history.append({"role": "user", "content": state["last_user_input"]})
        
        return {
            "current_evaluation": evaluation,
            "history": new_history,
            "ended": evaluation.decision.action == Action.END,
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens
        }

    async def run_turn(self, state: InterviewState) -> InterviewState:
        """Helper to run a single turn of the graph."""
        return await self.graph.ainvoke(state)
