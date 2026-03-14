from typing import Annotated, List, TypedDict, Dict, Any, Optional
import operator
import json
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from .schemas import QuestionEvaluation, EvaluateTurnOutput, Action


class InterviewState(TypedDict):
    # Full history (accumulates all turns, used for final feedback)
    history: List[dict]

    # Compressed state summaries (primary info source for turn generation)
    performance_summary: str       # Rolling evaluation of candidate quality
    context_summary: str           # Resume + JD summary from initialization

    # Skill coverage tracking
    skills_remaining: List[str]
    skills_covered: List[str]

    # Anti-repetition: every question the AI has asked, in order
    questions_asked: List[str]

    # Current turn context
    last_user_input: Optional[str]
    current_question: str          # The last question the AI posed (for accurate eval)
    current_evaluation: Optional[QuestionEvaluation]
    follow_up_hint: str            # Set by evaluate_turn when answer needs probing

    # Session configuration
    interview_type: str
    time_context: str
    role: str
    company: str

    # Developer-role coding enforcement
    is_developer_role: bool        # Detected from role/context at init
    coding_questions_asked: int    # How many coding questions asked so far
    turn_number: int               # Turn index for pacing decisions

    # Response classification from last evaluate_turn (drives question strategy)
    last_answer_type: str          # genuine_answer | confused | refused | off_topic | incomplete | not_applicable
    consecutive_non_answers: int   # How many confused/refused turns in a row

    # Control
    ended: bool

    # Token accounting
    input_tokens: int
    output_tokens: int


# ─────────────────────────────────────────────────────────────────────────────
# Per-interview-type rules (used in _produce_question prompt)
# ─────────────────────────────────────────────────────────────────────────────
_TYPE_RULES: Dict[str, str] = {
    "technical": """
INTERVIEW TYPE: TECHNICAL
• ALWAYS base your first 2-3 questions on the candidate’s OWN tech stack listed in their resume (React, Node, Python, etc.). Do NOT ask generic language-agnostic questions when you know what tools they use.
• Cover: data structures, algorithms, system design, language-specific deep-dives, debugging/performance reasoning.
• Ask the candidate to WRITE CODE (is_coding_question=true) when exploring algorithms or implementation tasks.
• When they submit code: evaluate correctness, edge-cases, and time/space complexity.
• Progression arc: warm-up on their primary stack → core CS problem → system design / architecture → harder algorithm or optimisation.
• For SENIOR candidates (5+ years): skip entry-level questions entirely. Focus on architecture decisions, trade-offs, performance bottlenecks, and scalability.
• For JUNIOR/MID: focus on fundamentals of their stated stack, basic algorithms, debugging, and clean code.
• MINIMUM DIFFICULTY FLOOR: Never ask a question that a first-year CS student with no experience could answer trivially (e.g. “write a function that adds two numbers” is only valid if the candidate has zero experience). If the candidate has 2+ years, start at intermediate level.
""",
    "behavioral": """
INTERVIEW TYPE: BEHAVIORAL
• Reference the candidate’s ACTUAL projects and companies from their resume when asking. E.g. “In your time at [Company], tell me about a time when…”
• Use the STAR method (Situation, Task, Action, Result) to evaluate every story.
• Ask for SPECIFIC past experiences, not hypotheticals.
• Probe: teamwork, conflict resolution, leadership under pressure, handling failure, cross-functional collaboration.
• Follow up if a story lacks a concrete Result or Action — push for specifics.
• Avoid asking two questions on the same competency back to back.
""",
    "problem": """
INTERVIEW TYPE: PROBLEM SOLVING
• Ground problems in the candidate’s domain (e.g. for a frontend dev, frame problems around UI performance, state management trade-offs, rendering optimisation).
• Present real-world scenarios or case studies — not abstract puzzles unless they fit the role.
• Ask them to break down ambiguous problems step-by-step: define → assumptions → approach → solution.
• You CAN present a practical implementation challenge (is_coding_question=true), especially when the candidate’s stack is known.
• Score the thought process as much as the final answer.
""",
    "hr": """
INTERVIEW TYPE: HR / CULTURE FIT
• Tone: warm, conversational, but precise.
• Cover: motivation, career goals, work-style, values alignment, salary expectations.
• Assess: communication, professionalism, long-term fit.
• Ask open-ended questions and listen for self-awareness and clarity of purpose.
• Do NOT ask coding questions. is_coding_question must always be false.
""",
}


class InterviewGraph:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        # Lower temperature for evaluation → consistent, reliable scores
        self.eval_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.25,
        )
        # Standard temperature for question generation → natural, varied questions
        self.gen_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.65,
        )

        # Structured outputs — include_raw=True lets us read usage_metadata
        self.eval_structured_llm = self.eval_llm.with_structured_output(
            EvaluateTurnOutput, include_raw=True
        )
        self.structured_llm = self.gen_llm.with_structured_output(
            QuestionEvaluation, include_raw=True
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(InterviewState)
        builder.add_node("evaluate_turn", self._evaluate_turn)
        builder.add_node("produce_question", self._produce_question)
        builder.set_entry_point("evaluate_turn")
        builder.add_edge("evaluate_turn", "produce_question")
        builder.add_edge("produce_question", END)
        return builder.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # Node 1: Evaluate the candidate's last answer
    # ─────────────────────────────────────────────────────────────────────────
    async def _evaluate_turn(self, state: InterviewState) -> Dict[str, Any]:
        """
        Classifies the candidate's response type (genuine / confused / refused / etc.)
        and updates the performance summary and skill coverage accordingly.
        """
        if not state.get("last_user_input"):
            return {
                "performance_summary": state.get("performance_summary", "Interview is starting."),
                "skills_covered": state.get("skills_covered", []),
                "skills_remaining": state.get("skills_remaining", []),
                "follow_up_hint": "",
                "last_answer_type": "not_applicable",
                "consecutive_non_answers": 0,
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
            }

        skills_list = ", ".join(state["skills_remaining"][:12]) or "all listed skills covered"
        covered_list = ", ".join(state["skills_covered"][-10:]) or "none yet"
        last_question = state.get("current_question") or "Opening / initial question"

        prompt = f"""You are a rigorous interview evaluator for a {state['interview_type']} round.
Role being interviewed for: {state.get('role', 'Not specified')}

CURRENT PERFORMANCE SUMMARY:
{state['performance_summary']}

SKILLS STILL TO COVER: {skills_list}
SKILLS ALREADY DEMONSTRATED: {covered_list}

─── LAST EXCHANGE ───
INTERVIEWER ASKED: {last_question}
CANDIDATE RESPONDED: {state['last_user_input']}
─────────────────────

STEP 1 — Classify answer_type FIRST (this drives the entire strategy):  end_requested    — candidate explicitly asked to stop, end, quit, leave, or said they are no longer interested in the interview — CHECK THIS FIRST  genuine_answer   — candidate actually attempted the question
  confused         — candidate said they don’t understand, asked for clarification, or said the question is unclear
  refused          — candidate explicitly said they don’t want to answer, or tried to avoid the topic
  off_topic        — candidate talked about something unrelated to what was asked
  incomplete       — candidate started answering but stopped or gave only a fragment
  not_applicable   — this is a greeting or a turn with no question yet

STEP 2 — Set answer_quality based on type:
  If answer_type is end_requested: set answer_quality to "not_applicable". The interview must end.
  If answer_type is wait_requested: set answer_quality to "not_applicable". The candidate has NOT answered — do NOT score or update skill coverage.
  If answer_type is confused/refused/off_topic: set answer_quality to "not_applicable". Do NOT penalise for confusion.
  If genuine_answer: strong | adequate | weak | incorrect based on actual content.
  If incomplete: usually "weak".

STEP 3 — Update new_summary:
  Record what happened. Be specific. If confused, note the topic they were confused about.
  If wait_requested: note that candidate is still thinking; do NOT update performance.
  Do NOT claim skill coverage for confused/refused/off_topic/wait_requested answers.

STEP 4 — should_follow_up:
  true ONLY if answer_type is genuine_answer AND quality is weak/incorrect AND the topic matters.
  false for confused/refused/wait_requested (they need a different strategy, not a follow-up drill).

STEP 5 — follow_up_hint: specific gap to probe IF should_follow_up is true. Empty otherwise.
"""

        try:
            result = await self.eval_structured_llm.ainvoke([
                SystemMessage(content="You are a precise interview evaluator. Classify the response type first, then evaluate."),
                HumanMessage(content=prompt),
            ])
            data: EvaluateTurnOutput = result["parsed"]
            raw_msg = result.get("raw")

            usage_meta = getattr(raw_msg, "usage_metadata", {}) or {}
            in_tok = usage_meta.get("input_tokens", 0)
            out_tok = usage_meta.get("output_tokens", 0)
            print(
                f"[Graph] evaluate_turn: type={data.answer_type} quality={data.answer_quality} "
                f"follow_up={data.should_follow_up} tokens=in:{in_tok} out:{out_tok}"
            )

            # Only credit skill coverage for genuine answers
            if data.answer_type in ("genuine_answer", "incomplete"):
                remaining = [s for s in state["skills_remaining"] if s not in data.newly_covered_skills]
                covered = list(set(state["skills_covered"] + data.newly_covered_skills))
            else:
                remaining = state["skills_remaining"]
                covered = state["skills_covered"]

            # Track consecutive non-answers
            prev_non = state.get("consecutive_non_answers", 0)
            if data.answer_type in ("confused", "refused", "off_topic", "wait_requested"):
                consecutive_non = prev_non + 1
            else:
                consecutive_non = 0  # end_requested / genuine / incomplete all reset the streak

            return {
                "performance_summary": data.new_summary,
                "skills_covered": covered,
                "skills_remaining": remaining,
                "follow_up_hint": data.follow_up_hint if data.should_follow_up else "",
                "last_answer_type": data.answer_type,
                "consecutive_non_answers": consecutive_non,
                "input_tokens": state.get("input_tokens", 0) + in_tok,
                "output_tokens": state.get("output_tokens", 0) + out_tok,
            }

        except Exception as e:
            print(f"[Graph] evaluate_turn error (non-fatal): {e}")
            return {
                "performance_summary": state.get("performance_summary", "Evaluation error — continuing."),
                "skills_covered": state.get("skills_covered", []),
                "skills_remaining": state.get("skills_remaining", []),
                "follow_up_hint": "",
                "last_answer_type": "genuine_answer",
                "consecutive_non_answers": 0,
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Node 2: Generate the next strategic question
    # ─────────────────────────────────────────────────────────────────────────
    async def _produce_question(self, state: InterviewState) -> Dict[str, Any]:
        history = state.get("history", [])
        turn_number = state.get("turn_number", 0)
        answer_type = state.get("last_answer_type", "genuine_answer")
        consecutive_non = state.get("consecutive_non_answers", 0)

        # ── Recent conversation buffer (last 8 entries ~4 Q+A pairs) ──
        recent = history[-8:] if len(history) > 8 else history
        recent_text = (
            "\n".join(
                f"{'INTERVIEWER' if m['role'] == 'model' else 'CANDIDATE'}: {m['content']}"
                for m in recent
            )
            if recent
            else "(No exchanges yet — this is the opening turn.)"
        )

        # ── Anti-repetition ──
        questions_asked = state.get("questions_asked", [])
        anti_repeat_block = ""
        if questions_asked:
            asked_summary = "\n".join(
                f"  {i+1}. {q[:130]}" for i, q in enumerate(questions_asked[-14:])
            )
            anti_repeat_block = (
                f"\n⛔ ALREADY ASKED — DO NOT REPEAT OR REPHRASE ANY OF THESE:\n{asked_summary}"
            )

        # ── Response-type routing (core new logic) ──
        routing_block = ""
        if answer_type == "end_requested":
            routing_block = """
🛑 CANDIDATE HAS EXPLICITLY ASKED TO END THE INTERVIEW.
You MUST do ALL of the following — no exceptions:
  1. Thank them warmly for their time in one sentence.
  2. Give a single short sentence summarising how the interview went based on RUNNING PERFORMANCE ASSESSMENT.
  3. Wish them well.
  4. Set action=END in your response. This is mandatory — do NOT set action=CONTINUE.
  5. The question field should contain only your closing statement (no new question).
"""
        elif answer_type == "wait_requested":
            last_q = state.get("current_question", "")
            routing_block = f"""
⏳ CANDIDATE ASKED FOR A MOMENT / IS STILL THINKING.
Strategy — you MUST follow this exactly:
  1. Respond with exactly ONE short, warm sentence acknowledging they are thinking (e.g. "Of course, take your time!").
  2. Immediately re-state the EXACT SAME question below — word for word or a very close paraphrase. Do NOT ask anything new.
  3. Do NOT move to any other topic. Do NOT ask a follow-up or a new question.
  4. action must remain CONTINUE.
Original question to re-state: {last_q[:400]}
"""
        elif answer_type == "confused":
            last_q = state.get("current_question", "")
            routing_block = f"""
🔄 CANDIDATE WAS CONFUSED by the last question. Follow this strategy exactly:
  1. Acknowledge briefly (one sentence, e.g. "No problem, let me rephrase that.").
  2. Ask the SAME CONCEPTUAL QUESTION rephrased more concisely and concretely.
     — Use an example or analogy from the candidate's OWN stack/experience (see CANDIDATE CONTEXT).
     — If the original had multiple parts, ask only the most essential ONE part.
  3. Do NOT jump to a different topic.
  4. Do NOT drop difficulty to trivial level — confusion about phrasing ≠ lack of knowledge.
     A React developer confused about one question still knows React well.
Original question to rephrase: {last_q[:300]}
"""
        elif answer_type == "refused":
            routing_block = """
⏭ CANDIDATE DECLINED TO ANSWER. Follow this strategy exactly:
  1. Acknowledge briefly and move on without pressure (one sentence).
  2. Pivot to a DIFFERENT topic from their confirmed tech stack in CANDIDATE CONTEXT.
  3. New question must still match their seniority level — do NOT simplify to beginner basics.
  4. Do NOT keep pressing on the refused topic this turn.
"""
        elif answer_type == "off_topic":
            routing_block = """
🚧 CANDIDATE WENT OFF-TOPIC.
Strategy: redirect briefly ("Interesting point, but let's stay focused on…") then ask a sharper,
more focused question on the same area they drifted from.
"""
        elif answer_type == "incomplete":
            routing_block = """
⏳ CANDIDATE GAVE AN INCOMPLETE ANSWER.
Strategy: acknowledge what they covered, then ask them to continue or address specifically
the part they left unfinished.
"""

        # ── Follow-up routing (only for genuine weak/incorrect answers) ──
        follow_up = state.get("follow_up_hint", "")
        follow_up_block = ""
        if follow_up and answer_type == "genuine_answer":
            follow_up_block = (
                f"\n🔁 FOLLOW-UP REQUIRED: Candidate's answer was weak on a key point. "
                f"Probe this specific gap BEFORE moving to any new topic: «{follow_up}»"
            )

        # ── Coding mandate for developer roles ──
        is_developer = state.get("is_developer_role", False)
        coding_asked = state.get("coding_questions_asked", 0)
        coding_block = ""
        if is_developer and answer_type not in ("end_requested",):
            if coding_asked == 0 and turn_number >= 1:
                coding_block = (
                    "\n💻 CODING MANDATE: This is a developer candidate and NO coding question has been asked yet. "
                    "You MUST ask a hands-on coding problem THIS TURN regardless of interview type — "
                    "set is_coding_question=true. "
                    "Pick a problem directly relevant to their primary stack (see CANDIDATE CONTEXT). "
                    "This is not optional — asking only verbal/conceptual questions to a developer is a failure."
                )
            elif coding_asked == 1 and turn_number >= 4 and answer_type not in ("confused", "refused"):
                coding_block = (
                    "\n💻 CODING RECOMMENDATION: Only one coding problem asked so far for a developer candidate. "
                    "If conversationally appropriate this turn, present a second coding problem (is_coding_question=true)."
                )

        # ── Difficulty calibration ──
        perf = state["performance_summary"].lower()
        strong_ct = perf.count("strong")
        weak_ct = perf.count("weak") + perf.count("incorrect")
        if strong_ct > weak_ct + 1:
            difficulty_hint = "Candidate is performing WELL → INCREASE difficulty this turn."
        elif weak_ct > strong_ct + 1:
            difficulty_hint = (
                "Candidate is struggling → break into a simpler sub-question. "
                "Do NOT drop below intermediate level for experienced candidates."
            )
        else:
            difficulty_hint = "Candidate is performing adequately → MAINTAIN current difficulty."

        # ── Seniority floor from context summary ──
        context = state["context_summary"]
        seniority_note = next(
            (ln.strip() for ln in context.splitlines()
             if "seniority" in ln.lower() or "year" in ln.lower()),
            ""
        )
        floor_block = ""
        if seniority_note:
            floor_block = (
                f"\n⚠ SENIORITY FLOOR: {seniority_note}. "
                f"Questions must match this level. NEVER ask trivial entry-level questions "
                f"(e.g. 'what is a variable', 'write a sum function') for a candidate with real experience."
            )

        # ── Type-specific rules ──
        type_rules = _TYPE_RULES.get(state["interview_type"], _TYPE_RULES["technical"])

        prompt = f"""You are an expert interviewer conducting a {state['interview_type'].upper()} interview.

{'═' * 60}
CANDIDATE CONTEXT (resume + JD summary):
{context}
Role: {state.get('role', 'Not specified')} | Company: {state.get('company', 'Not specified')}
{'═' * 60}

INTERVIEW PROGRESS — Turn {turn_number + 1}:
• Skills covered  : {', '.join(state['skills_covered']) if state['skills_covered'] else 'None yet'}
• Skills remaining: {', '.join(state['skills_remaining'][:10]) if state['skills_remaining'] else 'All covered — deepen existing topics'}
• Difficulty      : {difficulty_hint}

RUNNING PERFORMANCE ASSESSMENT:
{state['performance_summary']}

TIME: {state['time_context'] or 'No time limit.'}
{'═' * 60}

RECENT CONVERSATION:
{recent_text}
{'═' * 60}
{routing_block}
{follow_up_block}
{anti_repeat_block}
{coding_block}
{floor_block}

INTERVIEW TYPE RULES:
{type_rules}
{'═' * 60}

INSTRUCTIONS:
1. Opening turn (turn 0): greet candidate BY NAME from CANDIDATE CONTEXT; ask a warm-up question about their PRIMARY TECHNOLOGY listed in the resume.
2. All other turns: apply the routing strategy above FIRST, then pick the best next question.
3. ALWAYS anchor questions to the candidate's stated experience and stack from CANDIDATE CONTEXT. Never ask generic language-agnostic questions when their tech stack is known.
4. question = the exact words you say to the candidate. Natural, conversational, one focused question.
5. is_coding_question=true ONLY when asking them to write/implement actual code in the editor.
6. action=END only when: candidate explicitly requested to end (end_requested routing block above), time is truly up, or all key topics have been thoroughly explored.
7. If the 🛑 END routing block is present above, action MUST be END — this overrides everything else.
"""

        result = await self.structured_llm.ainvoke(prompt)
        evaluation: QuestionEvaluation = result["parsed"]
        raw_msg = result.get("raw")

        usage_meta = getattr(raw_msg, "usage_metadata", {}) or {}
        in_tok = usage_meta.get("input_tokens", 0)
        out_tok = usage_meta.get("output_tokens", 0)
        total = state.get("input_tokens", 0) + in_tok + state.get("output_tokens", 0) + out_tok
        print(
            f"[Graph] produce_question: coding={evaluation.next_step.is_coding_question} "
            f"diff={evaluation.next_step.difficulty} ans_type={answer_type} consecutive_non={consecutive_non} "
            f"tokens=in:{in_tok} out:{out_tok} cumulative:{total}"
        )

        new_history = list(history)
        if state.get("last_user_input"):
            new_history.append({"role": "user", "content": state["last_user_input"]})

        new_questions_asked = list(questions_asked) + [evaluation.next_step.question]
        new_coding_count = (
            state.get("coding_questions_asked", 0)
            + (1 if evaluation.next_step.is_coding_question else 0)
        )

        return {
            "current_evaluation": evaluation,
            "history": new_history,
            "ended": evaluation.decision.action == Action.END,
            "current_question": evaluation.next_step.question,
            "questions_asked": new_questions_asked,
            "coding_questions_asked": new_coding_count,
            "turn_number": turn_number + 1,
            "follow_up_hint": "",
            "input_tokens": state.get("input_tokens", 0) + in_tok,
            "output_tokens": state.get("output_tokens", 0) + out_tok,
        }

    async def run_turn(self, state: InterviewState) -> InterviewState:
        """Run a single interview turn through the graph."""
        return await self.graph.ainvoke(state)
