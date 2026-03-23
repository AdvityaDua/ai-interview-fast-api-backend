import enum
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Enums for constrained choices ---

class ConfidenceLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class SignalStrength(str, enum.Enum):
    WEAK = "weak"
    STRONG = "strong" # Inferred since user only gave 'weak', assuming strong/average exists
    AVERAGE = "average"

class ClarityDepth(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Correctness(str, enum.Enum):
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    CORRECT = "correct"

class Action(str, enum.Enum):
    CONTINUE = "continue"
    END = "end"

class NextStepType(str, enum.Enum):
    FOLLOW_UP = "follow_up"
    NEW_TOPIC = "new_topic"
    NONE = "none"

class Difficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class SignalType(str, enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TrendAdjustment(str, enum.Enum):
    INCREASE = "increase"
    MAINTAIN = "maintain"
    DECREASE = "decrease"

class SeniorityAssessment(str, enum.Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"

# --- 1. Per-Question Evaluation Schema ---

class MetaInfo(BaseModel):
    turn_id: int
    interview_phase: str
    confidence_level: ConfidenceLevel

class AnswerEvaluation(BaseModel):
    score: int = Field(..., description="Score between 0 and 10")
    signal: SignalStrength
    clarity: ClarityDepth
    depth: ClarityDepth
    correctness: Correctness
    key_gaps: List[str]
    positive_signals: List[str]

class CoverageUpdate(BaseModel):
    skills_covered_now: List[str]
    skills_remaining: List[str]
    topics_covered_now: List[str]

class Decision(BaseModel):
    action: Action
    reason: str
    termination_flag: bool

class NextStep(BaseModel):
    type: NextStepType
    difficulty: Difficulty
    question: str
    target_skill: str

class QuestionEvaluation(BaseModel):
    """Schema for the model's response after every user turn."""
    meta: MetaInfo
    last_answer_evaluation: Optional[AnswerEvaluation] = None
    coverage_update: Optional[CoverageUpdate] = None
    decision: Decision
    next_step: NextStep

# --- 2. Mid-Interview Snapshot Schema ---

class InterviewHealth(BaseModel):
    overall_signal: SignalType
    risk_of_failure: RiskLevel

class Recommendation(BaseModel):
    continue_focus: List[str]
    adjust_difficulty: TrendAdjustment

class MidInterviewSnapshot(BaseModel):
    """Schema for a mid-interview health check."""
    interview_health: InterviewHealth
    current_strengths: List[str]
    current_weaknesses: List[str]
    recommendation: Recommendation

# --- 3. Final Evaluation Schema ---

class Summary(BaseModel):
    overall_score: int = Field(..., description="Score between 0 and 100")
    hire_recommendation: str
    seniority_assessment: SeniorityAssessment
    confidence_assessment: ConfidenceLevel

class DimensionScores(BaseModel):
    technical_depth: int = Field(..., description="Score between 0 and 10")
    problem_solving: int = Field(..., description="Score between 0 and 10")
    system_design: int = Field(..., description="Score between 0 and 10")
    communication: int = Field(..., description="Score between 0 and 10")
    role_fit: int = Field(..., description="Score between 0 and 10")

class EvaluationDetail(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    ideal_answer_outline: List[str]

class QuestionAnalysis(BaseModel):
    question_id: int
    question: str
    user_answer_summary: str
    score: int = Field(..., description="Score between 0 and 10")
    evaluation: EvaluationDetail

class SkillGapAnalysis(BaseModel):
    critical_gaps: List[str]
    moderate_gaps: List[str]
    minor_gaps: List[str]

class BehavioralInsights(BaseModel):
    communication_style: str
    thinking_pattern: str
    pressure_handling: str

class ImprovementPlan(BaseModel):
    immediate_actions: List[str]
    plan_1_week: List[str] = Field(..., alias="1_week_plan")
    plan_1_month: List[str] = Field(..., alias="1_month_plan")

class Verdict(BaseModel):
    strengths_to_highlight: List[str]
    areas_to_fix_before_next_interview: List[str]
    final_recommendation_text: str

class FinalEvaluation(BaseModel):
    """Schema for the final comprehensive report."""
    summary: Summary
    dimension_scores: DimensionScores
    question_wise_analysis: List[QuestionAnalysis]
    skill_gap_analysis: SkillGapAnalysis
    behavioral_insights: BehavioralInsights
    improvement_plan: ImprovementPlan
    verdict: Verdict
