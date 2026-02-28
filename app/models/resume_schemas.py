from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

# Resume Analytics Inputs
class CVQualitySubscore(BaseSchema):
    dimension: str
    score: int
    max_score: int
    evidence: List[str]

class CVQuality(BaseSchema):
    overall_score: int
    subscores: List[CVQualitySubscore]

class JDMatchSubscore(BaseSchema):
    dimension: str
    score: int
    max_score: int
    evidence: List[str]

class JDMatch(BaseSchema):
    overall_score: int
    subscores: List[JDMatchSubscore]

class KeyTakeaways(BaseSchema):
    red_flags: List[str]
    green_flags: List[str]

class Analytics(BaseSchema):
    cv_quality: CVQuality
    jd_match: Optional[JDMatch] = None
    key_takeaways: KeyTakeaways
    overall_score: int

class TailoredResume(BaseSchema):
    summary: str
    experience: List[str]
    skills: List[str]
    projects: List[str]

class Top1PercentGap(BaseSchema):
    strengths: List[str]
    gaps: List[str]
    actionable_next_steps: List[str]

class Enhancement(BaseSchema):
    tailored_resume: TailoredResume
    top_1_percent_gap: Top1PercentGap
    cover_letter: str

class ResumeData(BaseSchema):
    id: str
    filename: str
    url: str
    analytics: Analytics
    enhancement: Enhancement

class ResumeAnalysisRequest(BaseSchema):
    message: Optional[str] = None
    resume: Optional[ResumeData] = None
    id: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    analytics: Optional[Analytics] = None
    enhancement: Optional[Enhancement] = None

# Resume Built Output Schemas
class ResumeBuilderContent(BaseSchema):
    personal_info: Dict[str, str]
    professional_summary: str
    skills: Dict[str, List[str]]
    experience: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    achievements: List[str]
    certifications: List[Dict[str, str]]
    languages: List[Dict[str, str]]

class ResumeBuilderResponse(BaseSchema):
    status: str
    resume_content: ResumeBuilderContent
    formatting_tips: List[str]
    message: str
