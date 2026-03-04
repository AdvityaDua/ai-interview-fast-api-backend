from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra='allow')

class GeminiBaseSchema(BaseModel):
    """Schema for Gemini output must not have extra properties."""
    model_config = ConfigDict(from_attributes=True)

# Resume Analytics Inputs
class CVQualitySubscore(BaseSchema):
    dimension: Optional[str] = None
    score: Optional[int] = 0
    max_score: Optional[int] = 100
    evidence: Optional[List[str]] = None

class CVQuality(BaseSchema):
    overall_score: Optional[int] = 0
    subscores: Optional[List[CVQualitySubscore]] = None

class JDMatchSubscore(BaseSchema):
    dimension: Optional[str] = None
    score: Optional[int] = 0
    max_score: Optional[int] = 100
    evidence: Optional[List[str]] = None

class JDMatch(BaseSchema):
    overall_score: Optional[int] = 0
    subscores: Optional[List[JDMatchSubscore]] = None

class KeyTakeaways(BaseSchema):
    red_flags: Optional[List[str]] = None
    green_flags: Optional[List[str]] = None

class Analytics(BaseSchema):
    cv_quality: Optional[CVQuality] = None
    jd_match: Optional[JDMatch] = None
    key_takeaways: Optional[KeyTakeaways] = None
    overall_score: Optional[int] = 0
    sections: Optional[List[Any]] = None

class TailoredResume(BaseSchema):
    summary: Optional[str] = None
    experience: Optional[List[Any]] = None
    skills: Optional[List[Any]] = None
    projects: Optional[List[Any]] = None

class Top1PercentGap(BaseSchema):
    strengths: Optional[List[str]] = None
    gaps: Optional[List[str]] = None
    actionable_next_steps: Optional[List[str]] = None

class Enhancement(BaseSchema):
    tailored_resume: Optional[TailoredResume] = None
    top_1_percent_gap: Optional[Top1PercentGap] = None
    cover_letter: Optional[Any] = None

class ResumeData(BaseSchema):
    id: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    analytics: Optional[Analytics] = None
    enhancement: Optional[Enhancement] = None

class ResumeAnalysisRequest(BaseSchema):
    message: Optional[str] = None
    resume: Optional[ResumeData] = None
    id: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    analytics: Optional[Analytics] = None
    enhancement: Optional[Enhancement] = None

# Resume Built Output Sub-Schemas (Explicit for Gemini)
class ResumePersonalInfo(GeminiBaseSchema):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

class ResumeExperience(GeminiBaseSchema):
    company: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[List[str]] = Field(default_factory=list)

class ResumeProject(GeminiBaseSchema):
    name: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = Field(default_factory=list)
    highlights: Optional[List[str]] = Field(default_factory=list)
    github: Optional[str] = None
    demo: Optional[str] = None

class ResumeEducation(GeminiBaseSchema):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    gpa: Optional[str] = None

class ResumeCertification(GeminiBaseSchema):
    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None

class ResumeLanguage(GeminiBaseSchema):
    name: Optional[str] = None
    proficiency: Optional[str] = None

class ResumeSkills(GeminiBaseSchema):
    frontend: Optional[List[str]] = Field(default_factory=list)
    backend: Optional[List[str]] = Field(default_factory=list)
    tools_cloud: Optional[List[str]] = Field(default_factory=list)

class ResumeBuilderContent(GeminiBaseSchema):
    personal_info: Optional[ResumePersonalInfo] = None
    professional_summary: Optional[str] = None
    skills: Optional[ResumeSkills] = None
    experience: Optional[List[ResumeExperience]] = Field(default_factory=list)
    projects: Optional[List[ResumeProject]] = Field(default_factory=list)
    education: Optional[List[ResumeEducation]] = Field(default_factory=list)
    achievements: Optional[List[str]] = Field(default_factory=list)
    certifications: Optional[List[ResumeCertification]] = Field(default_factory=list)
    languages: Optional[List[ResumeLanguage]] = Field(default_factory=list)

class ResumeBuilderResponse(GeminiBaseSchema):
    status: Optional[str] = "success"
    resume_content: Optional[ResumeBuilderContent] = None
    formatting_tips: Optional[List[str]] = None
    message: Optional[str] = ""
