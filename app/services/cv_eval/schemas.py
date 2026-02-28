from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class CVEvaluationRequest(BaseModel):
    cv_text: str = Field(..., description="CV text content")
    jd_text: str = Field(..., description="Job description text content")

class CVEvaluationResult(BaseModel):
    cv_quality: Dict[str, Any] = Field(..., description="CV quality scores")
    jd_match: Optional[Dict[str, Any]] = Field(None, description="JD match scores")
    fit_index: Optional[Dict[str, Any]] = Field(None, description="Overall fit index")