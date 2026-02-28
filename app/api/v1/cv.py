from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.cv_eval.evaluation_engine import evaluation_engine
from app.services.cv_eval.improvement_engine import Improvement

router = APIRouter(
    prefix="/v1/cv",
)

# ---------- Init Engines ----------
improvement_engine = Improvement()

# ---------- Request DTOs ----------
class CVScoreRequest(BaseModel):
    cv_text: str = Field(..., description="Raw resume text")

class FitIndexRequest(BaseModel):
    cv_text: str = Field(..., description="Raw resume text")
    jd_text: str = Field(..., description="Raw job description text")
    include_constraints: bool = True

class ImprovementRequest(BaseModel):
    cv_text: str = Field(..., description="Raw resume text")
    jd_text: str = Field(..., description="Raw job description text")

# ---------- Routes ----------
@router.post("/score", summary="Score CV Quality (CV only)")
def score_cv(payload: CVScoreRequest):
    """
    Returns ONLY CV Quality scores (no JD match, no fit index).
    """
    try:
        result = evaluation_engine.evaluate(cv_text=payload.cv_text, jd_text="")
        return {"cv_quality": result.get("cv_quality", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CV scoring failed: {str(e)}")


@router.post("/fit-index", summary="Score CV + JD (Fit Index)")
def score_fit_index(payload: FitIndexRequest):
    """
    Returns full evaluation (CV Quality + JD Match + Fit Index + Key Takeaways).
    """
    try:
        result = evaluation_engine.evaluate(
            cv_text=payload.cv_text,
            jd_text=payload.jd_text
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fit Index scoring failed: {str(e)}")


@router.post("/improvement", summary="Generate CV Improvements")
def improve_cv(payload: ImprovementRequest):
    """
    Generates CV improvements:
    - Tailored Resume
    - Top 1% Candidate Benchmark
    - Cover Letter (under 200 words, returned last)
    """
    try:
        result = improvement_engine.evaluate(
            cv_text=payload.cv_text,
            jd_text=payload.jd_text
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Improvement generation failed: {str(e)}")
