# import re
# from typing import List, Dict, Any, Tuple
# from .schemas import ScoreResult, SubScore


# CV_CRITERIA = {
#     "ats_structure": {"max_score": 10, "keywords": ["contact", "email", "phone", "linkedin"]},
#     "writing_clarity": {"max_score": 15, "keywords": ["clear", "concise", "professional"]},
#     "quantified_impact": {"max_score": 20, "keywords": ["%", "increased", "decreased", "reduced", "improved"]},
#     "technical_depth": {"max_score": 15, "keywords": ["architecture", "design", "optimization", "scalable"]},
#     "projects_portfolio": {"max_score": 10, "keywords": ["project", "portfolio", "github", "demo"]},
#     "leadership_skills": {"max_score": 10, "keywords": ["led", "mentored", "managed", "team"]},
#     "career_progression": {"max_score": 10, "keywords": ["senior", "lead", "principal", "promoted"]},
#     "consistency": {"max_score": 10, "keywords": ["consistent", "formatted", "dates"]},
# }

# JD_CRITERIA = {
#     "hard_skills": {"max_score": 35, "keywords": ["python", "java", "aws", "docker", "kubernetes"]},
#     "responsibilities": {"max_score": 15, "keywords": ["design", "implement", "develop", "build"]},
#     "domain_relevance": {"max_score": 10, "keywords": ["backend", "frontend", "full-stack", "web"]},
#     "seniority": {"max_score": 10, "keywords": ["senior", "lead", "principal", "years"]},
#     "nice_to_haves": {"max_score": 5, "keywords": ["open source", "agile", "scrum", "monitoring"]},
#     "education_certs": {"max_score": 5, "keywords": ["bachelor", "master", "certified", "degree"]},
#     "recent_achievements": {"max_score": 10, "keywords": ["recent", "achievement", "impact", "result"]},
#     "constraints": {"max_score": 10, "keywords": ["location", "remote", "travel", "authorization"]},
# }


# def score_cv_quality(cv_text: str) -> ScoreResult:
#     subscores = []
#     total_score = 0
#     for dim, criteria in CV_CRITERIA.items():
#         score, evidence = _score_dimension(cv_text, dim, criteria)
#         subscores.append(SubScore(dimension=dim, score=score, evidence=evidence, max_score=criteria["max_score"]))
#         total_score += score
#     return ScoreResult(overall_score=total_score, band=_band(total_score), subscores=subscores)


# def score_jd_match(cv_text: str, jd_text: str) -> ScoreResult:
#     subscores = []
#     total_score = 0
#     for dim, criteria in JD_CRITERIA.items():
#         if dim == "constraints":
#             score, evidence = _score_constraints(cv_text, jd_text, criteria)
#         else:
#             score, evidence = _score_jd_dimension(cv_text, jd_text, dim, criteria)
#         subscores.append(SubScore(dimension=dim, score=score, evidence=evidence, max_score=criteria["max_score"]))
#         total_score += score
#     return ScoreResult(overall_score=total_score, band=_band(total_score), subscores=subscores)


# # ----------------- internal helpers -----------------

# def _score_dimension(cv_text: str, dimension: str, criteria: Dict[str, Any]) -> Tuple[float, List[str]]:
#     score, evidence = 0.0, []
#     text_lower = cv_text.lower()
#     for kw in criteria["keywords"]:
#         if kw in text_lower:
#             evidence.append(_extract_context(cv_text, kw))
#             score += criteria["max_score"] / len(criteria["keywords"])
#     if dimension == "quantified_impact":
#         score += _score_quantified_metrics(cv_text)
#     elif dimension == "technical_depth":
#         score += _score_technical_depth(cv_text)
#     elif dimension == "career_progression":
#         score += _score_career_progression(cv_text)
#     return min(score, criteria["max_score"]), evidence[:3]


# def _score_jd_dimension(cv_text: str, jd_text: str, dim: str, criteria: Dict[str, Any]) -> Tuple[float, List[str]]:
#     cv_lower, jd_lower = cv_text.lower(), jd_text.lower()
#     if dim == "hard_skills":
#         return _score_skills_match(_extract_skills(cv_text), _extract_skills(jd_text), criteria["max_score"])
#     if dim == "responsibilities":
#         return _score_responsibilities_match(cv_text, jd_text, criteria["max_score"])
#     if dim == "seniority":
#         return _score_seniority_match(cv_text, jd_text, criteria["max_score"])
#     # generic keyword overlap
#     score, evidence = 0.0, []
#     for kw in criteria["keywords"]:
#         if kw in cv_lower and kw in jd_lower:
#             evidence.append(_extract_context(cv_text, kw))
#             score += criteria["max_score"] / len(criteria["keywords"])
#     return min(score, criteria["max_score"]), evidence[:3]


# def _score_constraints(cv_text: str, jd_text: str, criteria: Dict[str, Any]) -> Tuple[float, List[str]]:
#     evidence = []
#     if "location" in jd_text.lower() or "remote" in jd_text.lower():
#         evidence.append("Location/remote work requirements identified in JD")
#     if "authorization" in jd_text.lower() or "visa" in jd_text.lower():
#         evidence.append("Work authorization requirements identified in JD")
#     return criteria["max_score"], evidence


# def _extract_context(text: str, keyword: str, chars: int = 100) -> str:
#     i = text.lower().find(keyword.lower())
#     if i == -1:
#         return ""
#     return text[max(0, i - chars): i + len(keyword) + chars].strip()


# def _score_quantified_metrics(cv_text: str) -> float:
#     if re.search(r"\d+%|\d{1,3}(?:,\d{3})+|\d+\s*(users|ms|requests)", cv_text, re.I):
#         return 5
#     return 0


# def _score_technical_depth(cv_text: str) -> float:
#     count = sum(1 for t in ["microservices", "distributed", "observability"] if t in cv_text.lower())
#     return min(count, 5)


# def _score_career_progression(cv_text: str) -> float:
#     count = sum(1 for t in ["senior", "lead", "principal", "manager", "promoted"] if t in cv_text.lower())
#     return min(count, 5)


# def _extract_skills(text: str) -> List[str]:
#     skills = ["python","java","javascript","typescript","react","angular","node.js","django",
#               "fastapi","spring","postgresql","mongodb","redis","aws","azure","gcp","docker",
#               "kubernetes","terraform"]
#     return [s for s in skills if s in text.lower()]


# def _score_skills_match(cv_skills: List[str], jd_skills: List[str], max_score: float) -> Tuple[float, List[str]]:
#     if not jd_skills:
#         return max_score, ["No skills found in JD"]
#     overlap = set(cv_skills) & set(jd_skills)
#     return (len(overlap)/len(jd_skills))*max_score, [f"Matched: {', '.join(overlap)}"]


# def _score_responsibilities_match(cv_text: str, jd_text: str, max_score: float) -> Tuple[float, List[str]]:
#     kws = ["design","implement","develop","build","manage","lead","collaborate","mentor"]
#     cv_res, jd_res = [k for k in kws if k in cv_text.lower()], [k for k in kws if k in jd_text.lower()]
#     overlap = set(cv_res) & set(jd_res)
#     if not overlap: return 0, []
#     return (len(overlap)/len(jd_res))*max_score, [f"Overlap: {', '.join(overlap)}"]


# def _score_seniority_match(cv_text: str, jd_text: str, max_score: float) -> Tuple[float, List[str]]:
#     def _years(t: str) -> int:
#         m = re.search(r"(\d+)\+?\s*years?", t.lower())
#         return int(m.group(1)) if m else 0
#     cvy, jdy = _years(cv_text), _years(jd_text)
#     if not cvy or not jdy: return 0, []
#     if cvy >= jdy:
#         return max_score, [f"CV {cvy}y, JD {jdy}y"]
#     return (cvy/jdy)*max_score, [f"CV {cvy}y, JD {jdy}y"]


# def _band(score: float) -> str:
#     if score >= 90: return "Excellent"
#     if score >= 75: return "Strong"
#     if score >= 60: return "Partial"
#     return "Weak"



import re
from .schemas import ScoreResult, SubScore, Band


def _band(score: float) -> Band:
    if score >= 90:
        return Band.Excellent
    elif score >= 75:
        return Band.Strong
    elif score >= 60:
        return Band.Partial
    return Band.Weak


def score_cv_quality(cv_text: str) -> ScoreResult:
    """Very naive heuristic scoring for CV quality."""
    subscores = []

    dimensions = [
        ("ats_structure", 10, ["email", "phone", "linkedin"]),
        ("writing_clarity", 15, ["developed", "led", "built"]),
        ("quantified_impact", 20, ["%", "users", "reduced", "increased"]),
        ("technical_depth", 15, ["python", "java", "aws", "docker"]),
        ("projects_portfolio", 10, ["project", "github"]),
        ("leadership_skills", 10, ["mentored", "led", "managed"]),
        ("career_progression", 10, ["senior", "lead", "promotion"]),
        ("consistency", 10, ["2020", "2021", "2022"]),  # crude date check
    ]

    total = 0
    for dim, max_score, keywords in dimensions:
        found = [kw for kw in keywords if kw.lower() in cv_text.lower()]
        score = (len(found) / len(keywords)) * max_score if keywords else 0
        evidence = found or ["No evidence found."]
        subscores.append(SubScore(dimension=dim, score=score, max_score=max_score, evidence=evidence))
        total += score

    band = _band(total)
    return ScoreResult(overall_score=round(total, 2), band=band, subscores=subscores)


def score_jd_match(cv_text: str, jd_text: str) -> ScoreResult:
    """Very naive heuristic scoring for JD match."""
    subscores = []

    dimensions = [
        ("hard_skills", 35, ["python", "java", "aws", "docker", "kubernetes"]),
        ("responsibilities", 15, ["design", "implement", "develop"]),
        ("domain_relevance", 10, ["backend", "frontend"]),
        ("seniority", 10, ["5+ years", "senior"]),
        ("nice_to_haves", 5, ["open source", "scrum"]),
        ("education_certs", 5, ["bachelor", "master", "certified"]),
        ("recent_achievements", 10, ["reduced", "increased", "launched"]),
        ("constraints", 10, ["remote", "location"]),
    ]

    total = 0
    for dim, max_score, keywords in dimensions:
        found = [kw for kw in keywords if kw.lower() in (cv_text + jd_text).lower()]
        score = (len(found) / len(keywords)) * max_score if keywords else 0
        evidence = found or ["No evidence found."]
        subscores.append(SubScore(dimension=dim, score=score, max_score=max_score, evidence=evidence))
        total += score

    band = _band(total)
    return ScoreResult(overall_score=round(total, 2), band=band, subscores=subscores)
