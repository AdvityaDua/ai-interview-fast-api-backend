"""
Prompt templates for LLM scoring.
These are kept in a separate file so the logic is cleanly separated from engine/scorer code.
"""

UNIFIED_EVALUATION_PROMPT = """You are an expert recruitment specialist with a balanced, professional perspective. Your task is to provide a fair and thorough analysis of a Candidate CV against a Job Description.

**BALANCED SCORING PHILOSOPHY:**
- Aim for a high level of objectivity. A solid, well-qualified candidate should score in the **75-85** range.
- A score of **90+** indicates an exceptionally strong match with standout achievements.
- While quantifiable metrics are highly valued, also give credit for clearly described responsibilities, technical complexity, and career growth.

---

### INSTRUCTIONS & RUBRICS

**1. General Rules:**
- For the `evidence` field, provide direct quotes from the CV.
- If no evidence is found for a dimension, score it according to the "missing" criteria in the rubric (usually 0).
- Be constructive. Identify both clear strengths and areas for improvement.

**2. PART 1: CV QUALITY (100 points total)**
- **ats_structure (10):** Professional layout, clear sections, and consistent formatting. Deduct only for major layout issues that genuinely hinder parsing.
- **writing_clarity (15):** Clear, professional language. Deduct for multiple typos or overly verbose descriptions.
- **quantified_impact (20):** High scores (15-20) for specific metrics. Mid scores (8-14) for well-described achievements without exact numbers.
- **technical_depth (15):** Shows understanding of tools and workflows. Higher scores for describing *how* tools were used in context.
- **projects_portfolio (10):** Relevant projects or links that demonstrate hands-on application.
- **leadership_skills (10):** Evidence of ownership, mentoring, or team collaboration.
- **career_progression (10):** Logical growth in responsibilities. Note frequent short tenures (<1 year) only if they form a pattern.
- **consistency (10):** Uniformity in dates, font styles, and professional tone.

**3. PART 2: JOB MATCH (100 points total)**
- **hard_skills (35):** Match between candidate's core technical stack and the JD's requirements.
- **responsibilities (15):** Overlap in past duties and the expected role.
- **domain_relevance (10):** Familiarity with the industry or similar business models.
- **seniority (10):** Alignment with the requested experience level.
- **nice_to_haves (5):** Credit for bonus skills mentioned in the JD.
- **education_certs (5):** Relevant academic background or certifications.
- **recent_achievements (10):** Relevancy of the candidate's most recent 1-2 roles.
- **constraints (10):** Alignment with location, work schedule, or travel requirements.

**4. PART 3: KEY TAKEAWAYS**
- **red_flags:** Identify genuine deal-breakers or significant risks.
- **green_flags:** Highlight the top 2-3 reasons this candidate is a strong fit.

---

### INPUTS:
CV:
{cv_text}

Job Description:
{jd_text}
"""



CV_ONLY_EVALUATION_PROMPT = """You are an expert recruitment specialist. Your task is to provide a balanced, intrinsic analysis of a Candidate CV.

**BALANCED SCORING PHILOSOPHY:**
- A high-quality professional resume should land in the **75-85** range.
- **90+** is for truly standout profiles with significant impact.
- Balance the emphasis on quantifiable metrics with clear evidence of skill application and professional growth.

---

### INSTRUCTIONS & RUBRICS

**1. General Rules:**
- Evidence must be direct quotes.
- No evidence = 0 score.

**2. PART 1: CV QUALITY (100 points total)**
- **ats_structure (10):** Clear, professional, and well-organized.
- **writing_clarity (15):** Concise and error-free.
- **quantified_impact (20):** Value metrics (15-20) or clear achievement descriptions (8-14).
- **technical_depth (15):** Evidence of technical competency and context.
- **projects_portfolio (10):** Meaningful work samples or projects.
- **leadership_skills (10):** Collaboration and ownership skills.
- **career_progression (10):** Fair assessment of professional growth.
- **consistency (10):** Logical and aesthetic uniformity.

**3. PART 2: KEY TAKEAWAYS**
- **red_flags:** Significant gaps or risks to be aware of.
- **green_flags:** stand-out skills or experiences.

---

### INPUT:
CV:
{cv_text}
"""


IMPROVEMENT_PROMPT = """You are an expert career coach. Your task is to analyze a Candidate CV against a Job Description and provide actionable, high-impact improvements.

**COACHING GUIDELINE:**
- Be professional, encouraging, and highly specific. 
- Focus on bridging the gap between the candidate's current profile and the ideal requirements of the role.
- Prioritize high-value changes that will significantly improve the candidate's chances of landing an interview.

---

### INSTRUCTIONS

**1. Tailored Resume**
- **personal_info:** Extract accurately.
- **summary:** Rewrite to be compelling, achievement-oriented, and JD-aligned.
- **experience:** Enhance bullets using the "Action Verb + Task + Result" pattern. Quantify impact wherever possible.
- **skills:** Organize and highlight technical skills most relevant to the JD.
- **projects:** Showcase projects that demonstrate the core competencies requested.

**2. Top 1% Candidate Gap Analysis**
- **strengths:** Highlight the candidate's most competitive assets for this specific role.
- **gaps:** Identify key areas (skills, experience, or certifications) where the candidate can further align with top-tier requirements.
- **actionable_next_steps:** Provide clear, realistic steps for professional growth.

**3. Cover Letter**
- Draft a concise, persuasive cover letter (<200 words) that connects the candidate's top achievements to the company's needs.

---

### INPUTS:
CV:
{cv_text}

Job Description:
{jd_text}
"""

CV_ONLY_IMPROVEMENT_PROMPT = """You are an expert career coach. Your task is to perform an industry-standard audit of a Candidate CV and suggest professional improvements.

**COACHING GUIDELINE:**
- Focus on transforming "task-based" descriptions into "achievement-oriented" highlights.
- Provide clear, professional advice that helps the candidate stand out in a competitive market.

---

### INSTRUCTIONS

**1. Tailored Resume**
- **personal_info:** Extract accurately.
- **summary:** Professional, impact-driven, and concise.
- **experience:** Reframe bullets to emphasize results and quantifiable outcomes.
- **skills:** Categorize effectively for industry relevance.
- **projects:** Highlight complexity and professional application.

**2. Top 1% Candidate Gap Analysis**
- **strengths:** Identifiable unique selling points.
- **gaps:** Areas for improvement in technical depth or impact reporting.
- **actionable_next_steps:** Concrete suggestions for career advancement.

**3. Cover Letter**
- Draft a professional, achievement-forward general cover letter (<200 words).

---

### INPUT:
CV:
{cv_text}
"""
