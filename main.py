from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Optional

roles_df = pd.read_csv("roles_skills.csv")
# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- FastAPI app -----------------
app = FastAPI(
    title="Salary Predictor + AI Recruitment API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Load Dataset -----------------
try:
    jobs_df = pd.read_csv("cleaned_dataset.csv").fillna("")
except:
    jobs_df = pd.DataFrame()

# ----------------- Load Salary ML Artifacts -----------------
model = None
role_encoder = None
location_encoder = None
work_mode_encoder = None
employment_encoder = None
scaler = None
feature_columns = None

try:
    model = joblib.load("salary_model.pkl")
    role_encoder = joblib.load("encoder_role.pkl")
    location_encoder = joblib.load("encoder_location.pkl")
    work_mode_encoder = joblib.load("encoder_work_mode.pkl")
    employment_encoder = joblib.load("encoder_employment_type.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    logger.info("Salary ML artifacts loaded successfully")
except Exception as e:
    logger.error(f"Error loading salary ML artifacts: {e}")


# ----------------- Schemas -----------------
class SalaryInput(BaseModel):
    role: str
    location: str
    work_mode: str
    employment_type: str
    experience_years: float
    skills: str


class RecruitMatchInput(BaseModel):
    role: str
    candidates: list


# ----------------- Utility Functions -----------------
def normalize_text(text: str) -> str:
    return text.strip().lower() if text else ""


def clean_skills(skills: str) -> str:
    if not skills:
        return ""
    skills = skills.lower()
    for sep in [",", ";"]:
        skills = skills.replace(sep, "|")
    skills = skills.replace(" ", "|")
    parts = list(set([s.strip().title() for s in skills.split("|") if s.strip()]))
    return "|".join(parts)


# ----------------- Routes -----------------

@app.get("/")
def health_check():
    return {"status": "online", "message": "AI Career Intelligence API running"}


# ----------------- Salary Prediction -----------------
@app.post("/predict_salary")
def predict_salary(data: SalaryInput):
    try:
        if not all([model, role_encoder, location_encoder,
                    work_mode_encoder, employment_encoder,
                    scaler, feature_columns]):
            raise RuntimeError("Salary model artifacts not loaded")

        user_input = {
            "role": data.role.title(),
            "location": data.location.title(),
            "work_mode": data.work_mode.title(),
            "employment_type": data.employment_type.title(),
            "experience_years": data.experience_years,
            "skills": clean_skills(data.skills),
        }

        df = pd.DataFrame([user_input])

        df["role"] = role_encoder.transform(df["role"])
        df["location"] = location_encoder.transform(df["location"])
        df["work_mode"] = work_mode_encoder.transform(df["work_mode"])
        df["employment_type"] = employment_encoder.transform(df["employment_type"])

        df = df.reindex(columns=feature_columns, fill_value=0)
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)[0]

        return {"salary": float(round(prediction, 2))}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


# ----------------- AI Recruitment Matching -----------------
# ----------------- Lightweight AI Recruitment Matching -----------------
# ----------------- AI Recruitment Matching -----------------

class RecruitMatchInput(BaseModel):
    role: str
    candidates: list
    company_budget: Optional[float] = None


@app.post("/api/match_candidates")
def match_candidates(data: RecruitMatchInput):
    try:
        if not data.candidates:
            return {"matches": []}

        role_text = data.role.strip().lower()
        company_budget = data.company_budget if data.company_budget else 15.0

        # ✅ Proper role matching
        role_row = roles_df[
            roles_df["role"].str.lower().str.contains(role_text)
        ]

        if role_row.empty:
            return {"error": f"Role '{data.role}' not found in system"}

        required_skills = [
            skill.strip().lower()
            for skill in role_row.iloc[0]["required_skills"].split(",")
        ]

        ranked_results = []

        for candidate in data.candidates:
            candidate_copy = candidate.copy()
            candidate_skill_list = [
                s.strip().lower()
                for s in str(candidate.get("Skills", "")).split(",")
            ]

            matches = len(
                set(required_skills) & set(candidate_skill_list)
            )
            skill_score = matches / len(required_skills) if required_skills else 0

            # -------------------------
            # Experience Score
            # -------------------------
            try:
                years = float(candidate.get("Experience", 0))
            except:
                years = 0

            exp_score = min(years / 10, 1)

            # -------------------------
            # Salary Score
            # -------------------------
            try:
                expected_salary = float(candidate.get("Expected Salary", 0))
            except:
                expected_salary = 0

            if expected_salary > 0:
                salary_diff = abs(company_budget - expected_salary)
                salary_score = max(0, 1 - (salary_diff / company_budget))
            else:
                salary_score = 0

            # -------------------------
            # Final Weighted Score
            # -------------------------
           # Final Weighted Score
            final_score = (
                0.6 * skill_score +
                0.3 * exp_score +
                0.1 * salary_score
            )
            candidate_copy["match_score"] = round(final_score, 4)

# Only include if skill match exists
            if skill_score >= 0.3:
                ranked_results.append(candidate_copy)

        ranked_results.sort(key=lambda x: x["match_score"], reverse=True)

        return {"matches": ranked_results[:5]}

    except Exception as e:
        logger.error(f"Matching error: {e}")
        return {"error": "Internal matching failure"}


@app.get("/api/jobs")
def get_jobs(jobTitle: str = "", location: str = "", minSalary: float = 0):

    if jobs_df.empty:
        return {"jobs": []}

    df = jobs_df.copy()

    if jobTitle:
        df = df[df["role"].str.contains(jobTitle, case=False, na=False)]

    if location:
        df = df[df["location"].str.contains(location, case=False, na=False)]

    if minSalary and minSalary > 0:
        df = df[df["salary_lpa"] >= minSalary]

    df = df.head(50)

    jobs = []
    for _, r in df.iterrows():
        jobs.append({
            "title": r["role"],
            "company": r["company"],
            "location": r["location"],
            "salary": f"{r['salary_lpa']} LPA",
            "skills": r["skills"],
            "experience": f"{int(r['experience_years'])} yrs",
            "jobType": r["employment_type"]
        })

    return {"jobs": jobs}


# ----------------- Skill Gap Analysis -----------------
@app.get("/api/skills")
def skill_gap(role: str = "", user_skills: str = ""):

    df = jobs_df.copy()

    if role:
        df = df[df["role"].str.contains(role, case=False, na=False)]

    all_skills = []

    for skills in df["skills"].dropna():
        parts = [s.strip().lower() for s in skills.split("|")]
        all_skills.extend(parts)

    skill_counts = pd.Series(all_skills).value_counts().head(15)

    top_skills = list(skill_counts.index)

    user_skill_list = [s.strip().lower() for s in user_skills.split(",") if s.strip()]
    missing_skills = [s for s in top_skills if s not in user_skill_list]

    return {
        "top_skills": top_skills,
        "missing_skills": missing_skills
    }
