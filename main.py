from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
import logging

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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

# ----------------- Database Connection -----------------
def get_db_connection():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        db_url = "postgresql://job_user:3gx9r5k7H5cPbF7VGE76GmXdIX5Ai8Yu@dpg-d6a7kf3h46gs738aej5g-a.oregon-postgres.render.com/job_market_db_fdli"

    if "sslmode=" not in db_url:
        connector = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{connector}sslmode=require"

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    logger.info("Connecting to DB...")
    return psycopg2.connect(db_url)


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


# ----------------- Load AI Matching Model -----------------
embedding_model = None

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")


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
@app.post("/api/match_candidates")
def match_candidates(data: RecruitMatchInput):
    try:
        if not embedding_model:
            raise RuntimeError("Embedding model not loaded")

        if not data.candidates:
            return {"matches": []}

        # 1️⃣ Embed job role
        role_embedding = embedding_model.encode([data.role])

        # 2️⃣ Create candidate text corpus
        candidate_texts = []
        for c in data.candidates:
            text = f"{c.get('Name','')} {c.get('Skills','')} {c.get('Experience','')}"
            candidate_texts.append(text)

        candidate_embeddings = embedding_model.encode(candidate_texts)

        # 3️⃣ Compute similarity
        similarities = cosine_similarity(role_embedding, candidate_embeddings)[0]

        ranked_results = []

        for i, score in enumerate(similarities):
            candidate = data.candidates[i].copy()

            # 🔥 Experience Boost (optional weighting)
            experience = float(candidate.get("Experience", 0))
            weighted_score = float(score) + (experience * 0.02)

            candidate["match_score"] = round(weighted_score, 4)
            ranked_results.append(candidate)

        ranked_results.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "matches": ranked_results[:5]
        }

    except Exception as e:
        logger.error(f"Matching error: {e}")
        return {"error": str(e)}


# ----------------- Job Search API -----------------
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
