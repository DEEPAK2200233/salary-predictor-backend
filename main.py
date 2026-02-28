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
vectorizer = TfidfVectorizer()

@app.post("/api/match_candidates")
def match_candidates(data: RecruitMatchInput):
    try:
        if not data.candidates:
            return {"matches": []}

        # 1️⃣ Prepare text corpus
        candidate_texts = [
            f"{c.get('Skills', '')} {c.get('Experience', '')}"
            for c in data.candidates
        ]

        # Combine role + candidates
        corpus = [data.role] + candidate_texts

        # 2️⃣ TF-IDF Vectorization
        tfidf_matrix = vectorizer.fit_transform(corpus)

        role_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]

        # 3️⃣ Cosine Similarity
        similarities = cosine_similarity(role_vector, candidate_vectors)[0]

        ranked_results = []

        for i, score in enumerate(similarities):
            candidate = data.candidates[i].copy()

            base_score = float(score)

            # Experience boost (log scaling)
            years = float(candidate.get("Experience", 0))
            exp_factor = np.log1p(years) * 0.05

            final_score = base_score + exp_factor

            candidate["match_score"] = round(final_score, 4)
            ranked_results.append(candidate)

        # 4️⃣ Sort & return top 5
        ranked_results.sort(key=lambda x: x["match_score"], reverse=True)

        return {"matches": ranked_results[:5]}

    except Exception as e:
        logger.error(f"Matching error: {e}")
        return {"error": "Internal matching failure"}
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
