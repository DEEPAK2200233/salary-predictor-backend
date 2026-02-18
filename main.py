from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
import logging

try:
    jobs_df = pd.read_csv("cleaned_dataset.csv").fillna("")
except:
    jobs_df = pd.DataFrame()
# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- FastAPI app -----------------
app = FastAPI(
    title="Salary Predictor & Job Search API",
    version="1.0.0"
)

# CORS (allow your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Database connection -----------------
def get_db_connection():
    # Use the environment variable first
    db_url = os.getenv("DATABASE_URL")
    
    # Fallback to your hardcoded one if the env var is missing
    if not db_url:
        db_url = "postgresql://job_user:3gx9r5k7H5cPbF7VGE76GmXdIX5Ai8Yu@dpg-d6a7kf3h46gs738aej5g-a.oregon-postgres.render.com/job_market_db_fdli"

    # CRITICAL: Fix for Render's SSL requirement
    # Render URLs often need 'sslmode=require'
    if "sslmode=" not in db_url:
        connector = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{connector}sslmode=require"
    
    # Also, some libraries prefer 'postgresql://' over 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    logger.info("Connecting to DB...")
    return psycopg2.connect(db_url)

# ----------------- Load ML artifacts -----------------
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
    logger.info("ML artifacts loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML artifacts: {e}")

# ----------------- Schemas & utils -----------------
class SalaryInput(BaseModel):
    role: str
    location: str
    work_mode: str
    employment_type: str
    experience_years: float
    skills: str

def normalize_text(text: str) -> str:
    return text.strip().lower() if text else ""

def clean_role(role: str) -> str:
    role = normalize_text(role)
    mapping = {
        "data scientist": "Data Scientist",
        "datascientist": "Data Scientist",
        "ds": "Data Scientist",
        "data analyst": "Data Analyst",
        "analyst": "Data Analyst",
        "business analyst": "Business Analyst",
        "bi analyst": "BI Analyst",
        "analytics engineer": "Analytics Engineer",
        "senior data analyst": "Senior Data Analyst",
    }
    return mapping.get(role, role.title())

def clean_location(location: str) -> str:
    location = normalize_text(location)
    mapping = {
        "bangalore": "Bengaluru",
        "blr": "Bengaluru",
        "delhi": "Delhi",
        "mumbai": "Mumbai",
        "remote": "Remote",
    }
    return mapping.get(location, location.title())

def clean_employment(emp: str) -> str:
    emp = normalize_text(emp)
    mapping = {
        "fulltime": "Full-time",
        "full time": "Full-time",
        "ft": "Full-time",
        "intern": "Internship",
        "internship": "Internship",
        "contract": "Contract",
    }
    return mapping.get(emp, "Full-time")

def clean_work_mode(mode: str) -> str:
    mode = normalize_text(mode)
    mapping = {
        "remote": "Remote",
        "onsite": "Onsite",
        "office": "Onsite",
        "hybrid": "Hybrid",
    }
    return mapping.get(mode, "Remote")

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
    return {"status": "online", "message": "Salary Predictor & Job Search API is running"}

@app.post("/predict_salary")
def predict_salary(data: SalaryInput):
    try:
        if not all([model, role_encoder, location_encoder, work_mode_encoder,
                    employment_encoder, scaler, feature_columns]):
            raise RuntimeError("Model artifacts not loaded on server")

        user_input = {
            "role": clean_role(data.role),
            "location": clean_location(data.location),
            "work_mode": clean_work_mode(data.work_mode),
            "employment_type": clean_employment(data.employment_type),
            "experience_years": data.experience_years,
            "skills": clean_skills(data.skills),
        }

        df = pd.DataFrame([user_input])

        # Encode categorical features
        df["role"] = role_encoder.transform(df["role"])
        df["location"] = location_encoder.transform(df["location"])
        df["work_mode"] = work_mode_encoder.transform(df["work_mode"])
        df["employment_type"] = employment_encoder.transform(df["employment_type"])

        # Ensure correct column order
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale and predict
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]

        return {"salary": float(round(prediction, 2))}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

@app.get("/api/jobs")
def get_jobs(
    jobTitle: str = "",
    location: str = "",
    minSalary: float = 0
):
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
            "company": r.get("company", "Tech Company"),
            "location": r["location"],
            "salary": f"{round(r['salary_lpa'],1) if pd.notna(r['salary_lpa']) else 0} LPA",
            "skills": r["skills"],
            "experience": f"{r['experience_years']} yrs"
        })

    return {"jobs": jobs}
