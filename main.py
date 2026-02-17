from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
import logging

# Set up logging to see errors in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE CONNECTION ---
def get_db_connection():
    # It looks for DATABASE_URL in Render Environment Variables
    # If not found, it defaults to the string you provided
    db_url = os.getenv("DATABASE_URL", "postgresql://job_user:3gx9r5k7H5cPbF7VGE76GmXdIX5Ai8Yu@dpg-d6a7kf3h46gs738aej5g-a.oregon-postgres.render.com/job_market_db_fdli")
    return psycopg2.connect(db_url)

# --- LOAD ML ARTIFACTS ---
# Ensure these files are in your root directory
try:
    model = joblib.load("salary_model.pkl")
    role_encoder = joblib.load("encoder_role.pkl")
    location_encoder = joblib.load("encoder_location.pkl")
    work_mode_encoder = joblib.load("encoder_work_mode.pkl")
    employment_encoder = joblib.load("encoder_employment_type.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    logger.info("ML Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML artifacts: {e}")

# --- SCHEMAS & UTILS ---
class SalaryInput(BaseModel):
    role: str
    location: str
    work_mode: str
    employment_type: str
    experience_years: float
    skills: str

def normalize_text(text):
    return text.strip().lower() if text else ""

def clean_role(role):
    role = normalize_text(role)
    mapping = {
        "data scientist": "Data Scientist", "ds": "Data Scientist",
        "data analyst": "Data Analyst", "analyst": "Data Analyst",
        "business analyst": "Business Analyst", "bi analyst": "BI Analyst",
        "analytics engineer": "Analytics Engineer", "senior data analyst": "Senior Data Analyst"
    }
    return mapping.get(role, role.title())

def clean_location(location):
    location = normalize_text(location)
    mapping = {
        "bangalore": "Bengaluru", "blr": "Bengaluru",
        "delhi": "Delhi", "mumbai": "Mumbai", "remote": "Remote"
    }
    return mapping.get(location, location.title())

def clean_employment(emp):
    emp = normalize_text(emp)
    mapping = {"fulltime": "Full-time", "ft": "Full-time", "intern": "Internship", "contract": "Contract"}
    return mapping.get(emp, "Full-time")

def clean_work_mode(mode):
    mode = normalize_text(mode)
    mapping = {"remote": "Remote", "onsite": "Onsite", "office": "Onsite", "hybrid": "Hybrid"}
    return mapping.get(mode, "Remote")

def clean_skills(skills):
    if not skills: return ""
    skills = skills.lower()
    for sep in [",", ";"]:
        skills = skills.replace(sep, "|")
    skills = skills.replace(" ", "|")
    parts = list(set([s.strip().title() for s in skills.split("|") if s.strip()]))
    return "|".join(parts)

# --- ROUTES ---

@app.get("/")
def health_check():
    return {"status": "online", "message": "Salary Predictor API is running"}

@app.post("/predict_salary")
def predict_salary(data: SalaryInput):
    try:
        user_input = {
            "role": clean_role(data.role),
            "location": clean_location(data.location),
            "work_mode": clean_work_mode(data.work_mode),
            "employment_type": clean_employment(data.employment_type),
            "experience_years": data.experience_years,
            "skills": clean_skills(data.skills)
        }

        df = pd.DataFrame([user_input])

        # Encoding
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

@app.get("/api/jobs")
def get_jobs(
    jobTitle: str = Query("", description="Job Title search term"),
    location: str = Query("", description="Location search term"),
    minSalary: float = Query(0, description="Minimum salary in LPA")
):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT role, company, location, salary_lpa, skills, experience_years
            FROM job_postings
            WHERE 1=1
        """
        values = []

        if jobTitle:
            query += " AND role ILIKE %s"
            values.append(f"%{jobTitle}%")

        if location:
            query += " AND location ILIKE %s"
            values.append(f"%{location}%")

        if minSalary > 0:
            query += " AND salary_lpa >= %s"
            values.append(minSalary)

        query += " LIMIT 50"

        cursor.execute(query, tuple(values))
        rows = cursor.fetchall()

        jobs = []
        for r in rows:
            jobs.append({
                "title": r[0],
                "company": r[1],
                "location": r[2],
                "salary": f"{round(r[3], 1) if r[3] else 0} LPA",
                "skills": r[4],
                "experience": f"{r[5]} yrs"
            })

        cursor.close()
        return {"jobs": jobs}

    except Exception as e:
        logger.error(f"Database error: {e}")
        return {"error": "Internal Server Error", "details": str(e)}
    finally:
        if conn:
            conn.close()
