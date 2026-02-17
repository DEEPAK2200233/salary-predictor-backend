from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (safe for project/demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load all saved artifacts
model = joblib.load("salary_model.pkl")
role_encoder = joblib.load("encoder_role.pkl")
location_encoder = joblib.load("encoder_location.pkl")
work_mode_encoder = joblib.load("encoder_work_mode.pkl")
employment_encoder = joblib.load("encoder_employment_type.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


class SalaryInput(BaseModel):
    role: str
    location: str
    work_mode: str
    employment_type: str
    experience_years: float
    skills: str


@app.post("/predict_salary")
def predict_salary(data: SalaryInput):
    try:
        user_input = {
            "role": data.role,
            "location": data.location,
            "work_mode": data.work_mode,
            "employment_type": data.employment_type,
            "experience_years": data.experience_years,
            "skills": data.skills
        }

        df = pd.DataFrame([user_input])

        # Encode
        df["role"] = role_encoder.transform(df["role"])
        df["location"] = location_encoder.transform(df["location"])
        df["work_mode"] = work_mode_encoder.transform(df["work_mode"])
        df["employment_type"] = employment_encoder.transform(df["employment_type"])

        # Align columns
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]

        return {"salary": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
