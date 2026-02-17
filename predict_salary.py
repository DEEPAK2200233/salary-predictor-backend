import pandas as pd
import joblib

# =========================
# LOAD SAVED ARTIFACTS
# =========================
model = joblib.load("salary_model.pkl")
role_encoder = joblib.load("encoder_role.pkl")
location_encoder = joblib.load("encoder_location.pkl")
work_mode_encoder = joblib.load("encoder_work_mode.pkl")
employment_encoder = joblib.load("encoder_employment_type.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =========================
# USER INPUT (CHANGE VALUES TO TEST)
# =========================
user_input = {
    "role": "Data Scientist",
    "location": "Bengaluru",
    "work_mode": "Remote",
    "employment_type": "Full-time",
    "experience_years": 3,
    "skills": "Python|SQL|Machine Learning"
}

# =========================
# PREPROCESS INPUT
# =========================
# Convert to dataframe
df = pd.DataFrame([user_input])

# Lowercase skills (same as training)
df["skills"] = df["skills"].str.lower()

# Create empty feature row
X = pd.DataFrame(0, index=[0], columns=feature_columns)

# Encode categorical values
X["role"] = role_encoder.transform(df["role"])[0]
X["location"] = location_encoder.transform(df["location"])[0]
X["work_mode"] = work_mode_encoder.transform(df["work_mode"])[0]
X["employment_type"] = employment_encoder.transform(df["employment_type"])[0]

# Numeric feature
X["experience_years"] = df["experience_years"][0]

# Skill columns
for col in feature_columns:
    if col.startswith("has_"):
        skill_name = col.replace("has_", "").replace("_", " ")
        if skill_name in df["skills"][0]:
            X[col] = 1

# =========================
# SCALE FEATURES
# =========================
X_scaled = scaler.transform(X)

# =========================
# PREDICT
# =========================
prediction = model.predict(X_scaled)[0]

print("\n💰 Predicted Salary:", round(prediction, 2), "LPA\n")
