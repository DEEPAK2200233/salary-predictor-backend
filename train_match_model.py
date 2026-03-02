import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
roles_df = pd.read_csv("roles_skills.csv")
candidates_df = pd.read_csv("candidates list.csv")

training_rows = []

for _, role_row in roles_df.iterrows():

    role = role_row["role"].lower()
    required_skills = [
        s.strip().lower()
        for s in role_row["required_skills"].split(",")
    ]

    for _, candidate in candidates_df.iterrows():

        candidate_skills = [
            s.strip().lower()
            for s in str(candidate["Skills"]).split(",")
        ]

        # Feature 1: Skill overlap
        matches = len(set(required_skills) & set(candidate_skills))
        skill_score = matches / len(required_skills)

        # Feature 2: Experience
        exp_score = min(candidate["Experience"] / 10, 1)

        # Feature 3: Salary compatibility
        salary_score = 1 - abs(15 - candidate["Expected Salary"]) / 15
        salary_score = max(0, salary_score)

        # Label
        label = 1 if skill_score >= 0.4 else 0

        training_rows.append([
            skill_score,
            exp_score,
            salary_score,
            label
        ])

df = pd.DataFrame(training_rows, columns=[
    "skill_score",
    "exp_score",
    "salary_score",
    "label"
])

X = df[["skill_score", "exp_score", "salary_score"]]
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "candidate_match_model.pkl")

print("Model trained and saved!")