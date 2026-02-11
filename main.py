from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = joblib.load("xcod.pkl")
features = joblib.load("model_features.pkl")

app = FastAPI(title="Phone Addiction Predictor")

class UserInput(BaseModel):
    Age: int
    Gender: int
    School_Grade: int
    Daily_Usage_Hours: float
    Sleep_Hours: float
    Academic_Performance: int
    Social_Interactions: int
    Exercise_Hours: float
    Anxiety_Level: int
    Depression_Level: int
    Self_Esteem: int
    Parental_Control: int
    Screen_Time_Before_Bed: float
    Phone_Checks_Per_Day: int
    Apps_Used_Daily: int
    Time_on_Social_Media: float
    Time_on_Gaming: float
    Time_on_Education: float
    Phone_Usage_Purpose: int
    Family_Communication: int
    Weekend_Usage_Hours: float

def addiction_label(score: float):
    if score < 4:
        return "Low"
    elif score < 7:
        return "Moderate"
    else:
        return "High"

@app.get("/")
def root():
    return {
        "status": "Backend running 🚀",
        "docs": "/docs",
        "predict": "/predict"
    }

@app.post("/predict")
def predict(data: UserInput):
    df = pd.DataFrame([[getattr(data, f) for f in features]], columns=features)

    score = float(model.predict(df)[0])
    label = addiction_label(score)

    return {
        "addiction_score": round(score, 2),
        "addiction_level": label
    }

