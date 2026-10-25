"""Small HTTP service for Varman to request a prediction from the trained model."""
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "models" / "flood_model.joblib"
MAP_FEATURES = {
    "latitude", "longitude", "rainfall_hourly_mm", "hourly_rain_age_hours",
    "rainfall_daily_mm", "daily_rain_age_hours",
}
app = FastAPI(title="Varman flood prediction API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype API; restrict this to the deployed Varman origin for production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    features: dict
    model: str = "random_forest"


@lru_cache(maxsize=2)
def load_model(model_name: str):
    """Load each trained artifact once; area scans request many predictions."""
    model_path = MODEL_PATH.parent / f"{model_name}.joblib"
    if not model_path.exists():
        raise HTTPException(503, "Models not trained yet. Run train.py first.")
    artifact = joblib.load(model_path)
    columns = artifact["feature_columns"]
    if artifact.get("map_input_schema") != "rainfall_location_v1" or set(columns) != MAP_FEATURES:
        raise HTTPException(503, "This model uses the older river-level input schema. Retrain with the updated train.py before map predictions.")
    return artifact


@app.get("/health")
def health():
    available = [name for name in ("random_forest", "xgboost")
                 if (MODEL_PATH.parent / f"{name}.joblib").exists()]
    return {"ready": bool(available), "models": available}


@app.post("/predict")
def predict(request: PredictionRequest):
    if request.model not in {"random_forest", "xgboost"}:
        raise HTTPException(400, "model must be random_forest or xgboost")
    artifact = load_model(request.model)
    columns = artifact["feature_columns"]
    row = {column: request.features.get(column) for column in columns}
    frame = pd.DataFrame([row], columns=columns)
    pipeline = artifact["pipeline"]
    raw = pipeline.predict(frame)[0]
    if artifact["task"] == "classification":
        label = artifact["classes"][int(raw)]
        probabilities = pipeline.predict_proba(frame)[0]
        confidence = {str(name): float(prob) for name, prob in zip(artifact["classes"], probabilities)}
        return {"prediction": str(label), "probabilities": confidence,
                "model": artifact["model_name"], "target": artifact["target"]}
    return {"prediction": float(raw), "model": artifact["model_name"], "target": artifact["target"]}
