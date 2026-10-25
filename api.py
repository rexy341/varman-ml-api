"""Small HTTP service for Varman to request a prediction from the trained model."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
import threading
import time

import joblib
import pandas as pd
import httpx
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

# CWC's public Flood Forecast System supplies station coordinates, thresholds, and
# telemetry. Cache station metadata for 6 hours and gauge readings for 10 minutes so
# map scans do not repeatedly hit the government service.
CWC_BASE = "https://ffs.india-water.gov.in"
CWC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": CWC_BASE + "/",
    "Origin": CWC_BASE,
}
CWC_MAX_DISTANCE_KM = 50
CWC_FRESH_HOURS = 24
CWC_WATER_LEVEL_CODES = {"HHS", "HHA", "HHX", "HHT", "HZS", "HZF"}
_cwc_cache_lock = threading.RLock()
_cwc_stations_cache = {"expires": 0.0, "rows": None}
_cwc_readings_cache = {}


def _cwc_expression(field, operator, value):
    return {"expression": {"valueIsRelationField": False, "fieldName": field,
                           "operator": operator, "value": value}}


def _cwc_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "content"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _cwc_get(path, specification=None):
    params = None
    if specification is not None:
        params = {"specification": json.dumps(specification, separators=(",", ":"))}
    try:
        with httpx.Client(timeout=20, follow_redirects=True, headers=CWC_HEADERS) as client:
            response = client.get(CWC_BASE + path, params=params)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"CWC feed request failed: {path} ({exc})") from exc


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _load_cwc_stations():
    now = time.monotonic()
    with _cwc_cache_lock:
        cached = _cwc_stations_cache
        if cached["rows"] is not None and cached["expires"] > now:
            return cached["rows"]

    static_rows = _cwc_rows(_cwc_get("/iam/api/flood-forecast-static/"))
    geo_rows = []
    # Only river gauges; reservoir/inflow stations are not comparable to a nearby
    # river stage reading.
    for station_type in ("Level", "Base"):
        spec = {"where": _cwc_expression(
            "layerStationStationCode.floodForecastStaticStationCode.type", "eq", station_type
        )}
        geo_rows.extend(_cwc_rows(_cwc_get("/iam/api/layer-station-geo/specification/", spec)))

    static_by_code = {
        str(row.get("stationCode", "")).strip().upper(): row
        for row in static_rows if row.get("stationCode")
    }
    stations = []
    seen = set()
    for geo in geo_rows:
        code = str(geo.get("stationCode", "")).strip().upper()
        if not code or code in seen:
            continue
        static = static_by_code.get(code, {})
        station_type = str(static.get("type") or geo.get("type") or "").strip()
        if station_type not in {"Level", "Base"}:
            continue
        lat, lon = _number(geo.get("lat")), _number(geo.get("lon"))
        if lat is None or lon is None:
            continue
        seen.add(code)
        stations.append({
            "station_code": code,
            "source_station_code": str(geo.get("stationCode") or static.get("stationCode") or code).strip(),
            "name": str(geo.get("name") or static.get("stationName") or code),
            "type": station_type,
            "latitude": lat,
            "longitude": lon,
            "warning_level_m": _number(static.get("warningLevel")),
            "danger_level_m": _number(static.get("dangerLevel")),
        })

    if not stations:
        raise RuntimeError("CWC returned no geolocated river gauge stations")
    with _cwc_cache_lock:
        _cwc_stations_cache.update(rows=stations, expires=time.monotonic() + 6 * 60 * 60)
    return stations


def _parse_cwc_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            # The CWC portal timestamps are local station time (India Standard Time).
            parsed = parsed.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
        return parsed
    except (TypeError, ValueError):
        return None


def _load_cwc_reading(station_code):
    now_mono = time.monotonic()
    with _cwc_cache_lock:
        cached = _cwc_readings_cache.get(station_code)
        if cached and cached[0] > now_mono:
            return cached[1]

    since = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    spec = {
        "where": _cwc_expression("id.stationCode", "eq", station_code),
        "and": _cwc_expression("id.dataTime", "gt", since),
    }
    payload = _cwc_get("/iam/api/new-entry-data/specification/", spec)
    observations = []
    for row in _cwc_rows(payload):
        row_id = row.get("id") or {}
        code = str(row_id.get("datatypeCode", "")).strip().upper()
        if code not in CWC_WATER_LEVEL_CODES:
            continue
        value = _number(row.get("dataValidatedValue"))
        if value is None:
            value = _number(row.get("dataValue"))
        when = _parse_cwc_time(row_id.get("dataTime"))
        if value is not None and when is not None:
            observations.append((when, value, code))

    reading = max(observations, key=lambda item: item[0]) if observations else None
    with _cwc_cache_lock:
        _cwc_readings_cache[station_code] = (time.monotonic() + 10 * 60, reading)
    return reading


def _distance_km(lat1, lon1, lat2, lon2):
    rad = math.radians
    a = (math.sin(rad(lat2 - lat1) / 2) ** 2 +
         math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(rad(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _nearest_cwc_station(lat, lon, radius_km):
    stations = _load_cwc_stations()
    ranked = []
    for station in stations:
        distance = _distance_km(lat, lon, station["latitude"], station["longitude"])
        if distance <= radius_km:
            ranked.append((distance, station))
    if not ranked:
        return {"available": False, "reason": f"No CWC river gauge within {radius_km:g} km.",
                "source": "Central Water Commission Flood Forecast System"}

    ranked.sort(key=lambda item: item[0])
    # Try a few nearest gauges in case the closest station has not reported recently.
    candidates = ranked[:5]
    with ThreadPoolExecutor(max_workers=min(5, len(candidates))) as pool:
        readings = list(pool.map(lambda pair: _load_cwc_reading(pair[1]["source_station_code"]), candidates))

    picked = next(((pair, reading) for pair, reading in zip(candidates, readings)
                   if reading is not None and
        -3600 <= (datetime.now(timezone.utc) - reading[0]).total_seconds() <= CWC_FRESH_HOURS * 3600), None)
    if picked is None:
        pair, reading = next(((pair, reading) for pair, reading in zip(candidates, readings)
                              if reading is not None), (candidates[0], None))
    else:
        pair, reading = picked
    distance, station = pair
    if reading is None:
        return {"available": True, "fresh": False, "status": "no_recent_data", "station": station,
                "distance_km": round(distance, 1), "source": "Central Water Commission Flood Forecast System",
                "reason": "Station found, but no recent water-level reading was returned."}

    observed_at, level, datatype = reading
    age_hours = max(0.0, (datetime.now(timezone.utc) - observed_at).total_seconds() / 3600)
    warning, danger = station["warning_level_m"], station["danger_level_m"]
    if danger is not None and level >= danger:
        status = "above_danger"
    elif warning is not None and level >= warning:
        status = "above_warning"
    elif warning is not None:
        status = "below_warning"
    else:
        status = "thresholds_unavailable"
    return {
        "available": True,
        "fresh": age_hours <= CWC_FRESH_HOURS,
        "status": status,
        "station": station,
        "distance_km": round(distance, 1),
        "current_level_m": round(level, 3),
        "warning_level_m": warning,
        "danger_level_m": danger,
        "observed_at": observed_at.isoformat(),
        "observation_age_hours": round(age_hours, 1),
        "datatype_code": datatype,
        "source": "Central Water Commission Flood Forecast System",
        "source_url": "https://ffs.india-water.gov.in/",
    }


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


@app.get("/cwc/nearby")
def cwc_nearby(lat: float, lon: float, radius_km: float = CWC_MAX_DISTANCE_KM):
    """Return the closest recent CWC river-gauge reading and its published thresholds."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(422, "lat/lon must be valid WGS84 coordinates")
    if not (1 <= radius_km <= 100):
        raise HTTPException(422, "radius_km must be between 1 and 100")
    try:
        return _nearest_cwc_station(lat, lon, radius_km)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


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
