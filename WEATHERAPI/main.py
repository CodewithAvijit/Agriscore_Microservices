from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from typing import List


# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Weather API - Production Ready",
    description="Fetches weather data from Open-Meteo with authentication and metrics.",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# CORS (Set allowed origins for production)
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# -----------------------------------------------------------------------------
# Security Config (use environment variables in production!)
# -----------------------------------------------------------------------------
security = HTTPBasic()
API_USERNAME = os.getenv("API_USERNAME", "agriscore")
API_PASSWORD = os.getenv("API_PASSWORD", "SAPM2025")

# -----------------------------------------------------------------------------
# API Call Counter
# -----------------------------------------------------------------------------
call_count = 0

# -----------------------------------------------------------------------------
# Weather Codes Mapping
# -----------------------------------------------------------------------------
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
}

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------
class WeatherResponse(BaseModel):
    latitude: float
    longitude: float
    time: str
    temperature: float
    windspeed: float
    winddirection: float
    condition: str
    is_day: bool
    relative_humidity: int
    surface_pressure: float
    cloud_cover: int
    call_count: int

class TokenCountResponse(BaseModel):
    call_count: int

# -----------------------------------------------------------------------------
# Auth Dependency
# -----------------------------------------------------------------------------
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username == API_USERNAME and credentials.password == API_PASSWORD:
        return credentials.username
    raise HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )

# -----------------------------------------------------------------------------
# Weather Endpoint
# -----------------------------------------------------------------------------
@app.get("/weather", response_model=WeatherResponse, tags=["Weather Data"])
async def get_weather(
    lat: float = Query(..., ge=-90, le=90, description="Latitude of location"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude of location"),
    current_user: str = Depends(get_current_user)
):
    global call_count
    call_count += 1

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "relativehumidity_2m,surface_pressure,cloudcover",
        "timezone": "auto"
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        current = data.get("current_weather", {})
        hourly = data.get("hourly", {})

        if not current:
            raise HTTPException(status_code=404, detail="Weather data not found.")

        # Match hourly data with current time (nearest hour)
        current_hour_time = current.get("time", "").rsplit(':', 1)[0] + ":00"
        time_index = hourly.get("time", []).index(current_hour_time)

        weather_code = current.get("weathercode")
        condition = WEATHER_CODES.get(weather_code, "Unknown")

        return {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "time": current.get("time"),
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed"),
            "winddirection": current.get("winddirection"),
            "condition": condition,
            "is_day": bool(current.get("is_day")),
            "relative_humidity": hourly.get("relativehumidity_2m", [])[time_index],
            "surface_pressure": hourly.get("surface_pressure", [])[time_index],
            "cloud_cover": hourly.get("cloudcover", [])[time_index],
            "call_count": call_count
        }

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

# -----------------------------------------------------------------------------
# API Metrics Endpoint
# -----------------------------------------------------------------------------
@app.get("/token-count", response_model=TokenCountResponse, tags=["API Metrics"])
async def get_token_count(current_user: str = Depends(get_current_user)):
    global call_count
    return {"call_count": call_count}

# -----------------------------------------------------------------------------
# Run Command (for production with uvicorn/gunicorn)
# -----------------------------------------------------------------------------
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
