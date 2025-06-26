from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI(
    title="🌱 Crop Recommendation System 🌱",
    description="API to recommend suitable crops based on soil and weather data",
    version="1.0.1"
)

@app.get("/", response_class=PlainTextResponse, tags=["Root"])
def about():
    return "🌱 CROP RECOMMENDATION SYSTEM API 🌱"
