from fastapi import FastAPI,Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi.responses import PlainTextResponse
from sklearn.preprocessing import LabelEncoder,StandardScaler
from pydantic import BaseModel
import joblib 
app = FastAPI(
    title="🌱 Crop Recommendation System 🌱",
    description="API to recommend suitable crops based on soil and weather data",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['*'],
    allow_credentials=True,
)

scaler = joblib.load("ENCODER-DECODER/scalerx.pkl")
label_decoder = joblib.load("ENCODER-DECODER/label_en.pkl")
model = joblib.load("MODELS/random_forest.pkl")

class INPUT_DATA(BaseModel):
    n:int
    p:int
    k:int
    temp: float
    humidity: float
    ph: float 
    rainfall: float

@app.get("/", response_class=PlainTextResponse, tags=["Root"])
def about():
    return "🌱 CROP RECOMMENDATION SYSTEM API 🌱"
@app.post('/recommend-json',tags=['Recommendation'])
def recommend_json(data:INPUT_DATA):
    val=np.array([[data.n,data.p,data.k,data.temp,data.humidity,data.ph,data.rainfall]])
    scale=scaler.transform(val)
    prediction=model.predict(scale)
    output=label_decoder.inverse_transform(prediction)[0]
    return {output}
    

@app.post('/recommend',tags=['Recommendation'],response_class=PlainTextResponse)
def recommend(
    n: int = Form(..., ge=0, description='Nitrogen'),
    p: int = Form(..., ge=0, description='Phosphorus'),
    k: int = Form(..., ge=0, description='Potassium'),
    temp: float = Form(..., description='Temperature'),
    humidity: float = Form(..., description='Humidity'),
    ph: float = Form(..., ge=0.0, le=14.0, description='pH Level'),
    rainfall: float = Form(..., ge=0.0, description='Rainfall in mm')
):
    input=np.array([[n,p,k,temp,humidity,ph,rainfall]])
    scaled_input=scaler.transform(input)
    output=model.predict(scaled_input)
    crop=label_decoder.inverse_transform(output)[0]
    return f"recommended crop: {crop.capitalize()}"
