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
    

@app.post('/recommend', tags=['Recommendation'])
def recommend_top5_json(data: INPUT_DATA):
    val = np.array([[data.n, data.p, data.k, data.temp, data.humidity, data.ph, data.rainfall]])
    scale = scaler.transform(val)
    proba = model.predict_proba(scale)[0]  # get 1D array of probabilities for one sample
    top5_idx = np.argsort(proba)[::-1][:5]  # top 5 class indices, descending order
    top5_crops = label_decoder.inverse_transform(top5_idx)  # get crop names for top indices
    top5_probs = proba[top5_idx]  # their probabilities

    results = []
    for rank, (crop, prob) in enumerate(zip(top5_crops, top5_probs), 1):
        results.append({'rank': rank, 'crop': crop, 'probability': round(float(prob), 4)})
    return {"top_5_crops": results}
