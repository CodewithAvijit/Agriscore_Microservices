from fastapi import FastAPI,Form
import numpy as np
from fastapi.responses import PlainTextResponse
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib 
app = FastAPI(
    title="🌱 Crop Recommendation System 🌱",
    description="API to recommend suitable crops based on soil and weather data",
    version="1.0.1"
)

scaler=joblib.load(r"C:\Users\Avijit\Desktop\AgriAssure\CROP_RECOMMENDATION\ENCODER-DECODER\scalerx.pkl")
label_decoder=joblib.load(r"C:\Users\Avijit\Desktop\AgriAssure\CROP_RECOMMENDATION\ENCODER-DECODER\label_en.pkl")
model=joblib.load(r"C:\Users\Avijit\Desktop\AgriAssure\CROP_RECOMMENDATION\MODELS\random_forest.pkl")

@app.get("/", response_class=PlainTextResponse, tags=["Root"])
def about():
    return "🌱 CROP RECOMMENDATION SYSTEM API 🌱"

@app.post('/recommend',response_class=PlainTextResponse)
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
    return f"recommended crop:{crop}"
