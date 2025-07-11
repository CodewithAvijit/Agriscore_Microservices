from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import numpy as np
from fastapi.responses import PlainTextResponse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pydantic import BaseModel
import joblib 

app = FastAPI(
    title=" Yield Prediction ",
    description="API to Predict yield based on crop,season and humidity",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['*'],
    allow_credentials=True,
)

class INPUTDATA(BaseModel):
    crop: str
    season: str
    state: str
    Area: int
    Production: int
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

scaler = joblib.load("ENCODER-DECODER/scaler.pkl")
crop = joblib.load("ENCODER-DECODER/crop.pkl")
season = joblib.load("ENCODER-DECODER/season.pkl")
state = joblib.load("ENCODER-DECODER/state.pkl")
model = joblib.load("MODEL/histgradboosting.pkl")

@app.get('/', response_class=PlainTextResponse, tags=['root'])
def about():
    return "YIELD PREDICTION SYSTEM"

@app.post('/predict', response_class=PlainTextResponse, tags=['predict'])
def predict(data: INPUTDATA):
    if data.crop not in crop.classes_:
        return f"ERROR: Unknown crop '{data.crop}'"
    if data.season not in season.classes_:
        return f"ERROR: Unknown season '{data.season}'"
    if data.state not in state.classes_:
        return f"ERROR: Unknown state '{data.state}'"

    cropdata = crop.transform([data.crop])[0]
    seasondata = season.transform([data.season])[0]
    statedata = state.transform([data.state])[0]

    input = np.array([[cropdata, seasondata, statedata,
                       data.Area, data.Production,
                       data.Annual_Rainfall, data.Fertilizer, data.Pesticide]])
    
    scaled_input = scaler.transform(input)
    output = model.predict(scaled_input)[0]
    return f"PREDICTED YIELD (kg/hec ) {output:.2f}"

@app.post("/predict_form", response_class=PlainTextResponse, tags=["predict"])
def predict_form(
    crop_input: Literal[
        'Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley',
        'Blackpepper', 'Cardamom', 'Cashewnut', 'Castorseed', 'Coconut',
        'Coriander', 'Cotton(lint)', 'Cowpea(Lobia)', 'Drychillies',
        'Garlic', 'Ginger', 'Gram', 'Groundnut', 'Guarseed', 'Horse-gram',
        'Jowar', 'Jute', 'Khesari', 'Linseed', 'Maize', 'Masoor', 'Mesta',
        'Moong(GreenGram)', 'Moth', 'Nigerseed', 'Oilseedstotal', 'Onion',
        'OtherCereals', 'OtherKharifpulses', 'OtherRabipulses',
        'OtherSummerPulses', 'Peas&beans(Pulses)', 'Potato', 'Ragi',
        'Rapeseed&Mustard', 'Rice', 'Safflower', 'Sannhamp', 'Sesamum',
        'Smallmillets', 'Soyabean', 'Sugarcane', 'Sunflower',
        'Sweetpotato', 'Tapioca', 'Tobacco', 'Turmeric', 'Urad', 'Wheat',
        'otheroilseeds'
    ] = Form(...),
    season_input: Literal['Autumn', 'Kharif', 'Rabi', 'Summer', 'WholeYear', 'Winter'] = Form(...),
    state_input: Literal[
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
        'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
        'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka',
        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
        'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Sikkim',
        'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
        'Uttarakhand', 'West Bengal'
    ] = Form(...),
    Area: int = Form(...),
    Production: int = Form(...),
    Annual_Rainfall: float = Form(...),
    Fertilizer: float = Form(...),
    Pesticide: float = Form(...)
):
    cropdata = crop.transform([crop_input])[0]
    seasondata = season.transform([season_input])[0]
    statedata = state.transform([state_input])[0]

    input_data = np.array([[cropdata, seasondata, statedata,
                            Area, Production, Annual_Rainfall, Fertilizer, Pesticide]])
    
    scaled_input = scaler.transform(input_data)
    output = model.predict(scaled_input)[0]

    return f"PREDICTED YIELD (kg/hec ) {output:.2f}"