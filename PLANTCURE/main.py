from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import io

# --------------------- FastAPI Setup ---------------------
app = FastAPI(
    title="PLANT DISEASE DETECTION",
    description="Pipeline to classify plant image and detect its disease",
)

# --------------------- Device Setup ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Load Models ---------------------
plant_vs_other_model = torch.load("MODELS-PLANTvsOTHERS/mobilenet_full_model.pth", map_location=device)
plant_vs_other_model.to(device).eval()

health_check_model = torch.load("MODELS_HealthCheck/mobilenet_full_model.pth", map_location=device)
health_check_model.to(device).eval()

disease_model = torch.load("MODELS_DISEASE_DETECT/mobilenet.pth", map_location=device)
disease_model.to(device).eval()

# --------------------- Class Labels ---------------------
plant_vs_other_classes = ['OTHERS', 'PLANT']
health_classes = ['HEALTHY', 'UNHEALTHY']

disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato__Target_Spot', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato_healthy'
]

# --------------------- Transform ---------------------
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# --------------------- Helper Functions ---------------------

def prepare_image(upload_file: UploadFile):
    image_bytes = upload_file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def is_plant(image_tensor):
    with torch.no_grad():
        output = plant_vs_other_model(image_tensor)
        pred = int(torch.round(torch.sigmoid(output)).item())
        return plant_vs_other_classes[pred]

def is_healthy(image_tensor):
    with torch.no_grad():
        output = health_check_model(image_tensor)
        pred = int(torch.round(torch.sigmoid(output)).item())
        return health_classes[pred]

def predict_disease(image_tensor):
    with torch.no_grad():
        output = disease_model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        disease_name = disease_classes[pred_idx].replace("_", " ")
        return f"{disease_name}"

# --------------------- Routes ---------------------

@app.get("/")
def root():
    return {"message": "🚀 FastAPI is running with plant disease detection pipeline!"}

@app.post("/predict/", response_class=PlainTextResponse)
async def predict_image(Plant: UploadFile = File(..., description="Upload a plant image")):
    try:
        image_tensor = prepare_image(Plant)

        # Step 1: Check if image is of a plant
        category = is_plant(image_tensor)
        if category == "OTHERS":
            return "CHECK IMAGE: Use Another Image"

        # Step 2: Check if plant is healthy
        health = is_healthy(image_tensor)
        if health == "HEALTHY":
            return "HEALTHY"

        # Step 3: Predict disease
        disease = predict_disease(image_tensor)
        return disease

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
