from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import io
from ultralytics import YOLO

# --------------------- FastAPI Setup ---------------------
app = FastAPI(
    title="PLANT DISEASE DETECTION",
    description="Pipeline to classify plant image and detect its disease",
)

# --------------------- Device Setup ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Load Models ---------------------
# ✅ YOLO model for PLANT vs NONPLANT
plant_vs_other_model = YOLO("MODELS-PLANTvsOTHERS/best.pt")
plant_vs_other_model.to(device).eval()


# ✅ MobileNet model for HEALTHY vs UNHEALTHY
health_check_model = torch.load("MODELS_HealthCheck/mobilenet_full_model.pth", map_location=device)
health_check_model.to(device).eval()

# ✅ YOLOv8 classification model for disease detection
disease_model = YOLO("MODELS_DISEASE_DETECT/best.pt")

# --------------------- Class Labels ---------------------
health_classes = ['HEALTHY', 'UNHEALTHY']

# --------------------- Transform (for MobileNet only) ---------------------
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
    return tensor, image

# ✅ YOLO model for PLANT vs NONPLANT
def is_plant(image):
    results = plant_vs_other_model(image)
    probs = results[0].probs
    pred_idx = int(probs.top1)
    category = plant_vs_other_model.names[pred_idx]
    return category  # "PLANT" or "NONPLANT"

# ✅ MobileNet for HEALTHY vs UNHEALTHY
def is_healthy(image_tensor):
    with torch.no_grad():
        output = health_check_model(image_tensor)
        pred = int(torch.round(torch.sigmoid(output)).item())
        return health_classes[pred]

# ✅ YOLO disease classification
def predict_disease(image):
    results = disease_model(image)
    probs = results[0].probs
    pred_idx = int(probs.top1)
    disease_name = disease_model.names[pred_idx].replace("_", " ")
    return disease_name

# --------------------- Routes ---------------------
@app.get("/")
def root():
    return {"message": "🚀 FastAPI is running with plant disease detection pipeline!"}

@app.post("/predict/")
async def predict_image(Plant: UploadFile = File(..., description="Upload a plant image")):
    try:
        image_tensor, raw_image = prepare_image(Plant)

        # Step 1: Check if image is of a plant
        category = is_plant(raw_image)
        if category.upper() == "NONPLANT":
            return "CHECK IMAGE: Use Another Image"

        # Step 2: Check if plant is healthy
        health = is_healthy(image_tensor)
        if health == "HEALTHY":
            return "HEALTHY"

        # Step 3: Predict disease using YOLO
        disease = predict_disease(raw_image)
        return disease

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
