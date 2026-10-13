# AgriScore Microservices

A smart agriculture platform built as a set of microservices to assist farmers and agricultural stakeholders with data-driven decision making. This project brings together machine learning, plant disease detection, and real-time weather intelligence to improve crop planning, reduce losses, and increase productivity.

This repository contains multiple independent services:

- AgriYieldPro — Yield prediction service
- FarmPlanner — Crop recommendation system
- PlantCure — Plant disease detection pipeline
- WeatherAPI — Weather information service

---

## 1. Project Overview

Agriculture is a critical sector, but farmers still face common challenges such as:

- Uncertain crop yield
- Wrong crop selection
- Disease outbreaks in plants
- Lack of real-time weather data
- Poor decision-making due to limited information

This project solves these challenges by using Artificial Intelligence and Machine Learning to help predict:

- Which crop should be grown
- How much yield can be expected
- Whether a plant is healthy or diseased
- What the weather conditions are in a specific area

In simple words, this system helps make farming more scientific, efficient, and profitable.

---

## 2. Problem Statement

Traditional farming depends heavily on:

- Farmer experience
- Local knowledge
- Seasonal assumptions
- Limited weather access
- Delayed disease detection

These often lead to:

- Reduced crop productivity
- Financial loss
- Incorrect use of fertilizers and pesticides
- Crop failure
- Poor planning and low yield

The goal of this project is to reduce uncertainty by using AI models and environmental data.

---

## 3. Business Value

This project is valuable for:

- Farmers
- Agricultural consultants
- Agribusiness companies
- Research institutions
- Government agricultural support systems
- Smart farming applications

Benefits include:

- Better crop planning
- Increased yield
- Reduced crop loss
- Faster plant disease detection
- Better use of resources
- Improved sustainability in farming

---

## 4. Solution Architecture

This repository follows a microservice-based architecture. Each AI function is implemented as a separate service, making the system modular and easier to deploy, scale, and maintain.

```text
+------------------------+
|      Farmer / User     |
|    or Frontend App     |
+-----------+------------+
            |
            v
+------------------------+
|      WeatherAPI        |
| - Real-time weather    |
| - Climate data         |
+-----------+------------+
            |
            +-------------------------+
            |                         |
            v                         v
+---------------------+     +----------------------+
| FarmPlanner         |     | AgriYieldPro         |
| - Crop suggestion   |     | - Yield prediction   |
| - RandomForest      |     | - Regression model   |
+----------+----------+     +-----------+----------+
           |                              |
           |                              |
           v                              v
+-----------------------------------------------+
|                 PlantCure                     |
| - Plant vs Non-Plant                        |
| - Healthy vs Unhealthy                      |
| - Disease Classification                   |
+-----------------------------------------------+
```

This architecture allows each service to work independently and be deployed separately.

---

## 5. Modules in This Project

### 5.1 AgriYieldPro

This module predicts crop yield based on agricultural and environmental data such as:

- Crop type
- Season
- State
- Area
- Production
- Annual rainfall
- Fertilizer amount
- Pesticide amount

#### Technologies Used
- Python
- FastAPI
- scikit-learn
- joblib
- HistGradientBoostingRegressor

#### What It Does
It estimates yield in kg/hectare using a trained machine learning regression model.

#### Example Input

```json
{
  "crop": "Rice",
  "season": "Kharif",
  "state": "West Bengal",
  "Area": 500,
  "Production": 2500,
  "Annual_Rainfall": 1400,
  "Fertilizer": 100,
  "Pesticide": 50
}
```

#### Example Output

```text
PREDICTED YIELD (kg/hec ) 4125.33
```

#### Why It Matters
It helps farmers estimate expected yield before planting and better plan their production.

---

### 5.2 FarmPlanner

This service recommends the best crop to grow based on soil and climate conditions.

#### Input Parameters
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

#### Model Used
- RandomForestClassifier

#### Example Input

```json
{
  "n": 90,
  "p": 42,
  "k": 43,
  "temp": 20.8,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 200
}
```

#### Example Output

```json
{
  "top_5_crops": [
    {"rank": 1, "crop": "rice", "probability": 0.82},
    {"rank": 2, "crop": "maize", "probability": 0.72},
    {"rank": 3, "crop": "jute", "probability": 0.63}
  ]
}
```

#### Why It Matters
This helps farmers choose crops that are naturally suitable for land conditions, reducing risk and maximizing profit.

---

### 5.3 PlantCure

This module detects plant health and disease by analyzing uploaded plant images.

#### Pipeline Flow
1. Detect whether the image contains a plant or not
2. Check whether the plant is healthy or unhealthy
3. If unhealthy, predict the plant disease

#### Models Used
- YOLO model for plant vs non-plant classification
- MobileNet model for healthy vs unhealthy detection
- YOLO disease classification model

#### Typical Behavior
- If the image is not a plant → `CHECK IMAGE: Use Another Image`
- If the plant is healthy → `HEALTHY`
- If the plant is unhealthy → disease name is predicted

#### Why It Matters
This is extremely useful for early detection of plant diseases, which helps prevent crop loss.

---

### 5.4 WeatherAPI

This service fetches real-time weather details from Open-Meteo and exposes them through an API.

#### Features
- Latitude and longitude based weather
- Temperature
- Wind speed
- Wind direction
- Condition description
- Humidity
- Surface pressure
- Cloud cover

#### Security
The code uses Basic Authentication in the API layer for demonstration purposes. In real production, credentials should be stored in environment variables and protected properly.

#### Example Request

```http
GET /weather?lat=22.5726&lon=88.3639
```

#### Example Response

```json
{
  "latitude": 22.57,
  "longitude": 88.36,
  "time": "2026-01-16T12:00",
  "temperature": 28.2,
  "windspeed": 12.4,
  "winddirection": 180,
  "condition": "Partly cloudy",
  "is_day": true,
  "relative_humidity": 72,
  "surface_pressure": 1012.5,
  "cloud_cover": 35,
  "call_count": 12
}
```

#### Why It Matters
Crop yield and health are heavily influenced by weather. Real-time data helps farmers make better decisions.

---

## 6. Technology Stack

### Programming Language
- Python

### Frameworks and Libraries
- FastAPI
- Pydantic
- NumPy
- scikit-learn
- Joblib
- PyTorch
- Torchvision
- Ultralytics YOLO
- PIL (Python Imaging Library)

### Data Science Tools
- Jupyter Notebook
- Machine learning models
- Deep learning pipelines

### External Services
- Open-Meteo API

### Deployment Tools
- Docker
- Render
- GitHub

---

## 7. Folder Structure

```text
Agriscore_Microservices/
├── AGRIYIELDPRO/
│   ├── DATASET/
│   ├── DATA_PROCESSING/
│   ├── EDA/
│   ├── ENCODER-DECODER/
│   ├── MODEL/
│   ├── PROCESSED_DATASET/
│   ├── RESULT/
│   ├── TESTING/
│   ├── TRAINING/
│   ├── main.py
│   ├── start.sh
│   ├── Dockerfile
│   └── yieldenv.txt
│
├── FARMPLANNER/
│   ├── DATAFLOW/
│   ├── DATASET/
│   ├── ENCODER-DECODER/
│   ├── MODELS/
│   ├── PREPROCESS/
│   ├── PROCESS_DATASET/
│   ├── RESULT/
│   ├── TESTING/
│   ├── TRAINING/
│   ├── main.py
│   ├── start.sh
│   ├── Dockerfile
│   ├── requirements.txt
│   └── myenv/
│
├── PLANTCURE/
│   ├── MODELS-PLANTvsOTHERS/
│   ├── MODELS_HealthCheck/
│   ├── MODELS_DISEASE_DETECT/
│   ├── TRAINING_PLANTvsNONPLANT/
│   ├── TRAINING_HEALTHYorUNHEALTHY/
│   ├── TRAINING_DISEASE_DETECT/
│   ├── TESTIMAGE/
│   ├── RESULT/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── image_Augmentation.ipynb
│   └── ...
│
├── WEATHERAPI/
│   ├── main.py
│   └── Dockerfile
│
├── cronjob.py
├── crop.py
├── render.yaml
├── readme.md
├── .gitignore
├── AGROENVr.txt
└── ...
```

---

## 8. How the System Works in Real Life

A common workflow looks like this:

1. A farmer enters field and crop conditions
2. FarmPlanner recommends the best crop to grow
3. AgriYieldPro predicts expected yield
4. The farmer uploads a plant image
5. PlantCure checks whether the plant is healthy or diseased
6. WeatherAPI provides real-time climate information
7. The farmer makes a decision supported by AI insights

This creates a complete smart farming support system.

---

## 9. API Endpoints

### AgriYieldPro

#### Root
```http
GET /
```

#### Predict yield
```http
POST /predict
```

#### Predict yield with form data
```http
POST /predict_form
```

---

### FarmPlanner

#### Root
```http
GET /
```

#### Recommend crop
```http
POST /recommend-json
```

#### Get top 5 crop recommendations
```http
POST /recommend
```

---

### PlantCure

#### Root
```http
GET /
```

#### Predict image
```http
POST /predict/
```

Upload an image using multipart form data.

---

### WeatherAPI

#### Get weather
```http
GET /weather?lat=22.5726&lon=88.3639
```

#### Get API call count
```http
GET /token-count
```

---

## 10. Why This Project Stands Out

This project is a strong example of real-world AI application because it combines:

- agriculture
- machine learning
- deep learning
- API development
- microservice design
- cloud deployment

It does not just show models in isolation; it shows how AI can be used in real agricultural problem-solving.

---

## 11. Challenges and Limitations

Although the project is effective, there are some practical limitations:

1. Hardcoded authentication values in some services
2. Limited frontend or user interface
3. Dependence on data quality and training quality
4. Separate microservices need better orchestration in production
5. More robust production monitoring and logging are still needed

---

## 12. Future Improvements

This project can be extended with:

- A full web dashboard
- User authentication and role-based access
- Database storage for predictions and history
- IoT sensor integration
- Satellite and weather forecasting integration
- Farmer notifications
- Improved model retraining pipeline
- Better deployment automation
- CI/CD pipeline
- Kubernetes or cloud orchestration

---

## 13. Deployment

The repository includes deployment configuration via `render.yaml` and supports cloud deployment of the services.

This makes it possible to deploy each microservice independently and integrate them via API calls.

---

## 14. Why This Project Is Good for Interviews

This project is interview-friendly because it demonstrates:

- real-world problem solving
- machine learning and deep learning usage
- backend service development with FastAPI
- microservice architecture
- usage of AI in agriculture
- deployment and cloud concepts

### Interview summary
> This project is a smart agriculture platform that uses AI to help farmers make better decisions. It predicts crop yield, recommends suitable crops, detects plant diseases from image uploads, and provides weather information. The solution is built using FastAPI, Python, machine learning, and deep learning models, making it a practical real-world AI application.

---

## 15. Final Summary

Agriscore Microservices is an AI-powered agriculture solution designed to improve productivity, reduce uncertainty, and support farmers with intelligent recommendations. It combines several components into one platform:

- yield prediction
- crop recommendation
- plant health analysis
- weather intelligence

This project is a solid example of how technology can improve agricultural decision-making and sustainability.

---

## 16. License

This project is open for educational and learning purposes. Please check the repository and respect any original licensing terms if applicable.

---

## 17. Author

Developed by Avijit Bhadra

---

## 18. Conclusion

This repository is a practical and impactful AI solution for modern agriculture. It demonstrates the use of machine learning and deep learning for solving real-world agricultural problems, and it is a strong project for learning, portfolio building, and interview preparation.

