FROM python:3.10-slim

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy application code
COPY . .

# Explicitly copy model directories
COPY MODELS_DISEASE_DETECT MODELS_DISEASE_DETECT
COPY MODELS_HealthCheck MODELS_HealthCheck
COPY MODELS-PLANTvsOTHERS MODELS-PLANTvsOTHERS

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port (Render will map $PORT automatically)
EXPOSE 8000

# Run FastAPI app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
