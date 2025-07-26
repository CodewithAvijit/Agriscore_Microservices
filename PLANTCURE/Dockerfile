FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy application code
COPY . .

# Explicitly copy model directories
COPY MODELS_DISEASE_DETECT MODELS_DISEASE_DETECT
COPY MODELS_HealthCheck MODELS_HealthCheck
COPY MODELS-PLANTvsOTHERS MODELS-PLANTvsOTHERS

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
