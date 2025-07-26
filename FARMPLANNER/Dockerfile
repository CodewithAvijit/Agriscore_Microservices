# Use specific Python version
FROM python:3.10.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first for dependency caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy everything inside FARMPLANNER (Dockerfile is assumed here)
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
