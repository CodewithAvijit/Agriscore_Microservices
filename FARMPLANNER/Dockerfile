# Use specific Python version
FROM python:3.10.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first for dependency caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (just for documentation)
EXPOSE 8000

# Start FastAPI using the port Render provides via $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
