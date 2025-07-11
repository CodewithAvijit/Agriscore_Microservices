# Use Python 3.10.11 for a more specific and smaller image
FROM python:3.10.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies from yieldenv.txt
# Ensure yieldenv.txt contains uvicorn, fastapi, and all other necessary packages
RUN pip install --no-cache-dir -r yieldenv.txt

# Declare the port the application listens on (Render will map this)
EXPOSE 8000

# Command to start the Uvicorn server, binding to 0.0.0.0 and the Render-provided $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]