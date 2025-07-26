# Use Python 3.10.11 for a more specific and smaller image
FROM python:3.10.11-slim-buster

# Create a non-root user and group for security best practices
# 'appuser' is a common name, you can choose another if you prefer
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set the working directory in the container
# This directory will be owned by the non-root user
WORKDIR /app

# Copy the dependency file first to leverage Docker's build cache
# If yieldenv.txt changes, only this layer and subsequent layers rebuild
COPY yieldenv.txt .

# Switch to the non-root user before installing dependencies
# This addresses the "Running pip as the 'root' user" warning
USER appuser

# Install dependencies from yieldenv.txt
# Ensure yieldenv.txt contains uvicorn, fastapi, and all other necessary packages
RUN pip install --no-cache-dir -r yieldenv.txt

# --- IMPORTANT FIX FOR "uvicorn: not found" ---
# Add the user's local bin directory to the PATH
# This ensures executables installed by pip (like uvicorn) are found
ENV PATH="/home/appuser/.local/bin:${PATH}"
# -----------------------------------------------

# Temporarily switch back to root to change ownership of the /app directory
# This is necessary because WORKDIR initially creates /app as root,
# and the non-root user needs write access to copy the rest of the app.
USER root
RUN chown -R appuser:appgroup /app

# Switch back to the non-root user for copying the rest of the application
# and for running the application command
USER appuser

# Copy the rest of the project files into the container
# This copies main.py, your data directories, etc., into /app
COPY . .

# Declare the port the application listens on (Render will map this)
EXPOSE 8000

# Command to start the Uvicorn server
# IMPORTANT: Use the shell form for CMD to allow ${PORT} variable expansion by the shell
# ✅ This correctly expands $PORT using a shell
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
