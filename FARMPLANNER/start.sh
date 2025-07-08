#!/bin/bash

# Activate the virtual environment
source ../agroenv/Scripts/activate

# Run the FastAPI app
uvicorn main:app --host 0.0.0.0 --port $PORT
