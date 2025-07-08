#!/bin/bash

# Activate the virtual environment
source ../AGROENV/Scripts/activate

# Run the FastAPI app
uvicorn main:app --host 127.0.0.1 --port 10000
