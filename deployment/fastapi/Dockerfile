# Use official Python image
FROM python:3.13-slim

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Set work directory
WORKDIR /app

# Copy model and script
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip & pip install fastapi uvicorn scikit-learn numpy joblib

# Expose port
EXPOSE 8000

# Command to serve the model using MLflow
CMD ["uvicorn", "serve_model_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]