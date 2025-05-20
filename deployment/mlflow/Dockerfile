
# Use official Python image
FROM python:3.9-slim

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Install dependencies
RUN pip install --no-cache-dir mlflow scikit-learn pandas gunicorn

# Set work directory
WORKDIR /app

# Copy model and script
COPY model /app/model

# Expose port
EXPOSE 1234

# Command to serve the model using MLflow
CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "1234"]
