
# Use official Python image
FROM python:3.13-slim

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Set work directory
WORKDIR /app

# Copy model and script
COPY model /app/model

# Install dependencies
RUN pip install --upgrade pip & pip install -r /app/model/requirements.txt

# Expose port
EXPOSE 1234

# Command to serve the model using MLflow
CMD ["mlflow", "models", "serve", "-m", "model", "--no-conda", "-h", "0.0.0.0", "-p", "1234"]
