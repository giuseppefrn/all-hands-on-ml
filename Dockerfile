# Base image for runtime
FROM python:3.13-slim AS base

WORKDIR /app

# Copy the rest of the application files to the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN echo 'export PYTHONPATH="/app:$PYTHONPATH"' >> ~/.bashrc
