# syntax=docker/dockerfile:1.2
FROM python:3.11.10-slim

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc libssl-dev

# Define environment variable PORT for GCP
ENV PORT 8080

# Copy requirements files for the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the challenge folder project
COPY . .

# Run the application inside the container
CMD uvicorn challenge.api:app --host 0.0.0.0 --port $PORT --reload
