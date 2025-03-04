# Use an official Python runtime as a parent image (Python 3.10-slim)
FROM python:3.10-slim

# Install system dependencies including build tools (gcc is needed for building some packages)
RUN apt-get update && apt-get install -y \
    swig \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the entire project into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r src/requirements.txt
RUN pip install torch==2.6.0
RUN pip install "gymnasium[box2d]==1.1.0"
RUN pip install pygame

# Change working directory to src so that relative paths (../results/...) resolve correctly
WORKDIR /app/src

# Run the main script
CMD ["python", "main.py"]
