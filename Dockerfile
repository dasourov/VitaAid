# Use an official Python runtime as a parent image
# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Set environment variable for Python and pip to ensure executables are found
ENV PATH="/usr/local/bin:${PATH}"

# Install build-essential and other dependencies needed for compiling certain Python packages
# apt-get update: Updates package lists
# build-essential: Provides essential build tools (gcc, g++, make, libc6-dev, etc.)
# libpq-dev: If you ever use PostgreSQL, this is needed
# libffi-dev, libssl-dev: often needed for cryptography-related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . .

# Expose the port that your application listens on
EXPOSE 8000

# Command to run the application when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-w", "4", "main:app"]