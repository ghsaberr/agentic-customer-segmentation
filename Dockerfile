# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app/

ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
