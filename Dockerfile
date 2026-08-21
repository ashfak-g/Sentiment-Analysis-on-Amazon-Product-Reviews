# Production Dockerfile for Amazon Sentiment Analysis System
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Run NLTK download script
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"

# Run pipeline if models are missing
RUN python -m src.pipeline

EXPOSE 8501 8000

# Default command launches FastAPI backend server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
