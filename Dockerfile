FROM python:3.12-slim

# 1. Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# 2. Install system dependencies required for compilation and healthchecking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy dependencies and install using pip cache optimization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy source files
COPY . .

# 5. Create storage directories for data persistence and logs
RUN mkdir -p data/raw data/uploads data/processed data/chromadb logs

# 6. Expose port mapping
EXPOSE 8501

# 7. Configure Docker Healthcheck endpoint check using Streamlit's internal health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=8s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 8. Start Streamlit web service
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
