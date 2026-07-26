FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and to ensure stdout/stderr is unbuffered
ENV PYTHONDONTWRITEBYTECODE = 1
ENV PYTHONUNBUFFERED = 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the project
COPY . .

EXPOSE 8501

# Streamlit needs to bind to 0.0.0.0 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]