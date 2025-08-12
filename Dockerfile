FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY rag/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY rag/ ./rag/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: run both FastAPI and Streamlit
CMD uvicorn rag.main:app --host 0.0.0.0 --port 8000 & streamlit run rag/app.py --server.port 8501 --server.address 0.0.0.0