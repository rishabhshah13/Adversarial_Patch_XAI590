# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Run both backend and frontend
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port=8501 --server.enableCORS=false"]
