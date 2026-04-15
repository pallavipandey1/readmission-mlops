# Base image — Python 3.11 on slim Linux (small footprint)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# so installs are skipped if requirements haven't changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY src/ ./src/
COPY model_artifacts/ ./model_artifacts/
COPY setup.py .

# Install the package so imports work
RUN pip install -e .

# Tell Docker which port the server listens on
EXPOSE 8000

# Command that runs when container starts
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]