# Use official Python image
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY app.py .
COPY src/ src/
COPY models/saved_ner_model/ models/saved_ner_model/

RUN python --version
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]