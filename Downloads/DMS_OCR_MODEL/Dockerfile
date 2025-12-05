# Use Python 3.11 bookworm (Debian 12) for better package availability
FROM python:3.11-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies including Java, Tesseract, Ghostscript, and GTK
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    default-jre-headless \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5.3.3/tessdata
ENV PATH="${PATH}:/usr/bin/tesseract"
ENV TIKA_SERVER_JAR="/tika-server.jar"
ENV TIKA_SERVER_ENDPOINT="http://localhost:9998/"
ENV TESSERACT_PATH="/usr/bin/tesseract"

# Azure OpenAI Configuration - Use environment variables or Docker secrets in production
# These should be passed at runtime using --env-file or Docker secrets
# Example: docker run --env-file .env dms-ocr-local

# Install Tika server
RUN wget https://archive.apache.org/dist/tika/2.9.0/tika-server-standard-2.9.0.jar -O /tika-server.jar

# Copy requirements first to leverage Docker cache
COPY OCR/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose the port the app runs on
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["sh", "-c", "java -jar /tika-server.jar & python -c 'from tika import initVM; initVM()' && uvicorn OCR.src.backend.server:app --host 0.0.0.0 --port 8000"]
