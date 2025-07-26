FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF, pdfplumber, and spaCy
RUN apt-get update && \
    apt-get install -y gcc build-essential libpoppler-cpp-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (create this file in your project root)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the processing script and any other needed files
COPY process_pdfs.py .
COPY app ./app

CMD ["python", "process_pdfs.py"]