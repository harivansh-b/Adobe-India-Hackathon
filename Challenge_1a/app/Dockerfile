FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY main.py .

RUN pip install pymupdf

CMD ["python", "main.py"]
