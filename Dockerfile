FROM python:3.12-slim

WORKDIR /app

# Atualiza os pacotes e instala as dependências do sistema,
# incluindo o Tesseract (OCR), bibliotecas gráficas (libgl) e o poppler-utils para PDFs.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501