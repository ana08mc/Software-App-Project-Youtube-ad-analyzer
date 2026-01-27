FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data && \
    wget -q -O /app/data/haarcascade_frontalface_default.xml \
      https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN python -m nltk.downloader vader_lexicon stopwords

ENV TESSERACT_CMD=/usr/bin/tesseract
ENV CASCADE_PATH=/app/data/haarcascade_frontalface_default.xml

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
