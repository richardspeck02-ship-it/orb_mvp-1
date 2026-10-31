FROM python:3.12.0-slim
RUN mkdir -p /etc/orb-generation /orb-generation
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils ffmpeg libsm6 libxext6
WORKDIR /orb-generation
COPY embedding_layer/requirements.txt ./
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt
COPY embedding_layer/src/pre_download_yolox.py embedding_layer/src/dummy.pdf ./
RUN python3 pre_download_yolox.py
RUN pip install gradio
COPY embedding_layer/src/ ./