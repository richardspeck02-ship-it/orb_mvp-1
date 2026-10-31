FROM jnj.artifactrepo.jnj.com/python:3.12.0-slim
RUN mkdir -p /etc/orb-generation /orb-generation

WORKDIR /orb-generation
COPY retrieval_layer/requirements.txt ./
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt
COPY retrieval_layer/src/ ./