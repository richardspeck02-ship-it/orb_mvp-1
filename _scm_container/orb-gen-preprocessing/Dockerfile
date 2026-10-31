FROM jnj.artifactrepo.jnj.com/python:3.12.0-slim

RUN mkdir -p /etc/orb-generation /orb-generation /temp/orb-artifacts

WORKDIR /orb-generation
COPY preprocessing_layer/requirements.txt ./
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt
COPY preprocessing_layer/src/ ./
ENTRYPOINT ["python3", "./main.py"]