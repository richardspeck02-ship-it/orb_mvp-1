FROM jnj.artifactrepo.jnj.com/python:3.12.0-slim
RUN apt-get update -y && apt-get install -y --no-install-recommends curl libgraphite2-3 fontconfig fonts-noto-core fonts-noto-color-emoji fonts-noto-extrahinted \
    && fc-cache -f -v
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv ./tectonic /usr/local/bin
RUN mkdir -p /etc/orb-generation /orb-generation

WORKDIR /orb-generation
COPY generation_layer/requirements.txt ./
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt
COPY generation_layer/src/ ./