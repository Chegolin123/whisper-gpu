# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3-pip ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN pip install faster-whisper

WORKDIR /work
COPY transcribe.py bench.py ./

VOLUME ["/models", "/work"]
ENTRYPOINT ["python", "transcribe.py"]
