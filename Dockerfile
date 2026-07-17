# VSPG core image — classic control, MiniGrid, QHD, sgd_ablation.
# Everything except SustainGym (see docker/Dockerfile.sustaingym, a separate
# image: SustainGym's own conda-only build deps don't mix cleanly with this
# plain pip/CUDA image).
#
# Build:  docker compose build
# Run:    docker compose up -d
# Shell:  docker exec -it vspg_core bash

FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-distutils \
        git \
        build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY . /workspace/

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

CMD ["sleep", "infinity"]
