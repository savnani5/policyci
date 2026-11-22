FROM python:3.11.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       build-essential \
       libgl1 \
       libglew2.2 \
       libglfw3 \
       libosmesa6 \
       libosmesa6-dev \
       linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e . \
    && python -m pip install --no-cache-dir \
       robosuite==1.5.2 \
       mujoco==3.7.0 \
       imageio==2.37.0 \
       imageio-ffmpeg==0.6.0

COPY suites ./suites
COPY subjects ./subjects
COPY examples ./examples

CMD ["python", "-m", "policyci.cli", "--help"]
