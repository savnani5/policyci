FROM ros:jazzy-ros-base@sha256:da725acf8b0f9f30c683e33ffbdcd6482d077af96d6fdc7688c5f4f280b7d923

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3-pip \
       ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --break-system-packages --no-cache-dir --ignore-installed \
      "numpy>=2" \
      "psutil>=7"
RUN python3 -m pip install --break-system-packages --no-cache-dir -e ".[visualize,aloha-demo]"

COPY suites ./suites
COPY subjects ./subjects
COPY examples ./examples

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "policyci.cli", "--help"]
