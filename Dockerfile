# syntax=docker/dockerfile:1
# Stage 1: Build the project wheel
FROM python:3.11.14-slim-bookworm AS builder
WORKDIR /build
COPY pyproject.toml __init__.py ./
COPY yolo/ yolo/
COPY shared/ shared/
COPY dfine/ dfine/
COPY docs/ docs/
COPY requirements/ requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --no-deps --wheel-dir /wheels .

# Stage 2: Runtime image
FROM python:3.11.14-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EFFICIENT_DETECTION_RUNTIME=docker

# Leverage Docker cache: install dependencies first before code
COPY requirements/evaluation.lock.txt /tmp/evaluation.lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/evaluation.lock.txt \
    && rm -f /tmp/evaluation.lock.txt

# Install the application wheel
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    python -m pip install --no-cache-dir --no-deps /wheels/*.whl

# Model arguments (defaulting to baseline or best quantized model)
RUN useradd --create-home --uid 10001 runner \
    && mkdir /output && chown runner:runner /output
ARG MODEL=models/baseline_640.onnx
ARG MODEL_METADATA=models/baseline_640.json
COPY ${MODEL} /app/models/model.onnx
COPY ${MODEL_METADATA} /app/models/model.json

USER runner
ENTRYPOINT ["python", "-m", "efficient_detection.yolo.evaluate_yolo"]
CMD ["--model", "/app/models/model.onnx", "--images", "/data/images", "--annotations", "/data/test.json", "--output", "/output", "--threads", "4"]

FROM runtime AS openvino
USER root
COPY requirements/openvino.lock.txt /tmp/openvino.lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/openvino.lock.txt \
    && rm /tmp/openvino.lock.txt
USER runner
CMD ["--model", "/app/models/model.onnx", "--images", "/data/images", "--annotations", "/data/test.json", "--output", "/output", "--threads", "4", "--backend", "openvino"]

# Keep the default image free of the optional experiment dependencies.
FROM runtime AS default
