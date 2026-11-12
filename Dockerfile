FROM python:3.11.14-slim-bookworm AS builder
WORKDIR /build
COPY pyproject.toml __init__.py ./
COPY yolo/ yolo/
COPY shared/ shared/
COPY dfine/ dfine/
COPY docs/ docs/
COPY requirements/ requirements/
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.11.14-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EFFICIENT_DETECTION_RUNTIME=docker
COPY requirements/evaluation.lock.txt /tmp/evaluation.lock.txt
RUN python -m pip install --no-cache-dir -r /tmp/evaluation.lock.txt
COPY --from=builder /wheels/ /wheels/
RUN python -m pip install --no-cache-dir --no-deps /wheels/*.whl \
    && rm -rf /wheels /tmp/evaluation.lock.txt
ARG MODEL=models/baseline_640.onnx
ARG MODEL_METADATA=models/baseline_640.json
COPY ${MODEL} /app/models/model.onnx
COPY ${MODEL_METADATA} /app/models/model.json
RUN useradd --create-home --uid 10001 runner \
    && mkdir /output && chown runner:runner /output
USER runner
ENTRYPOINT ["python", "-m", "efficient_detection.yolo.evaluate_yolo"]
CMD ["--model", "/app/models/model.onnx", "--images", "/data/images", "--annotations", "/data/test.json", "--output", "/output", "--threads", "4"]
