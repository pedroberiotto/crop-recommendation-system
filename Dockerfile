FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends binutils libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --prefix=/install -r requirements-api.txt

RUN PYDIR="/install/lib/python3.11/site-packages" && \
    find "$PYDIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find "$PYDIR" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find "$PYDIR" -type d -name "test" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find "$PYDIR" -name "*.pyc" -delete && \
    find "$PYDIR" -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    rm -rf "$PYDIR/numba/cuda" \
           "$PYDIR/numba/roc" \
           "$PYDIR/sklearn/datasets/data" \
           "$PYDIR/sklearn/datasets/images" \
           "$PYDIR/sklearn/datasets/descr" \
           "$PYDIR/matplotlib/mpl-data/sample_data" \
           "$PYDIR/scipy/misc" \
           "$PYDIR/shap/datasets" \
           "$PYDIR/shap/benchmark" \
           2>/dev/null || true && \
    du -sh "$PYDIR" && \
    du -sh "$PYDIR"/* | sort -h | tail -15

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

LABEL maintainer="crop-recommendation-system"
LABEL description="Crop Recommendation API"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* /root/.cache

COPY --from=builder /install /usr/local

COPY src/           src/
COPY app/           app/
COPY pyproject.toml .

RUN pip install --no-deps --no-cache-dir -e . && \
    find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    mkdir -p /artifacts /logs && \
    useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app /artifacts /logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
