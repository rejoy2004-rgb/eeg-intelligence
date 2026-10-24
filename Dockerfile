# UNVERIFIED: Docker is not installed on the development machine, so this image has not been built or run.
# Build:  docker build -t eeg-intelligence .
# Run:    docker run --rm -p 8501:8501 -v "$(pwd)/data/raw:/app/data/raw:ro" -v "$(pwd)/models:/app/models" -v "$(pwd)/outputs:/app/outputs" eeg-intelligence
# The EEG data are NOT baked into the image (mount them read-only as above; see docs/DATASET.md).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA wheels), then the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && grep -v -E "^(torch|pytest)" requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY configs ./configs
COPY src ./src
COPY app.py pyproject.toml README.md ./

# Run as an unprivileged user
RUN useradd --create-home appuser && mkdir -p data/raw models outputs && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
