FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config ./config
COPY schemas ./schemas
COPY memory ./memory
COPY agents ./agents
COPY risk ./risk
COPY tools ./tools
COPY orchestrator ./orchestrator
COPY api ./api
COPY tests ./tests
COPY rl_design ./rl_design
COPY main.py ./

RUN pip install -U pip && pip install -e .

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/system/status || exit 1

EXPOSE 8000

CMD ["python", "main.py"]
