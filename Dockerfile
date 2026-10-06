FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}"

WORKDIR /app

# Bake the source revision into the image so /health can report which commit is running.
ARG GIT_SHA=unknown
ENV LEADSCOUT_BUILD=${GIT_SHA}

# A fixed unprivileged identity limits the impact of an application compromise.
RUN useradd --create-home --uid 10001 app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app leadscout/ ./leadscout/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app samples/ ./samples/
RUN mkdir -p /app/out && chown app:app /app/out

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "leadscout.api:app", "--host", "0.0.0.0", "--port", "8000"]
