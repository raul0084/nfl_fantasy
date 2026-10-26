
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY . .

RUN python -m pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "src/fantasy_ml/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
