FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies first for better build-cache reuse.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip && \
    python -m pip install .

# The app will be added later; default to the interactive dashboard entrypoint.
EXPOSE 8501
CMD ["streamlit", "run", "src/fantasy_ml/app.py", "--server.address=0.0.0.0"]
