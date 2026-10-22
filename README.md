# NFL Fantasy ML Optimizer

A reproducible Python project for Full PPR fantasy football projections, weekly lineup optimization, and waiver-wire analysis using historical NFL data and Sleeper's documented read-only API.

## Requirements

- Python 3.11 or newer (Docker uses Python 3.12)
- Git
- Optional: Docker with the Compose plugin

## Local setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env  # Windows: copy .env.example .env
```

Keep `.env` local. Never commit credentials, private league exports, or raw data.

## Repository layout

```text
configs/                  Versioned project configuration
 data/raw/                Immutable source data (ignored by Git)
 data/interim/            Intermediate datasets (ignored by Git)
 data/processed/          Validated model-ready datasets (ignored by Git)
 artifacts/models/        Generated model files (ignored by Git)
 artifacts/metrics/       Generated evaluation output (ignored by Git)
 notebooks/               Exploratory analysis
 src/fantasy_ml/data/     Source ingestion and validation
 src/fantasy_ml/features/ Leakage-safe feature engineering
 src/fantasy_ml/scoring/  Configurable fantasy scoring
 src/fantasy_ml/models/   Baselines and ML models
 src/fantasy_ml/evaluation/Chronological backtests and metrics
 src/fantasy_ml/decisions/Lineup and waiver optimization
 tests/                   Automated tests
 reports/                 Human-readable reports
```

## Development commands

```bash
pytest
ruff check .
ruff format --check .
```

These commands become useful as implementation and tests are added.

## Docker

```bash
docker build -t nfl-fantasy-ml-optimizer .
docker run --rm -p 8501:8501 --env-file .env nfl-fantasy-ml-optimizer
```

The Streamlit entrypoint will be added with the dashboard implementation; the image's default command is not expected to run until then.

## Modeling principles

- Default scoring is Sleeper-style Full PPR; actual league settings should override defaults when available.
- Use chronological, rolling-origin evaluation rather than random train/test splits.
- Features must use only information available before each player's kickoff; historical rolling features must be shifted appropriately.
- Preserve raw source data and record data provenance and coverage.
- Treat projections as estimates, document missing data, and never execute league transactions automatically.

## Current status

Initial repository scaffold only. Data-source coverage and schemas, ingestion, scoring implementation, tests, models, optimizer, and dashboard are not implemented yet.
