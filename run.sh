
#!/usr/bin/env bash
set -Eeuo pipefail

# Always run from the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not on PATH." >&2
    echo "Install Docker Desktop, start it, and try again." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose is unavailable." >&2
    exit 1
fi

# Keep persistent directories available on the host.
mkdir -p data artifacts/predictions

if [[ ! -f data/fantasy.duckdb ]]; then
    echo "Warning: data/fantasy.duckdb was not found."
    echo "The app may require database initialization before it can run."
fi

echo "Building and starting NFL Fantasy ML Optimizer..."
echo "When ready, open http://localhost:8501"
echo "Press Ctrl+C to stop."

docker compose up --build
