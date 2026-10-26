# NFL Fantasy ML Optimizer

Trying to up my fantasy game. 

## Features

- Weekly player point projections
- Fantasy lineup optimization
- Waiver-wire analysis
- Historical model evaluation
- Sleeper API integration
- Streamlit application

## Run Locally

```bash
git clone <repository-url>
cd nfl_fantasy
pip install .
streamlit run src/fantasy_ml/app.py
```

## Run with Docker

```bash
./run.sh
```

Open [http://localhost:8501](http://localhost:8501).

## Data

Historical NFL statistics are sourced from [nflverse](https://github.com/nflverse/nflverse-data). Sleeper integration uses the [Sleeper API](https://docs.sleeper.com/).