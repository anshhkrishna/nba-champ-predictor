# NBA Championship Predictor

Predict how far each NBA team will go in the playoffs — measured in **playoff wins**
(0–16, where 16 = champion) — from regular-season performance, then rank the league to
surface the season's title contenders.

The model learns from every playoff team since 2003. Each team-season is described purely by
its **league rank** in a stat (`1` = best in the NBA that year), so eras with very different
pace and scoring stay comparable.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook predict.ipynb
```

To refresh the data for a new season, bump `CURRENT_SEASON` / `LAST_COMPLETE_SEASON` in
`build_dataset.py` and run:

```bash
python build_dataset.py
```

## How it works

1. **`build_dataset.py`** — a reproducible data pipeline. It scrapes team, opponent, and
   advanced stats from [Basketball-Reference](https://www.basketball-reference.com/), converts
   every stat to a league rank (`1` = best for the team), and attaches each playoff team's
   actual playoff wins (parsed from the playoff bracket). It writes:
   - `past_league_rankings.csv` — training data, every playoff team **2003–2025**
   - `2026_prediction.csv` — the current season, all 30 teams, target left blank
2. **`predict.ipynb`** — loads the data, selects features by correlation with playoff wins,
   trains three regressors (Linear Regression, Random Forest, XGBoost), validates them
   honestly, and predicts the current title race.

## What's modeled

- **Target:** playoff wins (regression, 0–16).
- **Features:** team & opponent per-game stats plus advanced metrics (SRS, ORtg/DRtg, Four
   Factors, etc.), each as a league rank, filtered to `|correlation with playoff wins| > 0.25`.
- **Validation:** *walk-forward* — every season is predicted using only the seasons before it,
   so there is no leakage between teams in the same year. Reported metrics:
  - **MAE** of predicted playoff wins
  - **Champion hit-rate** — how often the model's #1 team actually won the title
  - **Top-3 hit-rate** — how often the eventual champion was in the model's top 3

## Results

### Backtest (walk-forward, scoring seasons 2013–2025)

Each season is predicted using only the seasons that came before it.

| Model | MAE | Champion hit-rate | Top-3 hit-rate |
| --- | --- | --- | --- |
| Linear Regression | 3.12 | 38% | 85% |
| Random Forest | 2.83 | 31% | 85% |
| XGBoost | 2.91 | 31% | 77% |

The model is much better at narrowing the field than at calling the exact winner: the eventual
champion lands in its **top 3 about 85%** of the time, while its single #1 pick wins the title
~31–38% of the time.

### 2025–26 prediction

Trained on all 23 completed seasons (2003–2025), averaged across the three models:

| Rank | Team | Predicted playoff wins |
| --- | --- | --- |
| 1 | **Oklahoma City Thunder** | 11.1 |
| 2 | San Antonio Spurs | 8.7 |
| 3 | Detroit Pistons | 8.5 |
| 4 | Los Angeles Lakers | 8.4 |
| 5 | Denver Nuggets | 7.8 |
| 6 | Boston Celtics | 7.4 |
| 7 | New York Knicks | 7.0 |
| 8 | Cleveland Cavaliers | 5.1 |

**Predicted champion: Oklahoma City Thunder.** Read the *ranking* as the title signal rather
than the raw win count — the regressor pulls every team toward the mean, so no one reaches the
16 wins an actual title requires. Numbers will shift as the season's stats change; re-run
`build_dataset.py` to refresh them.

> Caveat: the model only sees regular-season strength (as league ranks). It knows nothing about
> injuries, matchups, or playoff variance, so treat the output as "who is best positioned."

## Repo layout

| File | Purpose |
| --- | --- |
| `build_dataset.py` | Scrapes Basketball-Reference and regenerates the CSVs |
| `predict.ipynb` | Analysis, modeling, backtest, and current-season prediction |
| `past_league_rankings.csv` | Training data (2003–2025, generated) |
| `2026_prediction.csv` | Current-season feature ranks for prediction (generated) |
| `requirements.txt` | Python dependencies |
| `2020_league_rankings_before_orlando.csv` | Original hand-built file, kept for reference (superseded by the pipeline) |

## Notes & caveats

- Basketball-Reference rate-limits scrapers; `build_dataset.py` waits between requests and
  caches pages under `.cache/`, so re-runs are fast and gentle.
- The model captures *regular-season strength*, not injuries, matchups, or playoff variance —
  treat the output as "who is best positioned," not a guarantee.
