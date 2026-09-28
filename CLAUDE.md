# NFL Win Probability — rules for Claude

Predicts home-team win probability for NFL games from nflverse data, explains each prediction,
and benchmarks against Vegas. Runs weekly on GitHub Actions and publishes to GitHub Pages.

## Layout
- `src/nflpred/data.py` — downloads nflverse schedules + play-by-play (release parquet files), cached in `data/raw/`
- `src/nflpred/features.py` — all pre-game features. `FEATURES` is the model's input list.
- `src/nflpred/model.py` — models, season-forward backtest, explanations (`FACTOR_GROUPS`)
- `src/nflpred/pipeline.py` — CLI: `update`, `backtest`, `gate`
- `scripts/build_dashboard.py` — renders `output/*.json` into `site/`
- `model_baseline.json` — backtest log loss that CI must not regress past

## Commands
```
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q                                   # includes leakage tests
python -m nflpred.pipeline gate             # backtest 2018-2025 + regression gate
python -m nflpred.pipeline update           # refresh data, predict next ~9 days
python scripts/build_dashboard.py
```

## Non-negotiable rules
1. **No leakage.** A feature for a game may only use results from games with an earlier `gameday`.
   Rolling stats must `shift(1)` before aggregating. `test_no_future_leakage` erases all results
   from a cutoff date onward and asserts features on that date don't change; every new feature
   must pass it. Never weaken or skip that test.
2. **Time-based validation only.** No random train/test splits, no k-fold across seasons. Tune on
   2015–2019; 2020–2025 is the holdout reported in the dashboard.
3. **Vegas lines are a benchmark, not a feature** in the main model. A separate "model + market"
   variant is fine if it's clearly labeled and scored separately.
4. **Judge models by log loss and Brier score**, not accuracy. Report next to Vegas.
5. If a change improves the backtest, update `model_baseline.json` in the same PR and say why in the
   PR description. If it's worse by more than 0.002 log loss, CI fails; don't raise the tolerance.
6. Keep explanations exact: per-game factor contributions plus intercept must sum to the model's
   log-odds (`test_explanations_sum_to_model_logit`). If you switch to a non-linear model, use
   SHAP and keep an equivalent test.

## Ideas backlog (good next PRs)
- Opponent-adjusted EPA (adjust each game's EPA by opponent's defensive EPA)
- Injury data (`nflverse-data` releases tag `injuries`) for non-QB starters
- Weather/wind for outdoor games; travel distance and time zones
- Point-spread regression model → convert to probability
- Model + market blend as a separate labeled output
