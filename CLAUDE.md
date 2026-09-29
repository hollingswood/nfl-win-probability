# NFL Win Probability — rules for Claude

Predicts home-team win probability for NFL games from nflverse data, explains each prediction,
and benchmarks against Vegas. Runs weekly on GitHub Actions and publishes to GitHub Pages.

## Layout
- `src/nflpred/data.py` — downloads nflverse schedules + play-by-play (release parquet files), cached in `data/raw/`
- `src/nflpred/features.py` — all pre-game features. `FEATURES` is the model's input list.
- `src/nflpred/injuries.py` — injury-report feature (Out/Doubtful × prior snap share)
- `scripts/experiment.py` — harness: `run(name, feat_cfg, model_kind, features)` returns validation (2015-19) and holdout (2020-25) log loss. Pick changes on validation; report holdout.
- `src/nflpred/model.py` — `MarginModel` (production: ridge on point margin → win prob), baselines, backtest, explanations in points (`FACTOR_GROUPS`)
- `src/nflpred/pipeline.py` — CLI: `update`, `backtest`, `gate`
- `src/nflpred/travel.py`, `weather.py`, `odds.py`, `ngs.py` — context sources (travel/tz/body clock, Open-Meteo forecast, The Odds API multi-book lines, Next Gen Stats). Shown on picks; NOT model inputs (tested, no gain).
- `src/nflpred/news.py` — live injury/depth-chart news (Sleeper + ESPN public feeds) applied to upcoming games; tested offline with fixtures in tests
- `src/nflpred/qb_availability.py` — sit-probability blend for upcoming games with a hurt listed QB; `overrides.json` for late news
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
6. Keep explanations exact: per-game factor contributions must sum to the predicted margin
   (`test_explanations_sum_to_predicted_margin`). If you switch to a non-linear model, use
   SHAP and keep an equivalent test.

## Goal and how it's judged
The goal is an edge over the betting market. Closing lines are not beaten on this data (see README),
so judge progress by: (1) backtest log loss (CI gate), (2) `scripts/vs_vegas.py` blend test, which must
beat Vegas alone on 2020-2025 to claim new information, and (3) live closing line value in `clv` of
predictions.json. Never report a betting edge from in-sample or tuned-on-holdout results.

## Ideas backlog (good next PRs)
- Clinch-aware final-week feature (actual seeding scenarios) instead of the simple final-week interaction
- Once `history/odds_*.json` has a season of snapshots, measure CLV per book and per day of week
- Live formats checked 2026-09-29 via web fetch: ESPN summary `injuries` structure matches `parse_espn_summary` exactly; Sleeper field names match `parse_sleeper` (saw injury_status values IR/PUP; practice_participation values not yet observed live, confirm on first game-week run)
- QB rating should fade with time off (currently decays per game played only; overrated Keenum in 2026 wk3)
- Preseason priors: roster turnover / draft capital / coaching change for weeks 1-4
- Injury weighting by position (OL/CB clusters, WR1) instead of flat snap share
- Weather/wind for outdoor games; travel distance and time zones
- Already tried without gain (don't repeat without a new angle): garbage-time filter, opponent-adjusted EPA,
  weekly power ratings, recency weighting, QB/Elo parameter grids, gradient boosting,
  travel/time zones/body clock, weather interactions, Next Gen Stats (QB CPOE/TTT, RYOE, separation)
