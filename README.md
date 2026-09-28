# NFL Win Probability Board

Predicts each NFL game's win probability, explains the top factors behind every pick, and
scores itself against Vegas. Retrains and republishes automatically every week.

## Results (backtest 2018–2025, 2,219 games, each season predicted by a model trained only on earlier seasons)

| Model | Log loss | Brier | Picks correct |
|---|---|---|---|
| **Margin model (production, v3)** | **0.622** | **0.217** | **66.0%** |
| Logistic regression (v3 features) | 0.624 | 0.217 | 66.5% |
| Gradient boosting (calibrated) | 0.633 | 0.222 | 64.0% |
| v1 logistic (original features) | 0.628 | 0.219 | 65.5% |
| Vegas moneyline (no-vig) | 0.609 | 0.211 | 66.4% |
| Always pick home team | 0.690 | 0.248 | 54.4% |

The production model predicts the home team's point margin with ridge regression, then converts it to a
win probability (normal distribution, σ ≈ 13 points). Explanations are in points of margin.

### What changed in v2 (tuned on 2015–2019, confirmed on 2020–2025 holdout)
| Change | Validation LL | Holdout LL |
|---|---|---|
| v1 baseline | 0.6261 | 0.6295 |
| + QB change (starter vs. team's recent QBs) | 0.6246 | 0.6273 |
| + injuries (Out/Doubtful × recent snap share) | 0.6255 | 0.6258 |
| + predict margin instead of win/loss (v2) | 0.6219 | 0.6242 |
| + final-week rest interactions (v3) | **0.6205** | **0.6228** |

Tried and rejected (no consistent gain): garbage-time play filter, opponent-adjusted EPA,
weekly opponent-adjusted power ratings (`ratings.py`, off by default), recency-weighted training,
QB-rating and Elo parameter tuning, gradient boosting.

## Does it beat Vegas? (`scripts/vs_vegas.py` → `output/vs_vegas.json`)
No, not against closing lines. On the 2020–2025 holdout (1,688 games): Vegas 0.607 log loss, model 0.623,
model + Vegas blend (fit on 2015–2019) 0.610. Betting 1 unit whenever the model saw positive expected
value at the actual moneyline lost money at every edge threshold (−3% to −9% ROI).

The pipeline now measures **closing line value** instead: every run saves the line at publish time in
`history/`, and `clv_report` checks whether the market later moved toward the model's side. Positive
CLV over a few hundred picks is the standard evidence of a real edge against earlier (softer) lines.

## Features (all computed from games before kickoff)
Elo rating · starting QB EPA/dropback (shrunk toward replacement level, decays over ~20 games) ·
offensive and defensive EPA/play and success rate · passing and rushing EPA · turnover margin ·
recent point differential · starter vs. team's usual QB · injuries (non-QB players Out/Doubtful,
weighted by recent snap share) · rest days · home field / neutral site · division game.
Team stats are exponentially weighted (8-game half-life) and carry across seasons.

## Run it
```
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python -m nflpred.pipeline update && python scripts/build_dashboard.py
open site/index.html
```

## Automation
| Workflow | When | What |
|---|---|---|
| `ci.yml` | every push / PR | tests (incl. leakage) + backtest regression gate |
| `weekly.yml` | Tue, Thu, Sun | refresh data → retrain → predict → commit history → deploy GitHub Pages |
| `claude.yml` | PRs and `@claude` mentions | Claude reviews PRs against `CLAUDE.md`; `@claude` in an issue implements it |

### One-time setup
1. Create a GitHub repo and push this folder.
2. Settings → Pages → Source: **GitHub Actions**.
3. Settings → Actions → General → Workflow permissions: **Read and write**.
4. For the Claude workflow: in Claude Code run `/install-github-app`, or add an `ANTHROPIC_API_KEY` repo secret.
5. Run **Weekly predictions** once from the Actions tab (workflow_dispatch) to publish the first board.

Experiments: `PYTHONPATH=src:scripts python -c "from experiment import run; ..."` (see `scripts/experiment.py`).

Data: [nflverse](https://github.com/nflverse/nflverse-data). Not betting advice.
