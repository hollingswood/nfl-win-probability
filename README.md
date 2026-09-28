# NFL Win Probability Board

Predicts each NFL game's win probability, explains the top factors behind every pick, and
scores itself against Vegas. Retrains and republishes automatically every week.

## Results (backtest 2018–2025, 2,219 games, each season predicted by a model trained only on earlier seasons)

| Model | Log loss | Brier | Picks correct |
|---|---|---|---|
| Logistic regression (production) | 0.628 | 0.219 | 65.5% |
| Gradient boosting (calibrated) | 0.635 | 0.222 | 63.6% |
| Vegas moneyline (no-vig) | 0.609 | 0.211 | 66.4% |
| Always pick home team | 0.690 | 0.248 | 54.4% |

Logistic regression beat gradient boosting, so it runs in production. It also makes per-game explanations exact.

## Features (all computed from games before kickoff)
Elo rating · starting QB EPA/dropback (shrunk toward replacement level, decays over ~20 games) ·
offensive and defensive EPA/play and success rate · passing and rushing EPA · turnover margin ·
recent point differential · rest days · home field / neutral site · division game.
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

Data: [nflverse](https://github.com/nflverse/nflverse-data). Not betting advice.
