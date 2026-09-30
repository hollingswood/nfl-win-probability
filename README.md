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
| + final-week rest interactions (v3) | 0.6205 | 0.6228 |
| + blowout cap: single-game EPA ±0.3/play, point diff ±21 (v4) | 0.6206 | 0.6214 |
| + team coming off an overtime game (v5) | **0.6191** | **0.6208** |

Also tested: a "starting QB listed Questionable/Doubtful" feature (only ~150 games in history, the two
periods disagreed: not adopted as a feature). Instead, upcoming games use a **QB availability blend**:

| Starting QB status (final practice) | Historically started |
|---|---|
| Questionable (full) | 86% |
| Questionable (limited) | 53% |
| Questionable (did not practice) | 42% |
| Doubtful | ~5% |

P(win) = P(plays) × P(win with him) + (1 − P(plays)) × P(win with a replacement-level QB).
Late news the report hasn't caught: add `overrides.json`, e.g.
`{"2026_03_PHI_CHI": {"home_qb_play_prob": 0.4, "note": "no practice all week"}}`.

Tried and rejected (no consistent gain): garbage-time play filter, opponent-adjusted EPA,
weekly opponent-adjusted power ratings (`ratings.py`, off by default), recency-weighted training,
QB-rating and Elo parameter tuning, gradient boosting.

## Automatic injury and depth-chart news (`news.py`)
Every run before the games pulls free public feeds and applies them to upcoming games:

| Source | What we use | When |
|---|---|---|
| Sleeper API (`api.sleeper.app/v1/players/nfl`) | Injury status, practice participation, depth-chart order for every player | Daily run + Friday final-report run (Sleeper asks apps to pull this list at most daily) |
| ESPN game summaries (unofficial public endpoint) | Per-game injury lists | Every run, including ~75 min before each kickoff window |

- Live statuses replace the nflverse report for that week, so the injury feature and QB-availability blend use the newest information.
- If the listed starting QB is Out, the highest healthy QB on the depth chart becomes the starter.
- If a starter is Questionable/Doubtful, the "sits" scenario uses the actual backup's rating instead of a generic replacement level.
- Every change is listed on the dashboard ("Latest news applied").
- Precedence: `overrides.json` (manual) > live feeds > nflverse report. Feeds failing never block a run.
- Yahoo isn't used: its API needs a per-user OAuth login.

## Spreads and key numbers (`margins.py`, `spread_bets.py`, rules in `spread_rules.json`)
- **Key numbers:** margins are priced with a normal curve re-weighted by how often NFL games actually end on each
  margin (fit on 2012–2014): 3 is ~2.3× more common than a smooth curve says, 7 ~1.6×, 10 ~1.4×, ties ~0.2×.
  Better spread-cover log loss on both 2015–19 and 2020–25, and push rates on whole-number lines match reality
  (predicted 4.5% vs actual 4.3%). Moneyline win probabilities keep the smooth curve (key numbers didn't help there).
- **Spread margin:** 0.33 × model margin + 0.74 × market margin − 0.57 (fit on 2015–2019), σ = 12.8.
- **Backtest 2020–2025 at real closing spreads and prices, 3% edge threshold: 556 bets, +2.5% ROI** — the first
  positive result, but the noise band is about ±4%, so it is being paper-tested live. (Same strategy with a smooth
  curve instead of key numbers: −6.2%. Model alone: −1.7%.)
- **Line shopping:** every licensed book's actual spread and price is evaluated, so −3 at one book vs −3.5 at another
  is a real, priced half-point. Buying extra points is shown as an *estimate* only (typical cost 10¢, 20¢ onto/off 3,
  15¢ onto/off 7; the free odds plan has no alternate-line prices) and is never paper-bet.
- **Books:** `my_books.json` lists the sportsbooks you can actually use (default: licensed US books). Best prices and
  paper bets use only those; the market consensus uses every book.

## Grades (`grading.py`, version 1)
Every opportunity gets a grade from A+ to C, defined before any paper results: +1/+2 for edge (capped, because the
biggest model-vs-market gaps were not the best bets), −1 if the model is far from the market, ±1 if the line has
moved toward/against us since first seen, ±1 for the right/wrong side of 3 or 7 (spreads), −1 for a QB change.
Spread backtest 2020–25 by grade (no line-movement data historically): A +15.2% (47 bets), B+ +0.2% (168),
B +2.1% (226), C+ +1.8% (101), C −1.4% (14): ordered at the extremes, flat in the middle, small samples.
The paper record reports performance by grade for each track; that is the real test of whether grades mean anything.

## Paper bets → recommendations (`bets.py`, rules in `betting_rules.json`)
Two independent tracks with their own locked rules and records: **moneyline** (`betting_rules.json`, v1) and
**spread** (`spread_rules.json`, v1, same thresholds, sizing and validation test). A track switches to live
recommendations on its own record only. Historically the moneyline approach lost money at closing prices in every
variant tried (including moneylines derived from the market-anchored spread margin: −5.8%), so it serves as a control.

The model does not recommend bets until it has proven itself on bets logged in real time.

- **Shadow mode (default):** every run checks each upcoming game. A moneyline side qualifies when, using a
  model + market blend (weights fit on 2015–2019 only), expected value at the **best available price** is ≥ 3%,
  the odds are between −400 and +400, both starting QBs are confirmed, the injury report is out, and the line
  hasn't moved 2+ points against that side since we first saw it. It is logged once, at that price and book,
  in `history/paper_bets.json`, and never changed.
- **Stakes:** quarter-Kelly, 0.25–2 units per bet (1 unit = 1% of bankroll), max 8 units per week (biggest edges first).
- **Grading:** after each game: win/loss, profit in units, and closing line value (our price vs. the closing fair odds).
- **Switch to live (pre-registered, version 1):** ≥ 200 graded bets **and** average CLV > 0 with p < 0.05 **and**
  positive ROI. Then the dashboard shows "Recommended bets" and each new qualifying bet opens a GitHub issue
  (you get an email / GitHub app notification).
- Changing any rule starts a new version with a fresh record. Thresholds are never tuned to results.

## Context sources (shown on each pick, not model inputs)
| Source | What | Where it runs |
|---|---|---|
| The Odds API | Lines from every US sportsbook: consensus no-vig probability and spread, best moneyline per side and which book has it, model EV at that price. Saved to `history/odds_*.json` for CLV. | Weekly workflow, needs `ODDS_API_KEY` secret |
| Open-Meteo | Kickoff-hour temperature, wind, gusts, rain chance for outdoor games | Weekly workflow, no key |
| Travel (`travel.py`) | Distance, time zones crossed, kickoff time on each team's body clock | Everywhere |

### Candidate features tested 2026-09-29 (`scripts/extra_features.py`)
Rule: adopt only if log loss improves on BOTH 2015–2019 validation and 2020–2025 holdout (baseline 0.6203 / 0.6215).

| Candidate | Validation | Holdout |
|---|---|---|
| Special teams net EPA (kickoffs, punts, FGs, XPs) | −0.0003 | +0.0010 |
| Starters out by position group (OL, secondary, skill, front) | +0.0009 | +0.0006 |
| 2+ offensive-line starters out | +0.0008 | +0.0002 |
| QB prior by draft capital (4 buckets) | −0.0004 | −0.0004 |
| …same, simplified to 1st round vs. rest (robustness check) | −0.0006 | +0.0001 |
| QB rating fades with time off (1-year half-life) | −0.0001 | +0.0002 |
| New head coach | −0.0002 | +0.0003 |
| Early-season (weeks 1–4) interactions | −0.0002 | +0.0011 |
| Referee home-margin tendency | +0.0002 | +0.0008 |
| Turf mismatch (visitor's home surface ≠ venue) | −0.0018 | +0.0007 |
| Conference game | +0.0000 | −0.0004 |
| Thursday / Monday game | −0.0000 | +0.0002 |
| Home stand / second straight road game | +0.0006 | +0.0000 |
| Off a bye (incl. playoff bye) | −0.0000 | +0.0001 |
| Bounce-back after 17+ pt loss / letdown after 17+ pt win | +0.0003 | +0.0001 |
| **Coming off an overtime game — ADOPTED** | **−0.0012** | **−0.0007** |

The draft-capital prior technically passed, but its table rests on 25 QBs and the simpler version failed the
holdout, so it was treated as noise. The market already prices all of these.

Tested as model inputs on 2015–2019 / 2020–2025 and **rejected** (changes within ±0.002 log loss, inconsistent direction):
travel distance and time zones, body-clock kickoff, weather (wind × passing, cold × dome teams), Next Gen Stats
(QB CPOE and time to throw, team rushing yards over expected, receiver separation). The market prices these already.

Not testable: Super Bowl winner/loser next game (~2 games a year). Favorites-early/dogs-late will be measured from our own odds snapshots once enough accumulate.

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
| `weekly.yml` | Daily 7am AZ, Fri 1:30pm AZ (final injury report), ~75 min before each kickoff window | refresh data + news + odds → retrain → predict → commit history → deploy GitHub Pages |
| `claude.yml` | PRs and `@claude` mentions | Claude reviews PRs against `CLAUDE.md`; `@claude` in an issue implements it |

### One-time setup
1. Create a GitHub repo and push this folder.
2. Settings → Pages → Source: **GitHub Actions**.
3. Settings → Actions → General → Workflow permissions: **Read and write**.
4. Live odds: get a free key at the-odds-api.com and add it as repo secret `ODDS_API_KEY`.
5. For the Claude workflow: in Claude Code run `/install-github-app`, or add an `ANTHROPIC_API_KEY` repo secret.
6. Run **Weekly predictions** once from the Actions tab (workflow_dispatch) to publish the first board.

Experiments: `PYTHONPATH=src:scripts python -c "from experiment import run; ..."` (see `scripts/experiment.py`).

Data: [nflverse](https://github.com/nflverse/nflverse-data). Not betting advice.
