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

**Totals model (display only, `totals_model.py`):** ridge on each team's EWMA points scored/allowed, EPA per play
and league scoring level, walk-forward. Mean absolute error vs the actual total, 2020–2025 holdout: model 10.62,
closing total 10.31 (1,693 games); over/under calls vs the closing total hit 50.0%. Shown on cards; not a bet track.

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

## Making sure it runs

GitHub's own `schedule` is best-effort: runs start late and are sometimes skipped (the first
scheduled 7:00am run on 2026-09-30 never fired). So the same times are also triggered from
outside by [cron-job.org](https://cron-job.org) (free), which calls GitHub's API to start the
workflow. Both fire; whichever starts second sees the first one (within 40 minutes) and skips
itself, so odds-API credits aren't spent twice. Clicking **Run workflow** (run_type `manual`)
never skips and also runs the Vegas comparison.

Outside-scheduler jobs (timezone America/Phoenix, which has no daylight saving; game-time jobs
fire twice, one hour apart, so one run lands right for Eastern daylight time and the other for
Eastern standard time; the off-season one is just an extra refresh). Each is a POST to
`https://api.github.com/repos/hollingswood/nfl-win-probability/actions/workflows/weekly.yml/dispatches`
with headers `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`,
`Content-Type: application/json`; success = HTTP 204:

| Job | When (Arizona) | Crontab | run_type |
|---|---|---|---|
| Daily | every day 7:10 AM | `10 7 * * *` | daily |
| Friday final report | Fri 2:40 PM | `40 14 * * 5` | friday |
| Sunday early (1pm ET games) | Sun 8:45 + 9:45 AM | `45 8,9 * * 0` | gameday |
| Sunday late (4pm ET games) | Sun 12:05 + 1:05 PM | `5 12,13 * * 0` | gameday |
| Night games (Sun/Mon/Thu) | 4:05 + 5:05 PM | `5 16,17 * * 0,1,4` | gameday |

Body: `{"ref":"main","inputs":{"run_type":"<run_type>"}}`.

The token is a fine-grained personal access token limited to this repository with
**Actions: Read and write** and nothing else. Failed runs email the repo owner (GitHub default).

## Replay on real early-week lines (2026-09-30)

`scripts/replay_early_lines.py` replays the frozen rules (spread v1, moneyline v1, grading v1) on
2020-2025 using 1,891 historical odds snapshots (The Odds API, `data/historical_odds/`) taken at
the same moments the live pipeline runs, with a walk-forward model, the licensed books in
`my_books.json`, the injury-report and QB-confirmed timing rules, first-qualifying-run locking and
the weekly cap. An independent audit found three replay bugs (stale "first seen" line, QB check on
the after-the-fact starter, missing first prices); the numbers below are after the fixes.

| Track | Bets | Flat ROI | 95% range | Avg CLV | Beat the close |
|---|---|---|---|---|---|
| Spread v1 | 506 | -0.3% | -8% to +8% | -0.2% | 51% |
| Moneyline v1 | 487 | +1.8% | -9% to +13% | -0.5% | 44% |

Verdict: no demonstrated edge on the lines we'd actually bet; grades did not rank results.
Model-free check of "favorites early, dogs late": favorites covered 49.3% at the early line vs
48.4% at the close (lines did not drift toward favorites on average) — right direction, too small
to be an edge on its own. Post-hoc leads (NOT validated; would need a new pre-registered version
tested on unseen data): spread underdogs taken early beat the close 67% of the time (176 bets,
CLV +1.2%); 8%+ edges beat the close 67% (78 bets).

## Edge search with a locked holdout (2026-09-30)

`scripts/edge_lab.py` searched for angles on 2020-2022 only (model v2 ideas, market-only
"soft book vs sharp book" prices, line momentum, key numbers, timing, season phase, dynamic
gating on trailing CLV, low-volume long shots). Seven candidates were frozen in
`edge_candidates.json` and committed, then tested once on 2023-2025
(`output/edge_holdout_2023_2025.json`). Pass = CLV > 0 at one-sided p < 0.05/7.

| Candidate | Holdout bets | ROI | CLV (pre-registered) | Pre-reg result | CLV at closing *prices* |
|---|---|---|---|---|---|
| C1 spread dog, soft vs sharp ≥3% | 459 | -4.3% | +2.0% | pass | **-0.7% / -1.0%** |
| C2 ML dog +100..+400, soft vs sharp ≥3% | 155 | +20.2% | +1.9% (p=0.04) | fail | +1.9% / +2.3% |
| C3 spread soft vs sharp ≥2% + model ≥2% | 167 | +0.7% | +2.1% | pass | **-1.7% / -2.0%** |
| C4 ML soft vs sharp ≥2% + model agrees | 145 | +0.9% | +1.2% (p=0.03) | fail | +2.5% / +2.4% |
| C5 spread model v2 (dogs, model ≥3%) | 247 | +1.1% | +0.7% | fail | -1.9% / -2.1% |
| C6 ML long shots +150..+300, model ≥12 pts over sharp | 57 | -8.1% | +0.6% | fail | +0.6% / +1.0% |
| C7 spread soft vs sharp, gated on trailing CLV | 453 | -4.7% | +1.2% | fail | -1.2% / -1.4% |

Important finding: the pre-registered spread CLV valued our bet against the closing *number*
(nflverse `spread_line`) and ignored the closing *juice*. Re-measured against the closing prices
of the sharp books / all books (last snapshot, `scripts/edge_check_spread_clv.py`), every spread
candidate has negative CLV, so C1/C3's "pass" is an artifact. The key-number distribution itself
is well calibrated (predicted vs actual cover/push within ~0.5 pt). Moneyline CLV needs no
key-number model and holds up (`scripts/edge_check_ml_clv.py`): C2/C4 are the only credible
leads, positive in both periods but not significant after the multiple-testing correction.
The live spread track's CLV (spread_bets.grade) has the same flaw and must use closing prices.

## Research round 2 (2026-09-30): what else was tested

Each study developed rules on 2020-2022 (or earlier), froze them in `output/research/*_frozen.json`,
then ran 2023-2025 once. Honest CLV = against closing *prices* (`edge_lab.closing_fair`).
Details in `output/research/<topic>.md`, code in `scripts/research/`.

| Study | Result | Verdict |
|---|---|---|
| Teasers (Wong 6-pt, 2-team) | best rule (totals ≤48.5, best number pre-kick): leg 76.5%, ROI +6% at -120, -1% at -140; CIs ±15% | No edge at today's -130/-140 prices |
| Line-movement model | direction slightly predictable (corr 0.11-0.16) but CLV after price ≤0 (ML variant +1.7%, p=0.09) | No edge |
| QB news timing | close prices QB changes correctly; early lines miss by ~2 pts, but only if you know first | No public-info edge; log news timing live |
| Totals + wind | Tuesday unders with ≥15 mph wind: CLV +7% but uses recorded wind (future info); Friday version +1.2% | Lead: paper-trade with the real Tuesday forecast |
| Sunday-night openers | more price gaps but they don't predict the close (CLV ≈0 to +1.6%, fails) | No edge |
| Hourly price checks (2025 sample) | +0.3-0.6 extra soft-vs-sharp bets/week at CLV ~+1.4%; opportunities last ~5-6 h | Small gain; hourly checks ~2,900 credits/month |
| State-space (Kalman) team ratings | ≤0.0006 log-loss change; no info beyond the closing line | Do not adopt |
