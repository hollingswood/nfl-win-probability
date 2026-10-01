# Player props: projections + betting-test design (2026-10-01)

Code: `scripts/research/props.py` (`build` → `fit` → `evaluate`). Numbers: `output/research/props.json`.
This phase uses **no prop lines**. It builds and validates projections, then designs the data pull. It makes no edge claim.

## What was built
- **Outcomes.** Box-score stats come from nflverse pbp, REG+POST 2012-2025. They match nflverse weekly player stats exactly for 2020-24 (~100% of rows). That check is not independent, because nflverse weekly stats are also built from pbp. Official NFL stat corrections could differ slightly.
- **Universe (settlement-matched, prior-only eligibility).** The player must take at least 1 offensive snap. QB = the listed starter. RB rush = last-8 average of 25+ yds. WR/TE/RB receiving = last-8 average of 20+ yds. Every player needs 3+ prior games.
- **Decomposition.** Team opportunity is a ridge model on implied team total, expected margin, total, team/opponent decayed volume, pass rate over expected, wind, temp and roof. Player share is a decayed share of targets/carries/attempts, renormalised by teammates who are **active** ("late", bet after inactives) or **not Out/Doubtful** ("early", injury report). Efficiency is shrunk to position means, then multiplied by opponent factors (position-specific for targets).
- **Mean.** Structural mean = opportunity × share × efficiency × opponent. A LightGBM model with fixed parameters then corrects it. Both are walk-forward by season (train on seasons < S).
- **Distribution.** For each player-game, take the 400 out-of-sample (mean, outcome) pairs from the previous 4 seasons whose means are closest. Yards are rescaled to the new mean. Baselines get the same treatment, so CRPS isolates the quality of the mean.
- **Split.** Developed on ≤2022 (dev = 2019-22). The 2023-25 holdout was evaluated once.

## Holdout 2023-25, late model: MAE of median / CRPS (lower is better)
| market | n | season avg | last-4 | structural | **GBM** | ECE of P(over X) | P(over own median) pred/obs |
|---|---|---|---|---|---|---|---|
| pass_yds | 1615 | 64.2 / 51.1 | 62.6 / 45.9 | 60.0 / 42.6 | **58.5 / 41.5** | 0.009 | 0.500 / 0.492 |
| pass_att | 1615 | 6.90 / 4.89 | 6.87 / 4.90 | 6.74 / 4.81 | **6.66 / 4.76** | 0.016 | 0.476 / 0.457 |
| pass_cmp | 1615 | 4.84 / 3.44 | 4.80 / 3.44 | 4.70 / 3.36 | **4.62 / 3.30** | 0.019 | 0.464 / 0.446 |
| rush_yds | 2456 | 25.9 / 18.4 | 25.8 / 18.3 | 24.9 / 17.7 | **24.7 / 17.6** | 0.004 | 0.500 / 0.499 |
| rec_yds | 7176 | 23.9 / 17.0 | 24.2 / 17.1 | 23.0 / 16.2 | **22.8 / 16.1** | 0.003 | 0.500 / 0.499 |
| receptions | 7176 | 1.68 / 1.19 | 1.71 / 1.21 | 1.64 / 1.16 | **1.62 / 1.14** | 0.005 | 0.403 / 0.398 |

- **Gain over baselines.** GBM beats both baselines on every market, with game-clustered bootstrap 95% CIs that exclude 0. Absolute-error gain vs season average: pass yds 10.9 [8.7, 13.2], rec yds 1.6 [1.4, 1.9], rush yds 1.5 [1.0, 2.0], receptions 0.105 [0.09, 0.12].
- **QB baselines are weak.** The season average and last-4 include relief appearances, so a season average can be built from a few relief snaps.
- **Out-of-sample R² of the GBM mean** (from `paired` in props.json): rec yds 0.27, receptions 0.29, rush yds 0.25, pass yds 0.10, completions 0.10, attempts 0.07.
- **Early vs late.** The early (injury-report-only) model is barely worse: rec yds 22.95 vs 22.82, rush 24.99 vs 24.73. Pass yds 58.7 vs 58.5 is a wash.
- **Calibration.** P(over X) across a grid of fixed lines is well calibrated on the high-volume markets: ECE ≤ 0.005 for rec yds, receptions and rush yds; the reliability tables are in the json. PIT deciles are within 1.5 points of 10%.
- **QB volume drifted.** QB attempts and completions came in under the model in 2023-25 (P(over own median) observed 0.446-0.457 vs about 0.47 predicted). League passing volume fell, and a 4-season residual window does not adapt to that. Re-centre on recent seasons before betting QB volume.

## How accurate is the market? (ASSUMPTION, unmeasured)
I could not find a reliable published figure for how accurate prop lines are. In this repo, closing game lines are not beaten. **Working assumption A1:** closing prop lines are at least as accurate as this model, so our holdout MAE is an upper bound for market MAE: about 58 pass yds, 23 rec yds, 25 rush yds, 1.6 receptions. A plausible guess is that the market is 1-3% better.

What it takes to profit: under the model's own distribution, the line has to sit away from our median by:

| market | to reach 52.4% (-110 break-even) | to reach 55% |
|---|---|---|
| pass yds | 4 yds | 9 yds |
| rec yds | 1.6 yds | 3.7 yds |
| rush yds | 1.8 yds | 3.6 yds |

The pulled data will measure line MAE directly. A line is a median, so line MAE vs outcome is the right comparison.

## Data pull (The Odds API)
- **Endpoint and cost.** `/v4/historical/sports/americanfootball_nfl/events/{id}/odds` costs 10 credits per market returned per region. Up to 10 named bookmakers count as 1 region. The events list costs 1 credit per call. Props history starts 2023-05-03.
- **Settings.** `regions=us` (one region, includes lowvig/betonline). Market keys: `player_reception_yds`, `player_receptions`, `player_rush_yds`, `player_pass_yds`.
- **Events.** 2023-25 has 855 events (285 per season).

| plan, all of 2023-25 | credits |
|---|---|
| 1 market × 1 snapshot | 8.7K |
| **1 market × 2 snapshots** | **17.2K** |
| 2 markets × 2 snapshots | 34.3K |
| 3 markets × 2 snapshots | 51.4K |
| 4 markets × 2 snapshots | 68.5K |
| 1 market × 1 snapshot, one season | 2.85K |

Monthly plans: 20K = $30, 100K = $59.

**Snapshots.**
- **Early:** Fri 21:40 UTC, the repo's existing slot. For Thu/Sat games, use kickoff minus 24h.
- **Late:** about 75 min before kickoff, after inactives are announced. This is the CLV reference for early bets.

**Recommended pilot:** `player_reception_yds` × 2 snapshots × 2023-25 = **about 17.2K credits**. That fits one 20K month. It has the most lines per game (about 8), the best-calibrated distribution, and is where share and vacated-target information lives.
- If credits are tight now: 2025 only, 2 snapshots = 5.7K.
- Next step: add `player_rush_yds` and `player_pass_yds` (about +34K, on the 100K plan).

## Backtest protocol (pre-register before pulling)
1. **Match players.** Map Odds API player names to gsis ids (`players.parquet` display name + team). Report the unmatched rate. Drop players with 0 snaps, since those bets are void.
2. **Market fair probability.** Devig each book's over/under multiplicatively and take the median across books at the modal line. If books post different lines, shift the model's distribution to match each book, then take the consensus.
3. **Information test (main gate).** Fit a logistic model of outcome on logit(p_market) and logit(p_model) on **2023 only**. Score log loss on 2024-25 against market only, with a game-clustered bootstrap. Claim information only if the CI excludes 0.
4. **Bet rule.**
   - Bet the side whose blended EV at the best allowed-book price (`my_books.json`) is at least a threshold (for example +4%). Freeze the threshold and blend weight on 2023.
   - Flat 1 unit, at most 1 bet per player-market. Cap same-game exposure, because QB and WR props are correlated.
   - Early model at the Friday snapshot; late model at T-75.
5. **Grading.**
   - **CLV (primary):** P_close(T-75 consensus) × decimal odds − 1.
   - **PnL:** graded on the official stat, with pushes on whole-number lines.
   - **Breakdowns:** report overs and unders separately, against an "always under" baseline.
   - **Line accuracy:** line MAE vs model MAE (this tests assumption A1).
6. **Power.** 2024-25 gives about 4,300 rec-yds lines per snapshot. If about 25% qualify, that is about 1,000 bets. Win-rate standard error is about 1.6 percentage points, so only edges of roughly 3 points or more show in PnL. CLV is the realistic success metric.

## Risk and limits
- **Account limits.** Prop limits at DraftKings, FanDuel and similar books are small (often tens to low hundreds of dollars). Accounts that win on props get limited fast.
- **Where edges come from.** Real edges sit at line-open (Tue/Wed) and right after news. The 5-minute historical snapshots only capture them if a snapshot is pulled at that moment.
- **Noisy CLV.** Soft-book prop prices move on small money, so CLV against US books alone is noisy.
- **Weather.** The weather inputs are recorded conditions, not forecasts, which is optimistic for early bets. 2022 outdoor wind is 62% missing (filled with the median).
- **Lines.** The spread and total used are nflverse closing lines. An early bet would see the opener.
- **Early model blind spot.** Players newly placed on IR do not appear on the injury report, so the early model can count them as available for up to 3 games.
