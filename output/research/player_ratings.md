# Player-level value model (RAPM) for pricing absences

Script: `scripts/research/player_ratings.py`. Full numbers: `player_ratings.json` (in this folder).

**Verdict: do not adopt.** RAPM player ratings built from on-field personnel carry real team-strength signal. But the value they put on an individual absence does not improve the walk-forward model. It does not predict the ATS residual against the close, and it does not explain early-week to close line moves. The market moves on how much a missing player plays (snap share), not on our per-player value. The early-week edge on injury news is real only if you already know the final status. Once the final report is public, it is gone.

## Data and method
- **Plays.** nflverse pbp for 2016-2025 (pass/rush plays with EPA; kneels and spikes dropped). These are joined to nflverse `pbp_participation`, which lists the 22 gsis ids on the field for each play. Plays with 10-11 listed per side are kept: about 35-37k per season. Participation starts in 2016, so every feature is 0 for 2012-2015. Files are cached in `$PR_CACHE` and nothing is written to `data/`.
- **RAPM.** A weighted ridge fit: EPA ≈ Σ offense players + Σ defense players + home + on-field rookie counts. It is refit before every (season, week), using only plays from games before that week's first gameday, with a 3-season window and an exponential decay half-life.
  - Penalties and half-life were tuned on out-of-sample next-4-week play EPA, 2017-19 only: λ_off=2000, λ_def=8000, half-life 365 d.
  - Out-of-sample R² on plays: **0.284%** for players vs **0.168%** for a tuned team-level ridge. Individual resolution does add information about the next plays.
  - Defensive individual effects are barely identifiable: the SD of a regular defender is 0.2-0.25 pts/game, vs 0.5-0.6 for offensive skill players and OL.
- **Role.** Each player's EWMA on-field share over his own previous appearances (half-life 4).
- **Team-game features** (non-QB, in points/game = EPA/play × 62):
  - `miss` = Σ Out(1.0)/Doubtful(0.8) × role × (rating − top-backup rating).
  - `missrepl` and `missraw` are the same with a replacement-level or zero baseline.
  - `missnew` counts only fresh absences (the player played last game).
  - `lineup` = expected available unit strength, with roles capped to 10/11 slots and gaps filled at replacement level.
  - `delta` = lineup − the EWMA of lineups actually fielded in the last 8 games.
- **Sanity check.** Non-QB `lineup_diff` alone correlates **0.38** with the final margin, vs 0.42 for the production model and 0.46 for the closing spread.

## (a) Walk-forward ridge (same protocol as `scripts/experiment.py`)
Log loss: val = 2015-19, holdout = 2020-25. Vegas closing moneyline benchmark: 0.6187 / 0.6070.

| feature set | val | val 2017-19 | holdout | holdout Brier |
|---|---|---|---|---|
| **production** | 0.61915 | 0.61279 | **0.62085** | 0.21599 |
| + miss (off, def) | 0.62007 | 0.61432 | 0.62146 | 0.21624 |
| + miss net | 0.61919 | 0.61285 | 0.62079 | 0.21597 |
| + missrepl | 0.62016 | 0.61447 | 0.62135 | 0.21624 |
| **+ missnew net (val-selected)** | **0.61849** | 0.61169 | 0.62106 | 0.21607 |
| + fresh snap-share only (no ratings) | 0.61966 | 0.61364 | 0.62089 | 0.21600 |
| + delta net | 0.62041 | 0.61488 | 0.62088 | 0.21600 |
| + lineup (off, def) | 0.61946 | 0.61330 | 0.62068 | 0.21583 |
| replace inj_off/def with miss | 0.62045 | 0.61453 | 0.62314 | 0.21693 |

- **The val-selected set does not hold up.** `+missnew net` makes the holdout worse by +0.0002, with a 95% bootstrap CI of [−0.0005, +0.0010].
- **Its coefficient has the wrong sign.** Fit on all seasons, it is −0.49 pts per point of rated value, after the production injury counts.
- **The best holdout row was rejected on validation.** `+lineup` scored 0.62068 on holdout but was worse on val, and worse on holdout when training starts in 2016.
- **Validation is handicapped.** 2015 has no features and the 2016 ratings are still burning in, so the 2017 fold explodes for multi-feature sets (for example +miss+delta: 0.637).

## (b) Market tests on injury games
Seasons: 2020-22 dev (834 games), 2023-25 holdout (854 games).
- **Line move** is the first-seen consensus line versus the price-implied close (`closing_fair`). Coefficients are robust-SE regressions.

| line move explained by | dev | holdout |
|---|---|---|
| production inj counts (off / def), pts per starter | +0.49 (t 5.0) / +0.40 (t 4.2) | +0.42 (t 4.9) / +0.28 (t 3.5) |
| RAPM miss (off / def) | t 1.0 / −0.8, R² 0.003 | t 1.1 / 1.0, R² 0.003 |
| RAPM miss net, added to inj counts | −0.005 (t −0.03) | +0.04 (t 0.4) |
| fresh role + fresh RAPM value | role t 6.2, value −0.33 (t −1.6) | role t 6.3, value **−0.50 (t −2.8)** |

- **ATS residual against the close.** No measure is significant in either period. RAPM `miss`/`delta` |t| ≤ 0.65; production counts |t| ≤ 1.3.
- **CLV of a frozen rule.** The rule: bet the side favored when |signal| ≥ the dev 90th percentile. CLV is the best allowed-book price, valued at the price-implied close. The baseline of betting any side scores −3.0%.

| signal / timing | dev CLV (n) | holdout CLV (n) |
|---|---|---|
| production inj, first snapshot (**knows final status**) | **+4.7%** t 2.9 (84) | **+3.2%** t 2.1 (83) |
| production inj, after final report | −0.2% (84) | −2.4% (83) |
| RAPM miss, first / after report | −2.9% / −1.0% | −1.8% / −3.8% |
| RAPM missnew, first / after report | −2.6% / −0.9% | −3.4% / −2.6% |

## (c) Leakage test
- **Masking test.** All plays from the cutoff date on and all later injury reports were erased. The cutoff week's ratings were refit and the cutoff-date features recomputed. Max feature difference: 3e-15 pts (2021-11-14) and 1e-8 pts (2024-10-06). **PASS.**
- **Control.** Letting the fit see the cutoff day changes ratings by up to 0.009 EPA/play, so the test has power.
- **The test caught a real bug.** Warm-starting scipy `lsqr` (`x0`) damps the correction, not the coefficients, so the prior silently became "last week's ratings." The first run's holdout market numbers came from those faulty ratings. Both runs gave the same verdict. Everything above is from the corrected, reproducible fit.

## Caveats
- **Hindsight.** nflverse keeps only the final weekly status. The "first snapshot" CLV therefore assumes Monday knowledge of Friday's report, and is optimistic. Questionable players are treated as available, and IR players do not appear at all.
- **No 2026 participation file.** The participation file for the current season was not checked, so the live feed timing is unknown.
- **Partial coverage.** Ratings exist only from 2016, and QBs are left to the production QB features.
