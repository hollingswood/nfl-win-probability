# Seven-strategy modeling screen (2026-10-03)

Bar: one-sided p < 0.007 (7 strategies, Bonferroni) and positive in most seasons. Survivors are frozen as 2026 paper tracks.
Source: the deep-research report reports/Beating NFL lines with models.md.

| # | Strategy | Data | Result | Verdict |
|---|---|---|---|---|
| S6 | Preseason prior, NFL weeks 2-8: back the side preseason win totals favor when the market is 4+ pts away | 2013-2025 closing numbers and prices | 332 bets, **60.5% cover, ROI +16.5% ± 5.1%, p 0.0007**, 11/13 seasons positive; 2024-25 (after the paper) +7%; weeks 9-18 placebo 49%; not an underdog effect (dogs 61%, favorites 58%; all dogs 51.6%) | **PASS → track `preseason_prior`** (judged on results) |
| S3 | Predict the Tuesday→close spread move, bet the top decile now | hourly odds 2022-25, walk-forward 2023-25, Tuesday-only model (no injury report, no QB-change games) | 62 bets, **CLV +0.62 pts (p 0.0003)**, 57% moved our way / 23% against, 58% covers, ROI +11% ± 12% | **PASS → track `tuesday_move`** (judged on CLV) |
| S4 | Do model errors differ from the market's? Bet big model-vs-close gaps | 2014-2025 | ~20-25% of the model-vs-close gap shows up in results (β 0.22 ± 0.08), but betting gaps at closing prices 2020-25 (years the model wasn't tuned on): ROI −2.5% to +5%, n.s. | fail (informative) |
| S7 | Week-to-week adaptive model+market blend | 2015-2025 | log loss 0.6119 vs Vegas 0.6122 (z 0.18) | fail |
| S5 | Non-QB injury load (snap-weighted, final report) | 2013-25 outcomes; 2022-25 hourly timing | against heavily injured teams: ROI +2.5% (p 0.31); line keeps moving +0.08 pts after the Friday report (n.s., 100 games) | fail |
| S2 | News check on disagreements | applied to S6 bets | backed team's QB changed since week 1: 56.5% vs 62.7% unchanged (n.s., post-hoc) | label only |
| S1 | CFB median of ~50 public rating systems vs the opener, weeks 2-9 | Prediction Tracker 2010-2025 | lines move toward the ensemble (+1.17 pts, 60%/28%) **but bets cover 48.5% at the open**: the market moves toward computer ratings and overshoots | fail (CLV ≠ profit) |

## Bugs caught on the way (kept here so they are not repeated)
- The per-side `edge_lab` tables store results from the side's point of view; re-flipping them gave a fake 66% cover rate.
- First S3 run: the bet side was inverted (rising home point → bet AWAY now).
- First fixed S3 run used final-report injuries and actual starters (look-ahead): +1.0 pt / 64%, versus +0.62 pt / 58% with Tuesday-only information. Only the latter is frozen.
- The S4 2014-19 numbers are inflated: the model's features were selected on 2015-19.
