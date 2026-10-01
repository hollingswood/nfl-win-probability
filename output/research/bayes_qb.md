# Bayesian QB projection (hierarchical / empirical-Bayes Kalman)

Script: `scripts/research/bayes_qb.py` (stages `dev`, `holdout`, `report`). Output: `output/research/bayes_qb.json`.

**Verdict:** No gain; do not adopt. The Bayes posterior is a sound QB estimate (unbiased where production runs about +0.02 EPA/db low, nearly calibrated SD with z-SD 1.04-1.12, priors that make sense), but it is 0.94-0.95 correlated with the production EWMA. Swapping it in or adding it does not improve walk-forward log loss overall or on QB-change / <300-dropback games. Every difference is within about 2 SE, and the subset signs flip between validation and holdout. The best holdout cell, B4 on new-QB games at -0.0021±0.0010, was +0.0011 in validation. Using the posterior SD to widen σ is rejected on validation (c = 0). The fast-stabilising components add nothing over an EPA-only filter. Against the market, the Bayes-minus-production QB revision predicts the early-week to close line move on QB-change games (dev t = 3.7, holdout t = 2.3, p = 0.011). That does not clear the pre-registered Bonferroni bar, and the bettable version (E3) has −0.3% CLV in the holdout. E2 clears the bar nominally (+3.1% CLV, p = 0.007). It is hindsight on the actual starter, though, and production gets the same (E2p +2.9%), so the Bayes rating adds nothing to it. M1 flipped sign between dev and holdout. The closing line already has this information.

## Method

- Latent talent per QB = (EPA/dropback, success rate, CPOE, sack rate). Prior at first appearance: N(B·x, λ·T), x = draft pick (log, UDFA = 300), years in league at debut, age at entry, pre-2012-veteran flag.
- Observation per QB-game: component means with per-play noise covariance / n. The cross-covariances between talent components let faster-stabilising stats (success, CPOE, sacks) move the EPA estimate.
- Dynamics: random walk with drift a·T per game and b·T per season, plus a season-boundary development shift d by years in league × career backup (< 300 career dropbacks).
- Hyperparameters are walk-forward. H_s is fit on seasons < s (2015-19); H_2020 is fit on 2012-19 and frozen for 2020+.
- Pre-game posterior mean (bq) and SD (bq_sd) for the **actual** starter (nflverse `*_qb_id`, the same hindsight as production).

## Priors learned (H_2020, fit on 2012-2019)

Drift: a = 0.0 per game, b = 0.25 per season (× between-QB talent covariance, SD of EPA talent = 0.110). Prior SD of a debut QB's EPA/db = 0.119, equivalent to about 172 dropbacks of data. For comparison, production uses a fixed −0.10 worth 150 dropbacks.

| QB type | EPA/db mean | SD | success | CPOE | sack% |
|---|---|---|---|---|---|
| 1st-round top-10 rookie (pick 5, age 22) | -0.032 | 0.119 | 0.426 | -2.7 | 7.4 |
| late 1st-round rookie (pick 25, age 22.5) | -0.065 | 0.119 | 0.417 | -3.2 | 7.3 |
| 3rd-round rookie (pick 80) | -0.088 | 0.119 | 0.411 | -3.5 | 7.3 |
| 6th-round rookie (pick 190) | -0.105 | 0.119 | 0.407 | -3.7 | 7.3 |
| undrafted rookie | -0.114 | 0.119 | 0.405 | -3.8 | 7.4 |
| late-round backup, first snaps in year 4 (pick 200) | -0.108 | 0.119 | 0.412 | -3.3 | 8.7 |
| undrafted backup, first snaps in year 5 | -0.116 | 0.119 | 0.412 | -3.2 | 9.2 |

Season-boundary development shift d (EPA/db; `_bk` = under 300 career dropbacks entering the season):

| bucket | d_epa |
|---|---|
| y13+ | -0.026 |
| y2 | +0.031 |
| y2_bk | +0.117 |
| y3 | +0.003 |
| y4 | -0.012 |
| y5-7 | -0.013 |
| y5-7_bk | -0.050 |
| y8-12 | +0.003 |
| y8-12_bk | -0.051 |

## Next-game EPA/dropback prediction (QB-games with 10+ dropbacks, dropback-weighted)

| period \| group | n | MSE Bayes | MSE Bayes EPA-only | MSE production | bias Bayes | bias prod | z-SD Bayes |
|---|---|---|---|---|---|---|---|
| val 2015-19 \| all | 2762 | 0.08297 | 0.08286 | 0.08249 | -0.005 | 0.0228 | 1.082 |
| val 2015-19 \| career<300db | 418 | 0.10092 | 0.10078 | 0.10182 | -0.0034 | 0.0269 | 1.119 |
| val 2015-19 \| career>=300db | 2344 | 0.08009 | 0.07999 | 0.0794 | -0.0052 | 0.0222 | 1.076 |
| hold 2020-25 \| all | 3547 | 0.08142 | 0.0816 | 0.08197 | -0.0 | 0.0209 | 1.057 |
| hold 2020-25 \| career<300db | 512 | 0.09118 | 0.09173 | 0.09098 | -0.0152 | 0.007 | 1.044 |
| hold 2020-25 \| career>=300db | 3035 | 0.07992 | 0.08004 | 0.08059 | 0.0023 | 0.023 | 1.059 |

Correlation of bq_diff with production qb_diff: {'all': 0.944, 'qb_change_games': 0.9508, 'sd_bq_rev_pts_qb_change': 2.084, 'sd_mu_bayes_minus_mu_prod': 0.088, 'sd_mu_b1_minus_mu_prod': 0.488}

## (a) Walk-forward ridge log loss (tune 2015-19, holdout 2020-25)

Variants: B1 = bq_diff replaces qb_diff (and fw_qb_diff); B2 = B1 plus the Bayes qb_change; B3 = add bq_diff; B4 = add bq_diff + Bayes change; B5 = best mean variant plus σ widened by QB posterior variance (c chosen on validation = 0.0). Best by validation: **B3_add_bq_diff**. Cells are the log-loss difference versus production (negative = better) ± game-bootstrap SE.

| period \| subset | n | vegas | P0_production | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|---|---|
| val 2015-19 \| all | 1330 | 0.6187 | 0.6192 | +0.0004±0.0008 | +0.0012±0.0016 | +0.0001±0.0002 | +0.0001±0.0005 | +0.0001±0.0002 |
| val 2015-19 \| qb_change_game | 213 | 0.6277 | 0.6181 | +0.0019±0.0025 | +0.0023±0.0051 | -0.0002±0.0004 | +0.0001±0.0013 | -0.0002±0.0004 |
| val 2015-19 \| new_qb_game | 100 | 0.5977 | 0.5768 | +0.0043±0.0046 | +0.0091±0.0090 | +0.0008±0.0006 | +0.0011±0.0024 | +0.0008±0.0006 |
| val 2015-19 \| starter_<300_career_db | 342 | 0.5743 | 0.5577 | +0.0007±0.0017 | +0.0009±0.0039 | +0.0005±0.0003 | +0.0003±0.0010 | +0.0005±0.0003 |
| hold 2020-25 \| all | 1688 | 0.607 | 0.6209 | +0.0002±0.0006 | +0.0010±0.0013 | +0.0001±0.0001 | +0.0000±0.0002 | +0.0001±0.0001 |
| hold 2020-25 \| qb_change_game | 337 | 0.566 | 0.601 | +0.0000±0.0016 | -0.0019±0.0037 | +0.0001±0.0003 | -0.0009±0.0006 | +0.0001±0.0003 |
| hold 2020-25 \| new_qb_game | 159 | 0.5313 | 0.5876 | -0.0027±0.0021 | -0.0075±0.0060 | +0.0002±0.0005 | -0.0021±0.0010 | +0.0002±0.0005 |
| hold 2020-25 \| starter_<300_career_db | 399 | 0.586 | 0.592 | -0.0013±0.0014 | +0.0000±0.0037 | -0.0001±0.0002 | -0.0002±0.0005 | -0.0001±0.0002 |

## (b) Market tests

### ATS residual vs the nflverse closing spread (home margin − spread_line)

bq_rev_pts = 38 × [(Bayes − production) home QB − (Bayes − production) away QB]. mu = walk-forward ridge margin (2015+).

| test | n | slope | t |
|---|---|---|---|
| 2013-2019: qb_change_games: resid ~ bayes revision vs production (pts) | 297 | +0.403 ±0.356 | 1.13 |
| 2013-2019: all_games: resid ~ bayes revision (pts) | 1862 | -0.005 ±0.165 | -0.03 |
| 2013-2019: low_db_games: resid ~ bayes revision (pts) | 511 | +0.470 ±0.336 | 1.4 |
| 2013-2019: qb_change_games: resid ~ (mu_bayes - spread) | 213 | +0.643 ±0.265 | 2.43 |
| 2013-2019: qb_change_games: resid ~ (mu_prod - spread) | 213 | +0.643 ±0.266 | 2.42 |
| 2013-2019: qb_change_games: resid ~ (mu_bayes - mu_prod) | 213 | +4.276 ±5.289 | 0.81 |
| 2020-2025: qb_change_games: resid ~ bayes revision vs production (pts) | 337 | +0.288 ±0.283 | 1.02 |
| 2020-2025: all_games: resid ~ bayes revision (pts) | 1688 | +0.153 ±0.155 | 0.99 |
| 2020-2025: low_db_games: resid ~ bayes revision (pts) | 399 | +0.328 ±0.251 | 1.3 |
| 2020-2025: qb_change_games: resid ~ (mu_bayes - spread) | 337 | -0.117 ±0.202 | -0.58 |
| 2020-2025: qb_change_games: resid ~ (mu_prod - spread) | 337 | -0.121 ±0.203 | -0.59 |
| 2020-2025: qb_change_games: resid ~ (mu_bayes - mu_prod) | 337 | +4.288 ±7.157 | 0.6 |
| 2013-2025: qb_change_games: resid ~ bayes revision vs production (pts) | 634 | +0.374 ±0.223 | 1.68 |
| 2013-2025: all_games: resid ~ bayes revision (pts) | 3550 | +0.076 ±0.113 | 0.67 |
| 2013-2025: low_db_games: resid ~ bayes revision (pts) | 910 | +0.414 ±0.205 | 2.02 |
| 2013-2025: qb_change_games: resid ~ (mu_bayes - spread) | 550 | +0.231 ±0.176 | 1.31 |
| 2013-2025: qb_change_games: resid ~ (mu_prod - spread) | 550 | +0.228 ±0.177 | 1.29 |
| 2013-2025: qb_change_games: resid ~ (mu_bayes - mu_prod) | 550 | +4.682 ±4.369 | 1.07 |

### Early week, dev 2020-22: 834 games, early snapshot median 148h before kickoff

| test | n | slope | t | p (1-sided) |
|---|---|---|---|---|
| M1 move ~ (mu_bayes - mu_prod), qb_change games | 171 | -7.511 ±2.829 | -2.66 | 0.9961 |
| move ~ (mu_bayes - m_early), qb_change games | 171 | +0.370 ±0.060 | 6.14 |  |
| move ~ (mu_prod - m_early), qb_change games | 171 | +0.377 ±0.061 | 6.21 |  |
| M1 all games | 834 | -3.808 ±1.128 | -3.38 |  |
| M2 move ~ bq_rev_pts (Bayes minus production QB rating, pts), qb_change games | 171 | +0.401 ±0.108 | 3.7 | 0.0001 |
| M3 move ~ (mu_b1 - mu_prod), qb_change games | 171 | +0.301 ±0.857 | 0.35 | 0.3632 |

| rule | bets | CLV | p (1-sided) | sharp-close CLV | beat close |
|---|---|---|---|---|---|
| E1 bayes revision side, \|rev\|>=0.5 | 0 |  |  |  |  |
| E2 mu_bayes vs early line >= 1.5 | 108 | +4.2% ±1.5 | 0.0026 | +4.2% | 0.574 |
| E2p mu_prod vs early line >= 1.5 (reference) | 106 | +4.4% ±1.5 | 0.0019 | +4.5% | 0.575 |
| E3 side of bq_rev_pts, \|rev\|>=1.0 pt | 96 | +2.4% ±1.6 | 0.069 | +2.4% | 0.615 |
| E4 side of mu_b1 - mu_prod, \|rev\|>=0.5 pt | 23 | -5.8% ±4.9 | 0.8813 | -5.9% | 0.391 |
| N0 null: home side | 171 | -4.0% ±1.2 | 0.9997 | -3.9% | 0.345 |
| N0 null: away side | 171 | -2.4% ±1.2 | 0.9805 | -2.5% | 0.45 |

### Early week, HOLDOUT 2023-25 (run once): 854 games, early snapshot median 150h before kickoff

| test | n | slope | t | p (1-sided) |
|---|---|---|---|---|
| M1 move ~ (mu_bayes - mu_prod), qb_change games | 166 | +6.336 ±2.714 | 2.33 | 0.0099 |
| move ~ (mu_bayes - m_early), qb_change games | 166 | +0.276 ±0.056 | 4.9 |  |
| move ~ (mu_prod - m_early), qb_change games | 166 | +0.273 ±0.057 | 4.83 |  |
| M1 all games | 854 | +3.793 ±1.044 | 3.63 |  |
| M2 move ~ bq_rev_pts (Bayes minus production QB rating, pts), qb_change games | 166 | +0.201 ±0.088 | 2.29 | 0.011 |
| M3 move ~ (mu_b1 - mu_prod), qb_change games | 166 | -0.442 ±0.410 | -1.08 | 0.8599 |

| rule | bets | CLV | p (1-sided) | sharp-close CLV | beat close |
|---|---|---|---|---|---|
| E1 bayes revision side, \|rev\|>=0.5 | 0 |  |  |  |  |
| E2 mu_bayes vs early line >= 1.5 | 107 | +3.1% ±1.3 | 0.007 | +3.3% | 0.598 |
| E2p mu_prod vs early line >= 1.5 (reference) | 107 | +2.9% ±1.3 | 0.0119 | +3.0% | 0.589 |
| E3 side of bq_rev_pts, \|rev\|>=1.0 pt | 101 | -0.3% ±1.3 | 0.5841 | -0.5% | 0.455 |
| E4 side of mu_b1 - mu_prod, \|rev\|>=0.5 pt | 34 | -4.6% ±2.8 | 0.9476 | -4.7% | 0.382 |
| N0 null: home side | 166 | -2.6% ±1.0 | 0.9954 | -2.6% | 0.367 |
| N0 null: away side | 166 | -2.9% ±1.0 | 0.9981 | -3.0% | 0.422 |

Frozen rules (before the holdout):

```
{
 "note": "Thresholds fixed a priori (not tuned). E1/M1 were written before any result; after the first dev run showed the validation-chosen variant (B3) barely moves the margin (E1: 0 bets), M2/M3/E3/E4 were added using a-priori thresholds and the whole list is frozen here, before the 2023-25 holdout is run once. All CLV is price-based vs closing_fair().mu_close_all at the first early-week snapshot (>= 4h after both teams' previous kickoffs, <= 9 days out), best allowed book. Hindsight caveat: starters are the ACTUAL starters, which the early-week market may not know yet.",
 "E1": "QB-change games: bet side of mu_bayes - mu_prod (best-val variant B3) when |rev| >= 0.5 pt",
 "E2": "QB-change games: bet side with |mu_bayes - m_early| >= 1.5 pts (reference E2p: same with mu_prod)",
 "E3": "QB-change games: bet side of bq_rev_pts (38 x [Bayes - production QB rating diff]) when |rev| >= 1.0 pt",
 "E4": "QB-change games: bet side of mu_b1 - mu_prod (B1: Bayes replaces qb_diff) when |rev| >= 0.5 pt",
 "M1": "QB-change games: early->close consensus move (home pts) ~ mu_bayes - mu_prod, slope > 0",
 "M2": "QB-change games: move ~ bq_rev_pts, slope > 0",
 "M3": "QB-change games: move ~ mu_b1 - mu_prod, slope > 0",
 "success_criterion": "one-sided p < 0.05/7 (Bonferroni over E1-E4, M1-M3); E2 must also beat E2p to credit the Bayes rating",
 "frozen_at": "2026-10-01T09:26:33",
 "dev_result_known": true
}
```

## (c) Leakage masking test

All results from the cutoff on are erased (pbp and completion), hyperparameters and the filter are refit, and features for games on the cutoff date are compared.

| cutoff | games | max_abs_diff | pass |
|---|---|---|---|
| 2017-10-08 | 12 | 0.0 | True |
| 2022-10-09 | 14 | 0.0 | True |
| 2023-12-24 | 10 | 0.0 | True |
| 2024-09-08 | 13 | 0.0 | True |

## Caveats

- Starters are the actual starters (nflverse), the same as production. Early-week CLV tests therefore assume the QB news is known at the early snapshot (hindsight; see qb_timing.md).
- Career dropbacks count only 2012+ pbp. Pre-2012 veterans carry a separate prior and their own variance.
- Warm-up seasons 2012-14 (training rows only, never scored) use H_2015, which is fit on 2012-14.
- Opponent adjustment and the league-wide EPA level are not modelled. Both mostly cancel in a home-minus-away difference.
