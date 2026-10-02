# Grade v2 for SPREAD offers: predicted honest spread CLV

Code `scripts/research/grade_v2_spread.py` (dev -> freeze -> holdout once -> report). Numbers `grade_v2_spread.json`; frozen model, bands and strategies `grade_v2_spread_frozen.json` (frozen 2026-10-02T08:18:09, before the 2023-25 run). Mirrors the moneyline grade v2 (`grade_v2.md`). Caveat: 2023-25 was the holdout of earlier studies too; nothing here was tuned on it.

**Target**: EV of our point + price under the closing sharp (LowVig/BetOnline, fallback all books) spread+juice implied margin, valued with the total-aware key-number model (`margin_total_model.json`) at the closing total -- the same model prices every entry reference. Robustness column: the old single-weight model at `edge_lab.closing_fair` (= `edge_lab.spread_clv_price`).

**Verdict: DO NOT ADOPT spread grade v2** (pre-registered bar: realized CLV rising with grade [False] and A+ CLV > 0 at one-sided p < 0.05 [False]).

## Model

Candidates: 20610 offer rows / 802 games (dev; all main-line allowed-book offers 94652, mean CLV -4.45%). LOSO (2020/21/22) game-weighted MSE, R² vs constant: base_ev_sp_sharp 0.067, ridge_core_a1 0.0729, ridge_core_a300 0.0729, ridge_core_a3000 0.0714, ridge_market_a1 0.0708, ridge_market_a300 0.0709, ridge_market_a3000 0.0695, ridge_small_a1 0.0703, ridge_small_a300 0.0709, ridge_small_a3000 0.0722, ridge_all_a1 0.0559, ridge_all_a300 0.0585, ridge_all_a3000 0.0643, lgbm_small_100x4 0.0633, lgbm_small_200x4 0.0611, lgbm_all_100x4 0.0591, lgbm_all_200x4 0.0532. Chosen: **ridge_core_a1** (complexity tiers base < ridge core < ridge market < ridge small < ridge all < lgbm small < lgbm all; start from the base (raw EV vs sharp spread); move to the best model of a higher tier only if its LOSO game-weighted MSE beats the current choice by > 0.5% relative (moneyline grade v2 rule, generalized)).

Ridge coefficients (CLV per 1 SD): ev_sp_sharp +1.370%, ev_mlimp +0.360%

Univariate offer-level Spearman with CLV (dev candidates): ev_sp_sharp +0.393, ev_sp_cons +0.308, best_gap -0.250, ev_mlimp +0.246, pt_off_cons +0.169, pt_off_sharp +0.153, move_mu +0.140, key_pos +0.113, disp +0.111, key_pos_cons +0.108, on3 +0.103, key_cross +0.100, abs_pt -0.092, move_pts +0.092, model_elig -0.092, is_dog -0.083, log_hours +0.067, overround -0.052, p_imp -0.038, sharp_missing +0.031, bk_williamhill_us -0.026, bk_betrivers +0.025, on7 -0.024, week -0.022

Bands (pred CLV, chosen on dev): A+ >= +1.00%, A >= +0.00%, B >= -1.00%, else C.

Dev band scan (OOF, first bet per game with pred >= cut): -0.010: 506 bets, CLV -0.35% ± 0.39; -0.005: 374 bets, CLV -0.20% ± 0.46; +0.000: 238 bets, CLV +0.69% ± 0.58; +0.005: 151 bets, CLV +1.80% ± 0.70; +0.010: 84 bets, CLV +2.56% ± 0.97; +0.015: 51 bets, CLV +3.39% ± 1.41; +0.020: 29 bets, CLV +4.90% ± 1.98; +0.025: 21 bets, CLV +5.55% ± 2.19; +0.030: 18 bets, CLV +6.93% ± 2.21; +0.040: 12 bets, CLV +8.77% ± 2.66

## Dev 2020-22 (out-of-fold predictions)

By grade (one bet per game per grade, first snapshot where the game's best spread offer has that grade):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| A+ | 84 (28.0/season) | +2.11% | +2.56% ± 0.97 | 0.004 | 68% | +9.71% ± 10.29 | +2.22% ± 0.94 | +0.77 |
| A | 173 (57.7/season) | +0.43% | +0.04% ± 0.68 | 0.4769 | 57% | +8.42% ± 7.12 | -0.13% ± 0.68 | +0.42 |
| B | 399 (133.0/season) | -0.55% | -0.65% ± 0.42 | 0.9384 | 45% | -5.77% ± 4.76 | -0.73% ± 0.42 | +0.44 |
| C | 756 (252.0/season) | -2.56% | -2.35% ± 0.29 | 1.0 | 33% | -6.97% ± 3.46 | -2.46% ± 0.29 | +0.18 |

Spearman across letters: rho 1.0; A+ minus C CLV +4.91% ± 0.99 (game bootstrap, z 4.9339); bet-level rank corr 0.2379; deciles rho 1.0; offer-level rank corr 0.3822; calibration slope 0.9586.

Deciles (pred -> realized CLV): -4.48%->-4.29%, -3.75%->-4.28%, -3.36%->-2.95%, -2.99%->-2.82%, -2.61%->-2.36%, -2.26%->-1.98%, -1.85%->-1.66%, -1.32%->-1.07%, -0.71%->-0.65%, +0.52%->+0.26%

Strategies and the live spread rule (spread_rules.json v1):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| S1_Aplus_flat | 84 (28.0/season) | +2.11% | +2.56% ± 0.97 | 0.004 | 68% | +9.71% ± 10.29 | +2.22% ± 0.94 | +0.77 |
| S2_A_and_up_flat | 238 (79.3/season) | +0.92% | +0.69% ± 0.58 | 0.1173 | 60% | +9.01% ± 6.07 | +0.50% ± 0.58 | +0.52 |
| rule spread v1 | 323 (107.7/season) | n/a | -1.37% ± 0.34 | 1.0 | 37% | -2.90% ± 5.32 | -1.38% ± 0.35 | +0.28 |

Grade v1 (`grading.py`, production use: each snapshot's best blend-EV offer):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| v1 A+ | 49 (16.3/season) | n/a | +2.97% ± 1.24 | 0.0082 | 57% | +26.18% ± 13.29 | +2.46% ± 1.19 | +0.66 |
| v1 A | 189 (63.0/season) | n/a | +1.62% ± 0.74 | 0.014 | 58% | +9.42% ± 6.81 | +1.37% ± 0.74 | +0.59 |
| v1 B+ | 450 (150.0/season) | n/a | -0.40% ± 0.42 | 0.8251 | 42% | +2.49% ± 4.47 | -0.53% ± 0.42 | +0.40 |
| v1 B | 611 (203.7/season) | n/a | -1.17% ± 0.36 | 0.9993 | 38% | +1.99% ± 3.81 | -1.25% ± 0.37 | +0.41 |
| v1 C+ | 582 (194.0/season) | n/a | -1.96% ± 0.38 | 1.0 | 38% | +1.96% ± 3.94 | -1.90% ± 0.38 | +0.29 |
| v1 C | 402 (134.0/season) | n/a | -3.09% ± 0.42 | 1.0 | 31% | -3.02% ± 4.80 | -2.86% ± 0.42 | +0.25 |

v1 Spearman across 6 letters: rho 1.0; v1 A+ minus C +6.06% ± 1.29; offer-level rank corr of v1 score with CLV 0.1711.

Same bets (the 1412 grade-v2 by-grade bets above), relabelled with v1: rank corr with CLV v1 score 0.1109 vs v2 prediction 0.2379.

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| same bets, v1 A+ | 23 (7.7/season) | -0.53% | +4.17% ± 2.07 | 0.022 | 70% | +0.32% ± 20.49 | +3.68% ± 2.00 | +0.85 |
| same bets, v1 A | 97 (32.3/season) | -0.96% | +2.31% ± 0.79 | 0.0017 | 66% | +17.77% ± 9.26 | +1.98% ± 0.78 | +0.69 |
| same bets, v1 B+ | 296 (98.7/season) | -1.13% | -1.08% ± 0.45 | 0.9918 | 40% | -8.49% ± 5.52 | -1.26% ± 0.45 | +0.26 |
| same bets, v1 B | 417 (139.0/season) | -1.37% | -1.81% ± 0.40 | 1.0 | 36% | -8.54% ± 4.60 | -1.98% ± 0.40 | +0.25 |
| same bets, v1 C+ | 348 (116.0/season) | -1.36% | -1.20% ± 0.45 | 0.996 | 45% | -5.29% ± 5.13 | -1.34% ± 0.45 | +0.37 |
| same bets, v1 C | 231 (77.0/season) | -1.81% | -2.78% ± 0.59 | 1.0 | 32% | +3.81% ± 6.32 | -2.61% ± 0.59 | +0.22 |

## HOLDOUT 2023-25 (frozen model, run once)

By grade (one bet per game per grade, first snapshot where the game's best spread offer has that grade):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| A+ | 80 (26.7/season) | +1.96% | -1.96% ± 1.30 | 0.9331 | 46% | +19.03% ± 10.29 | -2.29% ± 1.30 | +0.11 |
| A | 180 (60.0/season) | +0.37% | +0.10% ± 0.58 | 0.428 | 58% | +8.23% ± 7.05 | +0.00% ± 0.57 | +0.43 |
| B | 473 (157.7/season) | -0.57% | +0.11% ± 0.33 | 0.3761 | 53% | +9.73% ± 4.36 | +0.10% ± 0.34 | +0.49 |
| C | 772 (257.3/season) | -2.44% | -2.38% ± 0.24 | 1.0 | 30% | -3.22% ± 3.44 | -2.42% ± 0.24 | +0.19 |

Spearman across letters: rho 0.2; A+ minus C CLV +0.42% ± 1.24 (game bootstrap, z 0.34); bet-level rank corr 0.2288; deciles rho 0.9879; offer-level rank corr 0.4355; calibration slope 0.9382.

Deciles (pred -> realized CLV): -4.24%->-4.26%, -3.46%->-3.69%, -3.03%->-2.64%, -2.66%->-2.54%, -2.35%->-2.55%, -2.02%->-2.03%, -1.66%->-1.08%, -1.21%->-0.90%, -0.70%->-0.42%, +0.32%->+0.30%

Strategies and the live spread rule (spread_rules.json v1):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| S1_Aplus_flat | 80 (26.7/season) | +1.96% | -1.96% ± 1.30 | 0.9331 | 46% | +19.03% ± 10.29 | -2.29% ± 1.30 | +0.11 |
| S2_A_and_up_flat | 230 (76.7/season) | +0.85% | -0.96% ± 0.60 | 0.9462 | 52% | +9.67% ± 6.20 | -1.14% ± 0.60 | +0.28 |
| rule spread v1 | 306 (102.0/season) | n/a | -2.13% ± 0.25 | 1.0 | 26% | +5.99% ± 5.45 | -2.13% ± 0.25 | +0.15 |

Grade v1 (`grading.py`, production use: each snapshot's best blend-EV offer):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| v1 A+ | 49 (16.3/season) | n/a | +2.20% ± 1.34 | 0.0496 | 47% | +6.61% ± 13.59 | +1.95% ± 1.32 | +0.66 |
| v1 A | 168 (56.0/season) | n/a | -0.40% ± 0.62 | 0.7409 | 39% | +1.75% ± 7.39 | -0.57% ± 0.62 | +0.38 |
| v1 B+ | 443 (147.7/season) | n/a | -0.74% ± 0.37 | 0.9764 | 40% | +0.28% ± 4.52 | -0.85% ± 0.37 | +0.37 |
| v1 B | 634 (211.3/season) | n/a | -1.03% ± 0.33 | 0.9992 | 39% | +1.18% ± 3.74 | -1.08% ± 0.33 | +0.47 |
| v1 C+ | 552 (184.0/season) | n/a | -1.66% ± 0.36 | 1.0 | 38% | +1.95% ± 4.04 | -1.60% ± 0.36 | +0.38 |
| v1 C | 442 (147.3/season) | n/a | -1.98% ± 0.40 | 1.0 | 32% | +3.47% ± 4.57 | -1.73% ± 0.41 | +0.39 |

v1 Spearman across 6 letters: rho 1.0; v1 A+ minus C +4.18% ± 1.37; offer-level rank corr of v1 score with CLV 0.0454.

Same bets (the 1505 grade-v2 by-grade bets above), relabelled with v1: rank corr with CLV v1 score 0.0852 vs v2 prediction 0.2288.

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| same bets, v1 A+ | 28 (9.3/season) | -0.95% | +3.86% ± 2.07 | 0.0308 | 57% | +5.75% ± 17.93 | +3.53% ± 2.01 | +0.89 |
| same bets, v1 A | 91 (30.3/season) | -1.30% | +0.20% ± 0.83 | 0.4061 | 41% | +6.50% ± 10.05 | +0.01% ± 0.82 | +0.48 |
| same bets, v1 B+ | 278 (92.7/season) | -1.11% | -1.16% ± 0.43 | 0.9967 | 43% | -5.54% ± 5.66 | -1.31% ± 0.43 | +0.22 |
| same bets, v1 B | 483 (161.0/season) | -1.12% | -1.00% ± 0.33 | 0.9988 | 43% | +5.35% ± 4.29 | -1.05% ± 0.33 | +0.41 |
| same bets, v1 C+ | 336 (112.0/season) | -1.50% | -1.67% ± 0.40 | 1.0 | 39% | +8.35% ± 5.22 | -1.72% ± 0.40 | +0.26 |
| same bets, v1 C | 289 (96.3/season) | -1.49% | -2.35% ± 0.45 | 1.0 | 37% | +1.78% ± 5.70 | -2.26% ± 0.45 | +0.17 |

Holdout baseline -- same bands on raw EV vs the sharp spread-implied margin (no model):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | points vs close |
|---|---|---|---|---|---|---|---|---|
| A+ | 224 (74.7/season) | +2.25% | -1.06% ± 0.62 | 0.9552 | 51% | +15.62% ± 6.22 | -1.23% ± 0.62 | +0.28 |
| A | 356 (118.7/season) | +0.39% | -0.33% ± 0.39 | 0.8029 | 56% | -1.33% ± 5.05 | -0.30% ± 0.40 | +0.40 |
| B | 378 (126.0/season) | -0.50% | -0.40% ± 0.29 | 0.9148 | 39% | +3.86% ± 4.93 | -0.45% ± 0.28 | +0.30 |
| C | 753 (251.0/season) | -2.79% | -2.42% ± 0.24 | 1.0 | 29% | -2.85% ± 3.48 | -2.48% ± 0.25 | +0.19 |

Holdout by grade and season (CLV, bets):

* A+: 2023: -4.93% (19), 2024: +1.76% (22), 2025: -2.60% (39)
* A: 2023: -0.47% (54), 2024: -0.01% (46), 2025: +0.56% (80)
* B: 2023: -0.67% (146), 2024: +0.70% (131), 2025: +0.29% (196)
* C: 2023: -1.90% (251), 2024: -2.39% (247), 2025: -2.79% (274)

## Post-hoc (after the one-shot holdout; descriptive, not part of the test)

* dev_2020_22_in_sample_fit: A+ by hours before kick: <3h: 21 bets, CLV +2.09% ± 0.77; 3-30h: 11 bets, CLV +0.94% ± 1.53; 30-80h: 22 bets, CLV +5.45% ± 2.28; >80h: 44 bets, CLV +1.26% ± 1.51
  A+ with no sharp spread: 7 bets, CLV +4.22% ± 4.57; by book: betmgm: 37 bets, CLV +4.07% ± 1.52; betrivers: 22 bets, CLV +2.53% ± 1.85; draftkings: 5 bets, CLV -0.02% ± 2.44; fanduel: 23 bets, CLV -0.08% ± 1.85; williamhill_us: 11 bets, CLV +2.31% ± 2.52
  offer-level Spearman with CLV: ev_sp_sharp 0.4005, ev_mlimp 0.2461, ev_model_eligible 0.0403, v1_edge_blend 0.1325, move_mu 0.1395, pt_off_cons 0.1692
* holdout_2023_25: A+ by hours before kick: <3h: 2 bets, CLV +3.34% ± 0.50; 3-30h: 3 bets, CLV +1.98% ± 0.43; 30-80h: 8 bets, CLV -1.29% ± 2.01; >80h: 67 bets, CLV -2.37% ± 1.53
  A+ with no sharp spread: 20 bets, CLV -4.03% ± 2.56; by book: betmgm: 6 bets, CLV +1.18% ± 4.45; betrivers: 4 bets, CLV -1.61% ± 12.03; draftkings: 35 bets, CLV +1.47% ± 1.48; espnbet: 9 bets, CLV -3.71% ± 2.32; fanatics: 2 bets, CLV +0.16% ± 3.10; fanduel: 21 bets, CLV -7.83% ± 2.23; hardrockbet: 3 bets, CLV -3.59% ± 18.57
  offer-level Spearman with CLV: ev_sp_sharp 0.4749, ev_mlimp 0.1071, ev_model_eligible 0.1081, v1_edge_blend 0.1268, move_mu 0.0717, pt_off_cons 0.1066

## Reading

* The pre-registered bar is NOT met: spread grade v2 fails on 2023-25. A+ (pred >= +1%) realized CLV -2.0% ± 1.3
  (80 bets; dev OOF +2.6% ± 1.0), letters rho 0.2, A+ minus C +0.4% ± 1.2. A and B were ~0 (+0.1%), C -2.4%.
  Strategies: A+ only -2.0% ± 1.3; A+ and A -1.0% ± 0.6. No band is +CLV. (A+ ROI +19% ± 10 is luck: CLV says no.)
* What still holds: in the bulk the prediction is calibrated and ranks (decile rho 0.99, slope 0.94, offer-level rank
  corr 0.44, raw EV vs the sharp spread 0.47). It separates 'bad' (C, -2.4%) from 'about break-even' (A/B, ~0%), but it
  cannot find POSITIVE CLV: the top tail is mostly early-week (> 80 h before kick) soft-book numbers that look off
  the sharp line and the market then moves to the soft number (post-hoc: 67 of 80 A+ bets, -2.4%; the 20 with no
  LowVig/BetOnline spread -4.0%; FanDuel A+ -7.8%). Dev had no such pattern (> 80 h A+ +1.3%).
* Features: only the two price references matter out of sample -- EV vs the sharp spread+juice margin (coef 0.66 per
  unit EV) and EV vs the sharp-ML-implied margin (0.11). Off-market number, key-number position (3/7, crossing),
  juice, line move, book dispersion, hours, book, week and the model EV had univariate correlation but added nothing
  beyond the two EVs (full-feature ridge / LightGBM had LOWER LOSO R²: 0.064 / 0.053-0.063 vs 0.073 for the 2-feature
  ridge); the moneyline-vs-spread gap enters only through ev_mlimp.
* Grade v1 (grading.py, never fit to data) ranked spread CLV in BOTH periods: holdout rho 1.0 over six letters, A+
  +2.2% ± 1.3 (49 bets, p 0.05), A+ minus C +4.2% ± 1.4. Its A+ needs blend-model edge >= 5% + line moved our way +
  right side of 3/7, and post-hoc the walk-forward model EV correlated with spread CLV on 2023-25 (rank 0.11) but not
  on 2020-22 (0.04), which is why a model fit on 2020-22 could not learn it. On the same bets as v2, v1 A+ +3.9% ± 2.1
  (28). The live spread track rule (blend EV >= 3%) is -2.1% ± 0.3 CLV on 2023-25: the v1 grade is better than the
  rule it labels, but one borderline holdout does not make v1 A+ a bet signal.
* Recommendation: do not add spread grade v2 to production (do not copy the frozen file to the repo root). Keep v1 as
  the spread label. If anything, a new pre-registered spread study should start from v1's A+ conjunction and the
  model-blend EV, judged on 2026 live CLV; and do not take early-week off-market soft-book spread numbers on EV vs the
  sharp line alone.
* Caveats: 2023-25 was already the holdout of earlier studies; the dev model choice was extended (feature-set tiers)
  after the first dev CV run showed the full set overfit -- all before the freeze, none on 2023-25.

