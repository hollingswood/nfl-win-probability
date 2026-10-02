# Grade granularity (how many letters?) and a totals grade v2

Code `scripts/research/grade_granular.py`; numbers `grade_granular.json`; frozen totals grade `grade_totals_frozen.json`.

## Q1. Finer letters for the moneyline grade v2?

Bands chosen on 2020-22 out-of-fold (LOSO) predictions only. **2023-25 was already used once for the 4-band grade v2 test, so every 2023-25 number below is descriptive, not a new test.** Bets: one per (game, band) at the first snapshot where the game's best predicted-CLV offer falls in that band (the grade_v2.py convention).

* NESTED 9 (refines the production 4: old A+ -> A+/A, A -> A-/B+, B -> B/B-, C -> C+/C/C-): A+ >= +3.00%, A >= +2.50%, A- >= +2.00%, B+ >= +1.50%, B >= +1.00%, B- >= +0.50%, C+ >= +0.00%, C >= -1.00%, else C-.
* EQUAL 9 (equal-count dev quantiles of snapshot-best predictions): A+ >= +0.93%, A >= +0.14%, A- >= -0.39%, B+ >= -0.86%, B >= -1.34%, B- >= -1.83%, C+ >= -2.33%, C >= -3.18%, else C-.

### dev_oof: four

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 89 | +3.01% | +4.16% ± 0.92 | 0.0 | 78% | -0.8% ± 14.8 |
| A | 203 | +1.95% | +1.53% ± 0.70 | 0.0151 | 66% | +5.0% ± 9.6 |
| B | 395 | +0.92% | +0.55% ± 0.52 | 0.1439 | 56% | -1.2% ± 6.7 |
| C | 827 | -1.30% | -0.77% ± 0.42 | 0.9655 | 46% | +3.3% ± 5.0 |

Spearman across bands 1.0; inversions 0; adjacent pairs distinguishable at z >= 1.96: **2 of 3**; bet-level rank corr 0.1825.

Adjacent differences (game-cluster bootstrap): A+ vs A: +2.64% (z 2.4296); A vs B: +0.97% (z 1.2777); B vs C: +1.32% (z 2.4645)

### dev_oof: nested9

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 54 | +3.35% | +4.65% ± 1.11 | 0.0 | 83% | +16.6% ± 20.5 |
| A | 59 | +2.71% | +3.46% ± 1.05 | 0.0005 | 75% | +0.2% ± 18.8 |
| A- | 101 | +2.21% | +1.60% ± 0.83 | 0.0267 | 69% | +4.9% ± 14.2 |
| B+ | 131 | +1.75% | +1.60% ± 0.94 | 0.0437 | 66% | +3.1% ± 11.4 |
| B | 188 | +1.23% | +0.66% ± 0.77 | 0.1964 | 59% | -14.6% ± 8.9 |
| B- | 287 | +0.74% | +0.50% ± 0.56 | 0.1896 | 54% | +7.0% ± 8.1 |
| C+ | 376 | +0.24% | +0.02% ± 0.57 | 0.4867 | 46% | +5.6% ± 6.6 |
| C | 653 | -0.51% | +0.37% ± 0.40 | 0.1814 | 49% | -1.3% ± 5.1 |
| C- | 765 | -2.22% | -1.89% ± 0.39 | 1.0 | 36% | -4.5% ± 5.2 |

Spearman across bands 0.9791; inversions 1; adjacent pairs distinguishable at z >= 1.96: **1 of 8**; bet-level rank corr 0.1998.

Adjacent differences (game-cluster bootstrap): A+ vs A: +1.18% (z 0.912); A vs A-: +1.87% (z 1.5439); A- vs B+: -0.01% (z -0.005); B+ vs B: +0.94% (z 0.8855); B vs B-: +0.16% (z 0.1954); B- vs C+: +0.48% (z 0.6771); C+ vs C: -0.35% (z -0.6611); C vs C-: +2.26% (z 5.5819)

### dev_oof: equal9

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 373 | +1.78% | +1.20% ± 0.58 | 0.0187 | 64% | -2.2% ± 6.9 |
| A | 433 | +0.49% | +0.06% ± 0.51 | 0.4553 | 48% | +6.7% ± 6.5 |
| A- | 441 | -0.14% | +0.13% ± 0.52 | 0.3983 | 49% | +4.7% ± 6.4 |
| B+ | 457 | -0.64% | +0.40% ± 0.45 | 0.1891 | 49% | -4.2% ± 5.9 |
| B | 464 | -1.09% | -0.92% ± 0.42 | 0.986 | 39% | -2.0% ± 5.9 |
| B- | 453 | -1.58% | -1.98% ± 0.35 | 1.0 | 31% | -2.6% ± 6.5 |
| C+ | 416 | -2.06% | -1.80% ± 0.39 | 1.0 | 30% | -9.7% ± 6.7 |
| C | 399 | -2.69% | -2.20% ± 0.45 | 1.0 | 29% | +8.8% ± 7.9 |
| C- | 271 | -4.27% | -3.66% ± 0.61 | 1.0 | 27% | -4.0% ± 10.7 |

Spearman across bands 0.9167; inversions 3; adjacent pairs distinguishable at z >= 1.96: **3 of 8**; bet-level rank corr 0.2152.

Adjacent differences (game-cluster bootstrap): A+ vs A: +1.14% (z 1.9337); A vs A-: -0.08% (z -0.1302); A- vs B+: -0.26% (z -0.4655); B+ vs B: +1.31% (z 2.7721); B vs B-: +1.06% (z 2.2379); B- vs C+: -0.18% (z -0.3934); C+ vs C: +0.40% (z 0.7944); C vs C-: +1.46% (z 2.481)

Bet-level calibration (realized on predicted, nested-9 bets): slope 0.89 ± 0.12, residual SD 10.3%, R² 0.021; offer-level OOS R² 0.1033, slope 1.0283.

### holdout_descriptive: four

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 163 | +3.15% | +2.34% ± 0.98 | 0.0086 | 60% | +22.2% ± 14.6 |
| A | 206 | +1.96% | +0.09% ± 0.73 | 0.4519 | 55% | -11.7% ± 10.5 |
| B | 432 | +0.92% | -0.09% ± 0.50 | 0.5737 | 49% | -7.8% ± 7.0 |
| C | 851 | -1.13% | -1.30% ± 0.40 | 0.9995 | 42% | -0.4% ± 4.8 |

Spearman across bands 1.0; inversions 0; adjacent pairs distinguishable at z >= 1.96: **2 of 3**; bet-level rank corr 0.1441.

Adjacent differences (game-cluster bootstrap): A+ vs A: +2.25% (z 2.1378); A vs B: +0.18% (z 0.2499); B vs C: +1.21% (z 2.4115)

### holdout_descriptive: nested9

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 113 | +3.48% | +2.76% ± 1.01 | 0.0031 | 65% | +38.2% ± 18.3 |
| A | 92 | +2.75% | +1.82% ± 1.53 | 0.117 | 55% | +23.0% ± 20.1 |
| A- | 103 | +2.25% | +0.94% ± 0.89 | 0.1457 | 63% | -4.5% ± 16.4 |
| B+ | 131 | +1.73% | -0.32% ± 0.94 | 0.6331 | 50% | -24.6% ± 11.4 |
| B | 210 | +1.22% | +0.83% ± 0.73 | 0.1299 | 54% | -9.9% ± 10.4 |
| B- | 321 | +0.74% | -0.51% ± 0.57 | 0.8159 | 45% | -7.2% ± 8.0 |
| C+ | 385 | +0.23% | -0.64% ± 0.50 | 0.9008 | 46% | -0.6% ± 7.1 |
| C | 672 | -0.52% | -0.62% ± 0.38 | 0.9496 | 42% | +2.3% ± 5.5 |
| C- | 786 | -2.09% | -2.53% ± 0.35 | 1.0 | 30% | -0.2% ± 5.0 |

Spearman across bands 0.9667; inversions 2; adjacent pairs distinguishable at z >= 1.96: **1 of 8**; bet-level rank corr 0.1807.

Adjacent differences (game-cluster bootstrap): A+ vs A: +0.94% (z 0.6593); A vs A-: +0.88% (z 0.5351); A- vs B+: +1.26% (z 0.9989); B+ vs B: -1.15% (z -1.0753); B vs B-: +1.34% (z 1.6616); B- vs C+: +0.13% (z 0.1944); C+ vs C: -0.02% (z -0.0316); C vs C-: +1.91% (z 4.7934)

### holdout_descriptive: equal9

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 419 | +1.96% | +0.92% ± 0.56 | 0.0506 | 54% | +1.6% ± 7.8 |
| A | 464 | +0.52% | -0.33% ± 0.47 | 0.7611 | 47% | -4.9% ± 6.6 |
| A- | 469 | -0.13% | -0.68% ± 0.45 | 0.9379 | 41% | -3.1% ± 6.3 |
| B+ | 446 | -0.63% | -0.75% ± 0.43 | 0.9613 | 42% | +0.6% ± 6.7 |
| B | 504 | -1.09% | -1.03% ± 0.38 | 0.9965 | 35% | +3.6% ± 6.3 |
| B- | 479 | -1.58% | -2.41% ± 0.37 | 1.0 | 27% | -2.7% ± 6.0 |
| C+ | 431 | -2.05% | -2.56% ± 0.38 | 1.0 | 24% | -4.7% ± 6.3 |
| C | 406 | -2.68% | -3.43% ± 0.43 | 1.0 | 22% | -11.2% ± 6.5 |
| C- | 304 | -4.30% | -5.32% ± 0.54 | 1.0 | 15% | -5.9% ± 9.1 |

Spearman across bands 1.0; inversions 0; adjacent pairs distinguishable at z >= 1.96: **3 of 8**; bet-level rank corr 0.2217.

Adjacent differences (game-cluster bootstrap): A+ vs A: +1.25% (z 2.1209); A vs A-: +0.35% (z 0.6517); A- vs B+: +0.07% (z 0.1331); B+ vs B: +0.28% (z 0.5248); B vs B-: +1.38% (z 3.0626); B- vs C+: +0.15% (z 0.4159); C+ vs C: +0.86% (z 1.8295); C vs C-: +1.89% (z 3.3198)

Bet-level calibration (realized on predicted, nested-9 bets): slope 0.88 ± 0.11, residual SD 10.1%, R² 0.022; offer-level OOS R² 0.1362, slope 1.0656.

### Resolution limit

Predicted-CLV gap two bands need so that their realized means differ at z = 1.96, by bets per band (1.96·√2·residual SD / (slope·√n)): dev calibration n=25: +6.47%, n=50: +4.57%, n=100: +3.23%, n=150: +2.64%, n=250: +2.05%, n=400: +1.62%; holdout calibration n=25: +6.36%, n=50: +4.50%, n=100: +3.18%, n=150: +2.60%, n=250: +2.01%, n=400: +1.59%.

Optimal partition (dynamic programming over cuts every 0.25%: for k bands, maximize the smallest adjacent expected z; expected CLV = bet-level calibration line, SE = residual SD / sqrt(n), n = first-appearance bets of the interval scaled to the horizon). Distinguishable bands = largest k with min z >= 1.96:

* 1_season, dev_preds_dev_calibration: **3 distinguishable bands** (best min adjacent z for k = 2..6: 2: 2.6906, 3: 2.2585, 4: 1.7328, 5: 1.4393, 6: 1.1356): [+1.25%, ∞) n=100.3 exp +2.02% ± 1.03; [-6.00%, +1.25%) n=276.7 exp -0.70% ± 0.62; [-∞, -6.00%) n=17.7 exp -6.47% ± 2.46
* 1_season, holdout_preds_holdout_calibration: **3 distinguishable bands** (best min adjacent z for k = 2..6: 2: 3.0777, 3: 2.38, 4: 1.9014, 5: 1.5813, 6: 1.2771): [+1.50%, ∞) n=97.3 exp +1.54% ± 1.03; [-6.00%, +1.50%) n=283.3 exp -1.29% ± 0.60; [-∞, -6.00%) n=17.0 exp -7.46% ± 2.46
* 3_seasons, dev_preds_dev_calibration: **6 distinguishable bands** (best min adjacent z for k = 2..6: 2: 4.6602, 3: 3.9119, 4: 3.0013, 5: 2.493, 6: 1.9669): [+1.75%, ∞) n=216.0 exp +2.32% ± 0.70; [-0.25%, +1.75%) n=624.0 exp +0.60% ± 0.41; [-1.50%, -0.25%) n=715.0 exp -0.56% ± 0.39; [-3.00%, -1.50%) n=655.0 exp -1.66% ± 0.40; [-6.00%, -3.00%) n=311.0 exp -3.20% ± 0.59; [-∞, -6.00%) n=53.0 exp -6.47% ± 1.42
* 3_seasons, holdout_preds_holdout_calibration: **6 distinguishable bands** (best min adjacent z for k = 2..6: 2: 5.3307, 3: 4.1222, 4: 3.2934, 5: 2.739, 6: 2.2119): [+2.00%, ∞) n=211.0 exp +1.89% ± 0.70; [-0.00%, +2.00%) n=622.0 exp -0.01% ± 0.41; [-1.50%, -0.00%) n=773.0 exp -1.32% ± 0.36; [-3.00%, -1.50%) n=679.0 exp -2.50% ± 0.39; [-6.00%, -3.00%) n=322.0 exp -4.06% ± 0.57; [-∞, -6.00%) n=51.0 exp -7.46% ± 1.42
* 10_seasons, dev_preds_dev_calibration: **9 distinguishable bands** (best min adjacent z for k = 2..6: 2: 8.5084, 3: 7.142, 4: 5.4797, 5: 4.5516, 6: 3.591): [+2.25%, ∞) n=396.7 exp +2.71% ± 0.52; [+0.75%, +2.25%) n=1276.7 exp +1.37% ± 0.29; [-0.00%, +0.75%) n=1493.3 exp +0.49% ± 0.27; [-1.00%, -0.00%) n=2176.7 exp -0.25% ± 0.22; [-2.00%, -1.00%) n=2133.3 exp -1.12% ± 0.22; [-2.75%, -2.00%) n=1530.0 exp -1.85% ± 0.26; [-4.25%, -2.75%) n=1153.3 exp -2.75% ± 0.30; [-6.00%, -4.25%) n=380.0 exp -4.13% ± 0.53; [-∞, -6.00%) n=176.7 exp -6.47% ± 0.78
* 10_seasons, holdout_preds_holdout_calibration: **10 distinguishable bands** (best min adjacent z for k = 2..6: 2: 9.7326, 3: 7.5261, 4: 6.0128, 5: 5.0006, 6: 4.0384): [+3.25%, ∞) n=283.3 exp +2.64% ± 0.60; [+1.25%, +3.25%) n=1053.3 exp +1.19% ± 0.31; [+0.75%, +1.25%) n=950.0 exp +0.20% ± 0.33; [-0.50%, +0.75%) n=2216.7 exp -0.61% ± 0.22; [-1.00%, -0.50%) n=1626.7 exp -1.33% ± 0.25; [-2.25%, -1.00%) n=2393.3 exp -2.03% ± 0.21; [-2.75%, -2.25%) n=1180.0 exp -2.85% ± 0.30; [-4.75%, -2.75%) n=1266.7 exp -3.70% ± 0.28; [-6.00%, -4.75%) n=273.3 exp -5.30% ± 0.61; [-∞, -6.00%) n=170.0 exp -7.46% ± 0.78

### Sizing (first-appearance bets; expected profit = Σ units × CLV)

| scheme | sample | bets | units | unit-weighted CLV ± SE | exp. profit u/season | realized PnL u/season | exp. profit / PnL SD |
|---|---|---|---|---|---|---|---|
| flat_Aplus(>=2.5%) | dev_oof | 89 | 89.0 | +4.16% ± 0.92 | 1.235 | -0.2378 | 0.282 |
| flat_A_and_up(>=1.5%) | dev_oof | 256 | 256.0 | +2.42% ± 0.63 | 2.0682 | 2.7936 | 0.2794 |
| prop_cal_B_and_up | dev_oof | 492 | 350.8308 | +1.40% ± 0.51 | 1.6412 | 3.4717 | 0.1994 |
| kelly_cal_Aplus(1u=1%bankroll,quarter) | dev_oof | 89 | 35.0524 | +3.77% ± 1.06 | 0.4405 | 1.2149 | 0.2532 |
| kelly_cal_A_and_up(quarter) | dev_oof | 256 | 89.7661 | +1.91% ± 0.66 | 0.5705 | 1.0548 | 0.2334 |
| flat_Aplus(>=2.5%) | holdout_descriptive | 163 | 163.0 | +2.34% ± 0.97 | 1.2717 | 12.0422 | 0.1605 |
| flat_A_and_up(>=1.5%) | holdout_descriptive | 292 | 292.0 | +1.03% ± 0.69 | 1.0043 | 2.8643 | 0.1069 |
| prop_cal_B_and_up | holdout_descriptive | 534 | 418.2381 | +1.04% ± 0.60 | 1.4553 | 0.6462 | 0.1305 |
| kelly_cal_Aplus(1u=1%bankroll,quarter) | holdout_descriptive | 163 | 50.3892 | +1.66% ± 0.90 | 0.2786 | 1.9059 | 0.1281 |
| kelly_cal_A_and_up(quarter) | holdout_descriptive | 292 | 83.4697 | +1.10% ± 0.65 | 0.305 | 0.1836 | 0.1252 |

## Q1b. Spread grade v1 (already 6 letters A+..C)

### dev (2020-22)

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 49 | n/a | +2.97% ± 1.24 | 0.0082 | 57% | +26.2% ± 13.3 | score +4.00
| A | 189 | n/a | +1.62% ± 0.74 | 0.014 | 58% | +9.4% ± 6.8 | score +3.00
| B+ | 450 | n/a | -0.40% ± 0.42 | 0.8251 | 42% | +2.5% ± 4.5 | score +2.00
| B | 611 | n/a | -1.17% ± 0.36 | 0.9993 | 38% | +2.0% ± 3.8 | score +1.00
| C+ | 582 | n/a | -1.96% ± 0.38 | 1.0 | 38% | +2.0% ± 3.9 | score +0.00
| C | 402 | n/a | -3.09% ± 0.42 | 1.0 | 31% | -3.0% ± 4.8 | score -1.15

Spearman 1.0; adjacent distinguishable at z >= 1.96: **2 of 5**. A+ vs A: +1.35% (z 1.0571); A vs B+: +2.01% (z 2.9986); B+ vs B: +0.78% (z 1.8181); B vs C+: +0.79% (z 1.721); C+ vs C: +1.12% (z 2.4213)

Per integer score (finest possible; C split): 4: 49 bets +2.97% ± 1.24; 3: 189 bets +1.62% ± 0.74; 2: 450 bets -0.40% ± 0.42; 1: 611 bets -1.17% ± 0.36; 0: 582 bets -1.96% ± 0.38; -1: 365 bets -3.39% ± 0.44; -2: 114 bets -2.00% ± 0.92 (Spearman 0.9643).

CLV on score: +0.99% per point (± +0.14%), residual SD +9.01%, R² 0.021; score points needed between bands: n=50: 3.5582, n=150: 2.0543, n=400: 1.258

Resolution limit (DP over score cuts, linear calibration): 1_season: **2** bands ([-0.5, ∞) n=275.7 exp -1.07% ± 0.54; [-∞, -0.5) n=134.0 exp -3.17% ± 0.78); 3_seasons: **3** bands ([+2.5, ∞) n=200.0 exp +1.04% ± 0.64; [-0.5, +2.5) n=819.0 exp -1.20% ± 0.32; [-∞, -0.5) n=402.0 exp -3.17% ± 0.45); 10_seasons: **5** bands ([+2.5, ∞) n=666.7 exp +1.04% ± 0.35; [+1.5, +2.5) n=1500.0 exp -0.04% ± 0.23; [+0.5, +1.5) n=2036.7 exp -1.03% ± 0.20; [-1.5, +0.5) n=2150.0 exp -2.29% ± 0.19; [-∞, -1.5) n=380.0 exp -4.01% ± 0.46)

Coarse alternatives: A+ | A..B+ | B..C: top 49 +2.97% ± 1.24, mid 474 +0.26% ± 0.44, low 779 -1.72% ± 0.35 (top vs mid: +2.72% (z 2.2154); mid vs low: +1.97% (z 4.4213)) | A+ | rest: top 49 +2.97% ± 1.24, rest 838 -0.74% ± 0.36 (top vs rest: +3.72% (z 3.0374))

### holdout (2023-25, descriptive)

| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| A+ | 49 | n/a | +2.20% ± 1.34 | 0.0496 | 47% | +6.6% ± 13.6 | score +4.00
| A | 168 | n/a | -0.40% ± 0.62 | 0.7409 | 39% | +1.8% ± 7.4 | score +3.00
| B+ | 443 | n/a | -0.74% ± 0.37 | 0.9764 | 40% | +0.3% ± 4.5 | score +2.00
| B | 634 | n/a | -1.03% ± 0.33 | 0.9992 | 39% | +1.2% ± 3.7 | score +1.00
| C+ | 552 | n/a | -1.66% ± 0.36 | 1.0 | 38% | +1.9% ± 4.0 | score +0.00
| C | 442 | n/a | -1.98% ± 0.40 | 1.0 | 32% | +3.5% ± 4.6 | score -1.22

Spearman 1.0; adjacent distinguishable at z >= 1.96: **1 of 5**. A+ vs A: +2.60% (z 1.9948); A vs B+: +0.35% (z 0.5703); B+ vs B: +0.29% (z 0.7367); B vs C+: +0.63% (z 1.5407); C+ vs C: +0.32% (z 0.7339)

Per integer score (finest possible; C split): 4: 49 bets +2.20% ± 1.34; 3: 168 bets -0.40% ± 0.62; 2: 443 bets -0.74% ± 0.37; 1: 634 bets -1.03% ± 0.33; 0: 552 bets -1.66% ± 0.36; -1: 381 bets -2.09% ± 0.42; -2: 155 bets -1.99% ± 0.61 (Spearman 0.9643).

CLV on score: +0.47% per point (± +0.13%), residual SD +8.23%, R² 0.006; score points needed between bands: n=50: 6.8331, n=150: 3.9451, n=400: 2.4159

Resolution limit (DP over score cuts, linear calibration): 1_season: **1** bands ([-∞, ∞) n=285.0 exp -1.20% ± 0.49); 3_seasons: **2** bands ([-0.5, ∞) n=838.0 exp -1.05% ± 0.28; [-∞, -0.5) n=442.0 exp -2.12% ± 0.39); 10_seasons: **4** bands ([+2.5, ∞) n=590.0 exp -0.06% ± 0.34; [+0.5, +2.5) n=2523.3 exp -0.91% ± 0.16; [-0.5, +0.5) n=1840.0 exp -1.54% ± 0.19; [-∞, -0.5) n=1473.3 exp -2.12% ± 0.21)

Coarse alternatives: A+ | A..B+ | B..C: top 49 +2.20% ± 1.34, mid 478 -0.71% ± 0.38, low 791 -1.25% ± 0.33 (top vs mid: +2.91% (z 2.387); mid vs low: +0.54% (z 1.3291)) | A+ | rest: top 49 +2.20% ± 1.34, rest 855 -1.04% ± 0.33 (top vs rest: +3.24% (z 2.4789))

## Q2. Totals grade v2

Dev 2020-22: 40810 allowed-book offers, 40810 candidates, 838 games; mean CLV all offers -4.52%, candidates -4.51%; close has a sharp book 100%. Books: betmgm, betrivers, draftkings, fanduel, williamhill_us.

LOSO R² vs constant: base_ev_sharp 0.0659, ridge_core_a1 0.0649, ridge_core_a300 0.065, ridge_core_a3000 0.065, ridge_market_a1 0.0764, ridge_market_a300 0.0766, ridge_market_a3000 0.0771, ridge_small_a1 0.0756, ridge_small_a300 0.076, ridge_small_a3000 0.0764, ridge_all_a1 0.0756, ridge_all_a300 0.076, ridge_all_a3000 0.0764, lgbm_small_100x4 0.0809, lgbm_small_200x4 0.0882, lgbm_all_100x4 0.0794, lgbm_all_200x4 0.0879. Chosen: **lgbm_small_200x4** (complexity tiers base (raw EV vs sharp fair total) < ridge core < ridge market < ridge small < ridge all < lgbm small < lgbm all; move to the best model of a higher tier only if its LOSO (2020/21/22) game-weighted MSE beats the current choice by > 0.5% relative (rule of grade_v2 / grade_v2_spread)).

Fit on 2020-22 (coef per 1 SD or gain share): ev_sharp 0.5274, is_over 0.1614, log_hours 0.1325, tot_level 0.0747, ev_cons 0.0304, pt_adv_sharp 0.0253, move_mu 0.0162, best_gap 0.014, disp 0.0126, key_right 0.0033, pt_adv_cons 0.0013, move_pts 0.0011, sharp_missing 0.0, on_key 0.0, key_cross 0.0, p_imp 0.0

Univariate offer-level Spearman with CLV (dev): ev_sharp 0.2796, ev_cons 0.2079, pt_adv_sharp 0.2063, pt_adv_cons 0.1675, best_gap -0.1547, key_cross 0.1016, is_over -0.0992, move_mu 0.0916, move_pts 0.0873, p_imp -0.046, overround -0.0184, key_right -0.0144, on_key 0.0101, bk_betrivers -0.0048

Dev band scan (OOF, first bet per game with pred >= cut): -2.0%: 710 bets, CLV +0.12% ± 0.33; -1.5%: 597 bets, CLV +0.65% ± 0.36; -1.0%: 472 bets, CLV +1.24% ± 0.39; -0.5%: 329 bets, CLV +1.80% ± 0.47; +0.0%: 218 bets, CLV +2.26% ± 0.61; +0.5%: 134 bets, CLV +1.73% ± 0.67; +1.0%: 50 bets, CLV +2.35% ± 1.19; +1.5%: 12 bets, CLV +0.81% ± 2.71; +2.0%: 4 bets, CLV -4.69% ± 4.77; +2.5%: 0 bets, CLV n/a; +3.0%: 0 bets, CLV n/a; +4.0%: 0 bets, CLV n/a

### Dev 2020-22 (OOF)

| grade | bets (/season) | pred CLV | realized CLV ± SE | p | beat close | ROI ± SE | CLV vs all-book close | points vs close |
|---|---|---|---|---|---|---|---|---|
| A+ | 218 (72.7) | +0.69% | +2.26% ± 0.61 | 0.0001 | 63% | +14.5% ± 6.3 | +2.27% | +1.15 |
| A | 356 (118.7) | -0.55% | +0.49% ± 0.42 | 0.1194 | 50% | +0.0% ± 5.0 | +0.30% | +0.79 |
| B | 558 (186.0) | -1.52% | -1.31% ± 0.32 | 1.0 | 36% | +10.3% ± 4.0 | -1.35% | +0.52 |
| C | 763 (254.3) | -2.85% | -2.34% ± 0.31 | 1.0 | 32% | +6.5% ± 3.4 | -2.42% | +0.36 |
| S1_Aplus_flat | 218 (72.7) | +0.69% | +2.26% ± 0.61 | 0.0001 | 63% | +14.5% ± 6.3 | +2.27% | +1.15 |
| S2_A_and_up_flat | 472 (157.3) | -0.09% | +1.24% ± 0.39 | 0.0007 | 56% | +5.0% ± 4.3 | +1.12% | +0.94 |

Spearman across letters 1.0; deciles rho 0.9879; offer-level rank corr 0.3054; calibration slope 1.0853; OOS R² 0.0882. A+ minus C +4.60% ± 0.61. Adjacent: A+ vs A: +1.77% (z 2.7354); A vs B: +1.80% (z 4.0259); B vs C: +1.03% (z 2.3535)

Deciles (pred -> realized): -3.90%->-4.68%, -3.32%->-3.54%, -2.98%->-3.32%, -2.72%->-2.53%, -2.45%->-1.82%, -2.10%->-1.38%, -1.74%->-1.46%, -1.26%->-1.34%, -0.70%->+0.42%, +0.39%->+1.58%

By season (A+): 2020: 43 bets +4.07%, 2021: 110 bets +1.84%, 2022: 65 bets +1.76%

### HOLDOUT 2023-25 (frozen, run once)

| grade | bets (/season) | pred CLV | realized CLV ± SE | p | beat close | ROI ± SE | CLV vs all-book close | points vs close |
|---|---|---|---|---|---|---|---|---|
| A+ | 243 (81.0) | +0.89% | +1.48% ± 0.66 | 0.0121 | 52% | +7.5% ± 6.1 | +1.36% | +0.97 |
| A | 410 (136.7) | -0.58% | -1.31% ± 0.43 | 0.9988 | 41% | +0.3% ± 4.7 | -1.50% | +0.48 |
| B | 585 (195.0) | -1.59% | -2.04% ± 0.30 | 1.0 | 35% | -3.5% ± 3.9 | -2.09% | +0.41 |
| C | 826 (275.3) | -2.70% | -2.55% ± 0.28 | 1.0 | 33% | -5.9% ± 3.3 | -2.65% | +0.30 |
| S1_Aplus_flat | 243 (81.0) | +0.89% | +1.48% ± 0.66 | 0.0121 | 52% | +7.5% ± 6.1 | +1.36% | +0.97 |
| S2_A_and_up_flat | 539 (179.7) | -0.03% | -0.26% ± 0.41 | 0.7402 | 45% | +3.1% ± 4.1 | -0.45% | +0.66 |

Spearman across letters 1.0; deciles rho 0.9636; offer-level rank corr 0.2203; calibration slope 0.8098; OOS R² 0.0521. A+ minus C +4.03% ± 0.67. Adjacent: A+ vs A: +2.78% (z 3.8802); A vs B: +0.73% (z 1.4987); B vs C: +0.52% (z 1.4126)

Deciles (pred -> realized): -3.66%->-3.56%, -3.17%->-3.60%, -2.91%->-3.14%, -2.68%->-2.84%, -2.40%->-2.92%, -2.10%->-2.36%, -1.83%->-2.40%, -1.39%->-1.90%, -0.70%->-1.49%, +0.54%->+0.70%

By season (A+): 2023: 74 bets +0.70%, 2024: 73 bets +1.69%, 2025: 96 bets +1.91%

### Holdout baseline: same bands on raw EV vs sharp fair (no model)

| grade | bets (/season) | pred CLV | realized CLV ± SE | p | beat close | ROI ± SE | CLV vs all-book close | points vs close |
|---|---|---|---|---|---|---|---|---|
| A+ | 344 (114.7) | +1.48% | -0.15% ± 0.52 | 0.6134 | 46% | +8.3% ± 5.1 | -0.23% | +0.67 |
| A | 381 (127.0) | -0.59% | -1.54% ± 0.42 | 0.9999 | 37% | -8.5% ± 4.9 | -1.62% | +0.41 |
| B | 532 (177.3) | -1.50% | -2.83% ± 0.30 | 1.0 | 29% | -8.9% ± 4.1 | -2.87% | +0.27 |
| C | 819 (273.0) | -3.11% | -3.29% ± 0.30 | 1.0 | 33% | -8.1% ± 3.3 | -3.37% | +0.15 |
| S1_Aplus_flat | 344 (114.7) | +1.48% | -0.15% ± 0.52 | 0.6134 | 46% | +8.3% ± 5.1 | -0.23% | +0.67 |
| S2_A_and_up_flat | 575 (191.7) | +0.51% | -0.78% ± 0.38 | 0.9796 | 42% | +0.4% ± 4.0 | -0.86% | +0.56 |

Spearman across letters 1.0; deciles rho 0.9515; offer-level rank corr 0.2084; calibration slope 0.6593; OOS R² 0.0293. A+ minus C +3.14% ± 0.61. Adjacent: A+ vs A: +1.39% (z 2.1793); A vs B: +1.29% (z 2.471); B vs C: +0.47% (z 1.0788)

Deciles (pred -> realized): -4.21%->-4.10%, -3.48%->-3.20%, -2.93%->-3.42%, -2.48%->-3.41%, -2.29%->-3.05%, -2.07%->-3.00%, -1.62%->-2.52%, -1.06%->-1.79%, -0.40%->-2.12%, +1.55%->+0.09%

By season (A+): 2023: 111 bets +0.01%, 2024: 105 bets -0.48%, 2025: 128 bets -0.02%

**Verdict (pre-registered bar): ADOPT** -- monotone [True], A+ CLV > 0 at p < 0.05 [True].

### Post-hoc (after the one-shot holdout; descriptive, not part of the test): what is totals A+?

* dev_in_sample_fit: A+ by side: over 9 bets, CLV +2.87% ± 1.34, ROI +6.1% ± 33.6; under 219 bets, CLV +2.83% ± 0.60, ROI +19.3% ± 6.2. By hours before kick: (0, 24] 8 bets, CLV +1.18% ± 1.43, ROI +44.0% ± 31.4; (120, 200] 149 bets, CLV +2.74% ± 0.74, ROI +21.4% ± 7.5; (24, 72] 41 bets, CLV +2.88% ± 0.93, ROI +16.7% ± 14.8; (72, 120] 30 bets, CLV +3.64% ± 1.99, ROI +1.8% ± 17.7. Blind early-week (>= 96 h) under at the best allowed price: 797 bets, CLV -0.64% ± 0.35, ROI +5.0% ± 3.3. Early under with EV >= 0 vs the sharp fair: 118 bets, CLV +2.30% ± 0.83, ROI +16.7% ± 8.5; A+ bets outside that rule: 110 bets, CLV +3.33% ± 0.79, ROI +18.4% ± 8.9.
* holdout: A+ by side: over 11 bets, CLV -0.97% ± 1.40, ROI -13.5% ± 30.0; under 232 bets, CLV +1.59% ± 0.68, ROI +8.5% ± 6.2. By hours before kick: (0, 24] 10 bets, CLV +0.77% ± 0.95, ROI -4.5% ± 31.9; (120, 200] 188 bets, CLV +1.81% ± 0.80, ROI +12.5% ± 6.9; (24, 72] 16 bets, CLV -1.33% ± 1.18, ROI -16.4% ± 24.5; (72, 120] 29 bets, CLV +1.13% ± 1.67, ROI -7.2% ± 18.2. Blind early-week (>= 96 h) under at the best allowed price: 853 bets, CLV -1.21% ± 0.34, ROI -0.6% ± 3.3. Early under with EV >= 0 vs the sharp fair: 122 bets, CLV +2.55% ± 0.90, ROI +10.5% ± 8.6; A+ bets outside that rule: 121 bets, CLV +0.20% ± 0.95, ROI +4.5% ± 8.7.

## Reading

**Q1 -- finer letters do not help.**

* Moneyline, 9 letters chosen on dev: realized CLV is still roughly ordered (Spearman 0.98 dev, 0.97 on 2023-25), but
  only 1 of 8 adjacent pairs is statistically distinguishable in either period (C vs C-), and 2023-25 shows two
  inversions (B+ -0.3% below B +0.8%; C+ = C). The 4 production letters: 2 of 3 adjacent pairs distinguishable in both
  periods (A+ vs A and B vs C; A vs B never). Equal-count 9-iles: 3 of 8.
* Why: a bet's realized CLV has a residual SD of ~10% (longshot prices), while the bet-level calibration slope is
  ~0.9. Two bands need a predicted-CLV gap of 1.96*sqrt(2)*SD/(slope*sqrt(n)) = 2.6% at 150 bets per band, 1.6% at 400.
  The useful (positive) range of predicted CLV is only about 0..+3.5%, so it holds ONE clean step (A+ vs the rest)
  plus the C tail. The DP resolution limit (optimistic: assumes the dev calibration line holds) is 3 bands for one
  season of bets, 6 over three seasons -- but only 2 of those 6 lie above 0% predicted CLV; the rest split the
  negative (never-bet) region.
* Recommendation: keep the 4 moneyline bands (A+ >= 2.5%, A >= 1.5%, B >= 0.5%, C); do not add +/-. Read them as
  3 tiers: A+ = bet, A/B = no edge shown (2023-25: +0.1% / -0.1%), C = avoid. If anything, a 5th band could split C
  (C vs C- is the one robust extra split), which has no betting value.
* Sizing: flat 1 unit on A+ only. Among the schemes (flat A+, flat A-and-up, units proportional to calibrated
  predicted CLV from B up, quarter-Kelly on calibrated edge), flat A+ had the best expected-profit-to-PnL-risk ratio
  in dev (0.28) and on 2023-25 (0.16; quarter-Kelly 0.25 / 0.13, proportional 0.20 / 0.13). Kelly mostly shrinks
  longshot stakes, which costs as much CLV as it saves variance; proportional sizing adds ~0-CLV A/B bets. The
  grade cannot rank within A+ (A+ split at 3.0%: +2.8% vs +1.8%, z 0.7), so predicted-CLV-scaled stakes are not
  supported.
* Spread grade v1 (6 letters): ranked in both periods (rho 1.0) but adjacent letters are mostly indistinguishable:
  2 of 5 pairs in 2020-22, 1 of 5 in 2023-25 (A+ vs A, z 2.0). CLV per score point fell from +1.0% (dev) to +0.5%
  (2023-25). Resolution: 3 bands over 3 seasons in dev, 2 in 2023-25. Its letters are best read as A+ (+2.2..3.0%)
  vs the rest; a 3-tier display (A+ / A..B+ / B..C) is the most the data supports.

**Q2 -- totals grade v2: passes its pre-registered bar, and it is a Tuesday-under detector.**

* Chosen model (rule fixed before looking): LightGBM on the 'small' feature set (LOSO R2 0.088 vs 0.066 for raw EV
  vs the sharp fair total). Gain: EV vs sharp 53%, over/under 16%, hours before kick 13%, total level 7%; key
  numbers, juice, book and dispersion add ~nothing. Bands (dev): A+ >= 0.0% predicted, A >= -1%, B >= -2%, else C.
* 2023-25 (frozen, one run): A+ 243 bets (81/season) CLV +1.48% +- 0.66 (p 0.012), +0.7 / +1.7 / +1.9% by season,
  same vs the all-book close (+1.36%), +0.97 points vs the closing number; ROI +7.5% +- 6.1 (noise). A -1.3%, B -2.0%,
  C -2.6%; letters rho 1.0; A+ minus C +4.0% +- 0.7. Raw EV vs sharp with the same bands fails (A+ -0.15%), so the
  model's side/timing terms carry the edge. Dev was stronger (A+ +2.3%) and the holdout calibration slope is 0.81.
* What A+ is (post-hoc): 95% unders, median 139 h before kick (Tuesday 14:10 UTC). Totals drift down during the week
  (both periods), so an early under at or better than the sharp fair beats the close: early unders with EV >= 0 vs the
  sharp fair were +2.3% (dev, 118) and +2.6% +- 0.9 (2023-25, 122), and every one of them was A+; the other A+ bets were
  +0.2% on 2023-25. Blind early unders at the best price were -1.2%. The earlier totals study's soft-vs-sharp rule (both
  sides) failed on 2023-25 because overs do not share the drift.
* Recommendation: totals grade v2 may be shown as a LABEL (like moneyline grade v2): A+ only means something. Do
  not start a track from this result alone; if a totals track is wanted, pre-register the simple rule (Tuesday/early
  under at the best allowed price with EV >= 0 vs LowVig/BetOnline) and paper-trade it in 2026, judged on CLV.
* Caveats: 2023-25 totals were used once before (totals.md), and that report (Tuesday unders cheaper than overs;
  soft-vs-sharp failed) was known when this study was designed; the user-specified feature list included over/under
  and hours. The candidate set was widened to all quotes after the first dev run (dev-only decision, before the
  freeze). Production needs each game's totals snapshot history (first-seen fair total) and all books' quotes; the
  research implied-total inversion is a grid version of TotalDist.implied_mu (max 0.03 points apart).

