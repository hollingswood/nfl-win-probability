# Alternative college football rating models (development seasons only)

Fit on 2014-18 (3833 FBS games), validated on 2019-21 (2078). 2022-25 not examined.

| model | MAE vs result | MAE of close | info beyond close (coef on model−close) | \|pred−close\|≥3: bets, cover | ≥5: bets, cover |
|---|---|---|---|---|---|
| margin_ridge | 13.10 | 12.49 | +0.065 | 1024, 0.482 | 548, 0.482 |
| elo | 13.25 | 12.49 | +0.074 | 1154, 0.498 | 724, 0.496 |
| points | 13.80 | 12.49 | +0.045 | 1315, 0.477 | 858, 0.480 |
| ppa | 13.84 | 12.49 | -0.030 | 1289, 0.483 | 865, 0.485 |
| success | 14.02 | 12.49 | +0.039 | 1413, 0.497 | 1032, 0.500 |
| explosive | 16.61 | 12.49 | -0.957 | 1659, 0.489 | 1506, 0.493 |
| rush_pass | 13.92 | 12.49 | -0.004 | 1346, 0.495 | 948, 0.485 |
| line | 14.81 | 12.49 | +0.071 | 1593, 0.501 | 1273, 0.497 |
| downs | 13.80 | 12.49 | -0.007 | 1281, 0.488 | 886, 0.489 |
| talent_ret | 15.43 | 12.49 | +0.021 | 1628, 0.491 | 1365, 0.493 |
| all_linear | 12.92 | 12.49 | +0.177 | 1010, 0.510 | 536, 0.473 |
| all_gbm | 13.23 | 12.49 | +0.071 | 1190, 0.500 | 747, 0.490 |

Residual model (LightGBM predicting result − close from every feature + the close): validation corr +0.033; top 20% most confident picks cover 0.531 (216-191).

Totals (points + pace + PPA + success + explosiveness sums, fit 2014-18): MAE 13.19 vs close 12.93; |pred−close|≥3: 930 bets, hit 0.517.

## Opening lines (fit 2021-22, validate 2023)

| model | corr(model − open, close − open) | \|model−open\|≥5: bets, line moved our way / against, cover vs open |
|---|---|---|
| margin_ridge | +0.259 | 167, 51% / 41%, 0.503 |
| elo | +0.119 | 246, 43% / 48%, 0.502 |
| points | +0.182 | 252, 50% / 43%, 0.520 |
| ppa | +0.193 | 325, 55% / 38%, 0.508 |
| success | +0.154 | 365, 54% / 39%, 0.496 |
| explosive | -0.013 | 539, 47% / 46%, 0.513 |
| rush_pass | +0.176 | 355, 55% / 39%, 0.501 |
| line | +0.106 | 461, 50% / 42%, 0.532 |
| downs | +0.193 | 313, 54% / 38%, 0.516 |
| talent_ret | +0.032 | 469, 47% / 46%, 0.491 |
| all_linear | +0.267 | 187, 54% / 38%, 0.551 |
