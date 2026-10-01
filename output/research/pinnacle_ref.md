# Pinnacle as the sharp reference (2024-25, descriptive)

2024-25 overlaps the 2023-25 holdout already used once; this is a measurement study, not a validation. No thresholds were tuned. Script: `scripts/research/pinnacle_ref.py`.

## Coverage

- Sharp books actually present in the main odds files: betonlineag, lowvig (circasports and bookmaker never appear, so 'sharp median' = BetOnline/LowVig, one book group).
- All main snapshots at Pinnacle request times (any horizon, incl. lines weeks ahead; per-game coverage is in Q1 headers): 7596; Pinnacle missing 54.54% (sharp missing 58.24%). By type: fri2140: n=1965, pin missing 67.23%, sharp missing 71.86%; other: n=4940, pin missing 57.13%, sharp missing 60.75%; t75: n=691, pin missing 0.0%, sharp missing 1.59%
- Median quote age (snapshot - last_update, min): {'pinnacle': 0.4, 'lowvig': 0.7, 'betonlineag': 0.6, 'draftkings': 0.7, 'fanduel': 0.4}

## Q1: accuracy at the same snapshots

### fri2140 (games=507; latest snapshot <=7d before kick: Pinnacle missing 7.21%, sharp missing 9.84% of 569 games)

Joint regression (nflverse close - all-book) on (pin - all-book, sharp - all-book): {'b_pin': 0.289, 'se_pin': 0.171, 'b_sharp': -0.131, 'se_sharp': 0.19}

| target (RMSE, pp) | pinnacle | sharp median | all-book | avg pin+sharp | n | slope b (se) |
|---|---|---|---|---|---|---|
| nflverse_close | 2.365 | 2.393 | 2.32 | 2.354 | 507 | 0.64 (0.15) |
| own_close_all | 2.061 | 2.092 | 2.047 | 2.047 | 507 | 0.633 (0.131) |
| own_close_sharp | 2.245 | 2.271 | 2.284 | 2.231 | 500 | 0.619 (0.143) |
| own_close_pinnacle | 2.143 | 2.206 | 2.2 | 2.147 | 507 | 0.78 (0.137) |

Log loss on results (n=507): pinnacle 0.59245, sharp 0.59336, all-book 0.59302, avg 0.59288, nflverse close 0.58857. Pin - sharp = -0.0009 ± 0.00062; pin - all-book = -0.00056 ± 0.00061. Mean |pin - sharp| = 0.532 pp.

Slope b: regression of (target - sharp) on (pinnacle - sharp) through the origin; b=1 means the target moved all the way to Pinnacle, b=0 means Pinnacle adds nothing beyond the sharp median. 'other' = a Pinnacle request time that was another game's T-75 (e.g. Sunday 15:45 UTC for a 20:20 kickoff). At t75 the own-close targets are the same snapshot, so they are not forecasts.

### other (games=553; latest snapshot <=7d before kick: Pinnacle missing 65.03%, sharp missing 1.41% of 569 games)

Joint regression (nflverse close - all-book) on (pin - all-book, sharp - all-book): {'b_pin': 0.048, 'se_pin': 0.135, 'b_sharp': 0.241, 'se_sharp': 0.154}

| target (RMSE, pp) | pinnacle | sharp median | all-book | avg pin+sharp | n | slope b (se) |
|---|---|---|---|---|---|---|
| nflverse_close | 1.961 | 1.922 | 1.882 | 1.913 | 553 | 0.327 (0.123) |
| own_close_all | 1.643 | 1.596 | 1.587 | 1.585 | 553 | 0.328 (0.101) |
| own_close_sharp | 1.829 | 1.775 | 1.857 | 1.771 | 545 | 0.28 (0.114) |
| own_close_pinnacle | 1.732 | 1.754 | 1.789 | 1.711 | 553 | 0.589 (0.11) |

Log loss on results (n=553): pinnacle 0.59986, sharp 0.59968, all-book 0.60043, avg 0.59974, nflverse close 0.59881. Pin - sharp = 0.00018 ± 0.00058; pin - all-book = -0.00057 ± 0.0006. Mean |pin - sharp| = 0.521 pp.

Slope b: regression of (target - sharp) on (pinnacle - sharp) through the origin; b=1 means the target moved all the way to Pinnacle, b=0 means Pinnacle adds nothing beyond the sharp median. 'other' = a Pinnacle request time that was another game's T-75 (e.g. Sunday 15:45 UTC for a 20:20 kickoff). At t75 the own-close targets are the same snapshot, so they are not forecasts.

### t75 (games=560; latest snapshot <=7d before kick: Pinnacle missing 0.53%, sharp missing 1.58% of 569 games)

Joint regression (nflverse close - all-book) on (pin - all-book, sharp - all-book): {'b_pin': -0.124, 'se_pin': 0.07, 'b_sharp': -0.154, 'se_sharp': 0.074}

| target (RMSE, pp) | pinnacle | sharp median | all-book | avg pin+sharp | n | slope b (se) |
|---|---|---|---|---|---|---|
| nflverse_close | 1.306 | 1.288 | 1.041 | 1.245 | 560 | 0.455 (0.072) |
| own_close_all | 0.675 | 0.638 | 0.008 | 0.547 | 560 | 0.453 (0.032) |
| own_close_sharp | 0.727 | 0.028 | 0.638 | 0.364 | 560 | 0.001 (0.002) |
| own_close_pinnacle | 0.0 | 0.727 | 0.676 | 0.363 | 560 | 1.0 (0.0) |

Log loss on results (n=560): pinnacle 0.59885, sharp 0.59799, all-book 0.59858, avg 0.59839, nflverse close 0.59652. Pin - sharp = 0.00086 ± 0.00064; pin - all-book = 0.00026 ± 0.00062. Mean |pin - sharp| = 0.554 pp.

Slope b: regression of (target - sharp) on (pinnacle - sharp) through the origin; b=1 means the target moved all the way to Pinnacle, b=0 means Pinnacle adds nothing beyond the sharp median. 'other' = a Pinnacle request time that was another game's T-75 (e.g. Sunday 15:45 UTC for a 20:20 kickoff). At t75 the own-close targets are the same snapshot, so they are not forecasts.

## Q2: C2 / C4 at Pinnacle snapshots, by fair-probability reference

| rule / ref | bets | CLV nflverse (t) | beat | CLV own close all (t) | CLV own close sharp (t) | CLV own close Pinnacle (t) | ROI ± SE | snaps |
|---|---|---|---|---|---|---|---|---|
| C2_ml_sharp_dog|sharp | 73 | +0.0203 (+1.24) | 0.62 | +0.0216 (+1.35) | +0.0212 (+1.26) | +0.0257 (+1.60) | +0.351 ± 0.231 | {'other': 60, 't75': 7, 'fri2140': 6} |
| C4_ml_sharp_and_model|sharp | 87 | +0.0147 (+1.66) | 0.61 | +0.0307 (+4.82) | +0.0356 (+4.87) | +0.0383 (+5.36) | -0.044 ± 0.190 | {'other': 43, 'fri2140': 22, 't75': 22} |
| C2_ml_sharp_dog|pin | 79 | +0.0139 (+1.10) | 0.61 | +0.0224 (+1.87) | +0.0209 (+1.63) | +0.0359 (+3.11) | +0.207 ± 0.211 | {'other': 65, 't75': 11, 'fri2140': 3} |
| C4_ml_sharp_and_model|pin | 109 | +0.0189 (+2.51) | 0.62 | +0.0308 (+5.68) | +0.0269 (+4.18) | +0.0491 (+8.80) | -0.038 ± 0.175 | {'other': 52, 't75': 33, 'fri2140': 24} |
| C2_ml_sharp_dog|avg | 66 | +0.0141 (+0.76) | 0.62 | +0.0223 (+1.24) | +0.0217 (+1.16) | +0.0337 (+1.92) | +0.432 ± 0.247 | {'other': 53, 't75': 7, 'fri2140': 6} |
| C4_ml_sharp_and_model|avg | 91 | +0.0203 (+2.37) | 0.63 | +0.0370 (+6.13) | +0.0359 (+5.06) | +0.0551 (+8.95) | -0.012 ± 0.200 | {'other': 45, 't75': 27, 'fri2140': 19} |
| C2_ml_sharp_dog|sharp_all_snapshots | 119 | +0.0235 (+1.81) | 0.64 | +0.0283 (+2.27) | +0.0252 (+1.95) | +0.0320 (+2.56) | +0.238 ± 0.172 | {'other': 111, 't75': 5, 'fri2140': 3} |
| C4_ml_sharp_and_model|sharp_all_snapshots | 109 | +0.0119 (+1.46) | 0.57 | +0.0240 (+3.89) | +0.0266 (+3.84) | +0.0334 (+4.77) | -0.088 ± 0.167 | {'other': 87, 't75': 13, 'fri2140': 9} |
| C4_sharp_only_bets | 22 | -0.0084 (-0.64) | 0.36 | +0.0026 (+0.36) | +0.0039 (+0.43) | -0.0121 (-1.38) | -0.550 ± 0.255 | {'other': 9, 't75': 7, 'fri2140': 6} |
| C4_pin_only_bets | 44 | +0.0094 (+1.00) | 0.55 | +0.0127 (+2.33) | -0.0060 (-1.02) | +0.0357 (+5.56) | -0.297 ± 0.252 | {'other': 19, 't75': 17, 'fri2140': 8} |

CLV vs own close Pinnacle is circular for the Pinnacle reference (and vs own close sharp for the sharp reference), and at t75 the own close IS the bet snapshot. CLV vs nflverse close is the neutral yardstick.

C4 overlap (game, side) between sharp- and Pinnacle-referenced selections: {'sharp_only': 22, 'pin_only': 44, 'both': 65}
