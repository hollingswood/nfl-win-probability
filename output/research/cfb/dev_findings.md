# College football: first model checks, development seasons only (2026-10-03)

No holdout seasons were examined (closing-line tests hold out 2022-25; opener tests hold out 2024-25).
Data: CollegeFootballData.com (data/cfb/raw). Closing lines 2014+; opening lines only 2021+. FBS vs FBS games only.

## Model
Walk-forward ratings (scripts/research/cfb/model.py, fast.py):
- **Margin ridge:** shrunk toward a preseason prior: 0.69 × last season + 3.35 × talent z-score + 0.30 × last season × returning-production deviation. Coefficients fit on 2015-20.
- **Points offense/defense ridge.**
- **Per-play EPA (PPA) offense/defense ridge.**

## Results (dev)
| check | result |
|---|---|
| Margin MAE, model vs close (2019-21, calibrated) | 13.10 vs 12.49: the close is far better |
| Model picks with \|model − close\| ≥ 3, cover rate (2014-21) | 47-48% (the model leans heavily to underdogs; mean-based edges ≠ cover edges) |
| Opening line MAE vs close (2021-23) | 12.37 vs 12.21; mean \|open→close move\| 1.43 pts |
| corr(model − open, close − open) | 0.09: weak |
| Bets at open when \|model − open\| ≥ 5 | 595 bets, line moved our way 44% / against 45%, cover vs open 52.0%: below the vig |
| Totals model vs close (2014-21) | MAE 13.62 vs 13.15; no side edge |

## Takeaway
A public-data ratings model does not beat college football closing lines, and only faintly anticipates opener moves. This mirrors the NFL.

The next tests are the price-based ones that worked in the NFL. They need The Odds API college odds:
- historical, 2021-25 + 2026 to date: plan `cfb`, ~35k credits a season;
- live logging, started 2026-10-03.

Tests to run:
- soft books vs Pinnacle;
- moneyline vs sharp spread;
- opener drift;
- line shopping across books.
