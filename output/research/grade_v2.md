# Grade v2: predicted CLV as the bet grade (moneyline)

Code `scripts/research/grade_v2.py` (dev -> freeze -> holdout once -> report). Numbers `grade_v2.json`; frozen model, thresholds and strategies `grade_v2_frozen.json` (frozen 2026-10-01T22:13:59, before the 2023-25 run). Caveat: 2023-25 was already the holdout of earlier studies (edge_lab, ml_spread_consistency -> v4); nothing here was tuned on it, but the v4 feature itself was proposed after earlier dev work on 2020-22.

**Verdict: ADOPT grade v2** (pre-registered bar: realized CLV rising with grade [True] and A+ CLV > 0 at one-sided p < 0.05 [True]).

## Model

Candidates: 27870 offer rows / 832 games (dev). LOSO (2020/21/22) game-weighted MSE, R² vs constant: base_ev_ml_sharp 0.0811, ridge_a1 0.0925, ridge_a30 0.0925, ridge_a300 0.0926, ridge_a3000 0.0899, ridge_a30000 0.0716, lgbm_mono_200x4 0.1033, lgbm_mono_400x7 0.0898. Chosen: **lgbm_mono_200x4** (lowest LOSO game-weighted MSE; ridge kept unless a GBM beats the best ridge by > 0.5% relative MSE).

Feature importance (share of total split gain, fit on 2020-22): ev_ml_sharp 51.7%, p_imp 17.6%, ev_sp_sharp 11.6%, ev_cons 5.6%, log_hours 4.8%, ev_model 2.7%, move_p 1.8%, move_pts 1.6%, best_gap 1.4%, model_elig 0.6%, on3 0.3%, is_dog 0.1%, key_pos 0.1%, disp 0.1%; all other features 0. Production: `lightgbm.Booster(model_str=frozen['model']['model_string'])`, features in `feature_order`.


Thresholds (pred CLV): A+ >= +2.50%, A >= +1.50%, B >= +0.50%, else C.

## Dev 2020-22 (out-of-fold predictions)

By grade (one bet per game per grade, first snapshot where the game's best offer has that grade):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± boot SE | CLV vs Pinnacle close |
|---|---|---|---|---|---|---|---|
| A+ | 89 (29.7/season) | +3.01% | +4.16% ± 0.92 | 0.0 | 78% | -0.80% ± 14.82 | n/a (n=0) |
| A | 203 (67.7/season) | +1.95% | +1.53% ± 0.70 | 0.0151 | 66% | +5.03% ± 9.62 | n/a (n=0) |
| B | 395 (131.7/season) | +0.92% | +0.55% ± 0.52 | 0.1439 | 56% | -1.18% ± 6.66 | n/a (n=0) |
| C | 827 (275.7/season) | -1.30% | -0.77% ± 0.42 | 0.9655 | 46% | +3.30% ± 5.05 | n/a (n=0) |

Spearman across letters: rho 1.0; across pred-CLV deciles (snapshot best offers): rho 0.9515 (p 0.0); offer-level rank corr 0.4322; calibration slope 1.0283.

Deciles (pred -> realized CLV): -5.03%->-4.26%, -2.83%->-2.71%, -2.20%->-2.19%, -1.76%->-1.83%, -1.32%->-1.44%, -0.89%->-0.58%, -0.47%->+0.42%, +0.00%->-0.10%, +0.62%->-0.22%, +1.96%->+2.08%

Strategies and existing rules:

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± boot SE | CLV vs Pinnacle close |
|---|---|---|---|---|---|---|---|
| S1_Aplus_flat | 89 (29.7/season) | +3.01% | +4.16% ± 0.92 | 0.0 | 78% | -0.80% ± 14.82 | n/a (n=0) |
| S2_A_and_up_flat | 256 (85.3/season) | +2.26% | +2.42% ± 0.63 | 0.0001 | 71% | +3.27% ± 8.62 | n/a (n=0) |
| S3_B_and_up_prop | 492 (164.0/season) | +1.83% | +1.47% ± 0.50 | 0.0015 | 60% | +2.83% ± 7.45 | n/a (n=0) |
| rule v2 | 125 (41.7/season) | n/a | +4.14% ± 0.64 | 0.0 | 76% | +14.51% ± 16.55 | n/a (n=0) |
| rule v3 | 124 (41.3/season) | n/a | +4.19% ± 0.64 | 0.0 | 75% | +17.17% ± 16.76 | n/a (n=0) |
| rule v4 | 101 (33.7/season) | n/a | +2.43% ± 0.77 | 0.0009 | 63% | -1.50% ± 13.85 | n/a (n=0) |

## HOLDOUT 2023-25 (frozen model, run once)

By grade (one bet per game per grade, first snapshot where the game's best offer has that grade):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± boot SE | CLV vs Pinnacle close |
|---|---|---|---|---|---|---|---|
| A+ | 163 (54.3/season) | +3.15% | +2.34% ± 0.98 | 0.0086 | 60% | +22.16% ± 14.77 | +2.94% ± 1.07 (n=132) |
| A | 206 (68.7/season) | +1.96% | +0.09% ± 0.73 | 0.4519 | 55% | -11.65% ± 10.49 | +1.03% ± 0.85 (n=152) |
| B | 432 (144.0/season) | +0.92% | -0.09% ± 0.50 | 0.5737 | 49% | -7.75% ± 7.13 | +0.45% ± 0.60 (n=313) |
| C | 851 (283.7/season) | -1.13% | -1.30% ± 0.40 | 0.9995 | 42% | -0.35% ± 4.82 | -0.95% ± 0.48 (n=567) |

Spearman across letters: rho 1.0; across pred-CLV deciles (snapshot best offers): rho 0.9394 (p 0.0001); offer-level rank corr 0.4581; calibration slope 1.0656.

Deciles (pred -> realized CLV): -4.94%->-6.45%, -2.74%->-3.20%, -2.08%->-2.44%, -1.62%->-2.46%, -1.21%->-0.97%, -0.78%->-1.13%, -0.33%->-1.23%, +0.20%->-0.44%, +0.90%->+0.14%, +2.58%->+1.54%

Strategies and existing rules:

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± boot SE | CLV vs Pinnacle close |
|---|---|---|---|---|---|---|---|
| S1_Aplus_flat | 163 (54.3/season) | +3.15% | +2.34% ± 0.98 | 0.0086 | 60% | +22.16% ± 14.77 | +2.94% ± 1.07 (n=132) |
| S2_A_and_up_flat | 292 (97.3/season) | +2.49% | +1.03% ± 0.71 | 0.0719 | 55% | +2.94% ± 9.69 | +2.03% ± 0.81 (n=219) |
| S3_B_and_up_prop | 534 (178.0/season) | +2.07% | +1.11% ± 0.51 | 0.0152 | 51% | +0.88% ± 8.68 | +1.39% ± 0.59 (n=379) |
| rule v2 | 145 (48.3/season) | n/a | +2.46% ± 0.58 | 0.0 | 67% | +0.88% ± 15.17 | +3.34% ± 0.70 (n=109) |
| rule v3 | 149 (49.7/season) | n/a | +2.42% ± 0.56 | 0.0 | 67% | -1.83% ± 15.12 | +3.44% ± 0.68 (n=113) |
| rule v4 | 162 (54.0/season) | n/a | +1.56% ± 0.65 | 0.0082 | 52% | +11.10% ± 13.65 | +2.01% ± 0.76 (n=125) |

Holdout baseline -- same grade cuts applied to raw EV vs sharp no-vig ML (no model):

| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± boot SE | CLV vs Pinnacle close |
|---|---|---|---|---|---|---|---|
| A+ | 300 (100.0/season) | +4.95% | +0.98% ± 0.80 | 0.1107 | 55% | +4.51% ± 10.30 | +2.68% ± 0.87 (n=220) |
| A | 240 (80.0/season) | +1.99% | -0.15% ± 0.72 | 0.5816 | 50% | +3.13% ± 10.67 | +0.03% ± 0.79 (n=180) |
| B | 451 (150.3/season) | +0.96% | +0.04% ± 0.48 | 0.4691 | 50% | +5.10% ± 7.00 | +0.11% ± 0.57 (n=328) |
| C | 838 (279.3/season) | -1.73% | -1.77% ± 0.37 | 1.0 | 39% | -3.16% ± 4.64 | -1.30% ± 0.45 (n=554) |

Holdout by grade and season (CLV):

* A+: 2023: +1.63% (31), 2024: +1.41% (53), 2025: +3.25% (79)
* A: 2023: -0.82% (54), 2024: +0.47% (71), 2025: +0.36% (81)
* B: 2023: -0.01% (119), 2024: -1.18% (135), 2025: +0.67% (178)
* C: 2023: -1.10% (284), 2024: -1.60% (285), 2025: -1.20% (282)

Strategies (frozen):

* **S1_Aplus_flat** -- One bet per game at the first snapshot where the best offer is A+ (pred CLV >= A+ cut); 1 unit.
* **S2_A_and_up_flat** -- One bet per game at the first snapshot where the best offer is A or A+; 1 unit.
* **S3_B_and_up_prop** -- One bet per game at the first snapshot where the best offer is B or better; units = pred CLV / 2% clipped to 0.25..2.

## Post-hoc (after the one-shot holdout; descriptive, not part of the test): A+ vs the live v2 rule

* A+ bets that v2 also took (80 games): 80 bets, CLV +3.40% ± 0.74, ROI +30.76% ± 21.99
* A+ bets v2 did not take: 83 bets, CLV +1.32% ± 1.79, ROI +13.87% ± 19.70
* v2 bets that were not A+ bets: 65 bets, CLV +2.11% ± 1.07, ROI -36.28% ± 20.37
* v2 bets by the grade of their own offer at bet time: A: 37 bets, CLV +0.77% ± 0.82, ROI -10.90% ± 28.33; A+: 59 bets, CLV +3.16% ± 0.67, ROI +31.29% ± 24.35; B: 14 bets, CLV +3.24% ± 1.31, ROI -18.21% ± 43.88; C: 35 bets, CLV +2.76% ± 1.85, ROI -30.29% ± 34.28

## Reading

* The pre-registered bar is met: realized holdout CLV rises with the letter (A+ > A > B > C, rho 1.0; decile rho
  0.94) and A+ CLV is positive at p < 0.01, confirmed by the Pinnacle close on 2024-25. Grade v1 never ranked bets;
  grade v2 does, so it should replace v1 as the DISPLAYED grade (GRADING_VERSION 2).
* But the separation is at the ends only: A and B realized ~0% CLV on 2023-25 (dev: +1.5% / +0.6%). Only A+ is a
  bet; A and B are 'lean / no bet'. 'Grade-scaled stakes' (S3) and 'A and up' (S2) dilute A+ with ~zero-CLV bets.
* A+ is not a better bet selector than the live v2 rule: S1 = CLV +2.3% on 54/season vs v2 +2.5% on 48/season
  (same holdout, v2 SE smaller). Post-hoc, half the A+ bets are v2 bets and those carry the edge; A+ bets outside v2
  were +1.3% ± 1.8. The model's top feature is the same EV-vs-sharp-ML that defines v2 (gain 52%), then price level
  (18%) and the sharp-spread EV of v4 (12%). Line move, hours-to-kick, model EV, best-price gap add a little;
  key number, dispersion, book and the signal count add nothing.
* ROI by grade is noise at these sample sizes (A+ ROI +22% ± 15); judge by CLV. Do not add a new bet track from this
  and do not use the grade to veto v2/v3/v4 bets: post-hoc, v2 bets on games A+ never flagged still had positive
  CLV (+2.1% ± 1.1; the C-graded ones were +650 longshots, +2.8% ± 1.9). Use grade v2 as the label / confidence
  display, and A+ as the only 'bet' letter if it is ever used for selection.
* Caveats: 2023-25 was the holdout of earlier studies; Pinnacle could not be a model input (no 2020-22 data); the
  historical QB-change flag uses the actual starter and was dropped as leakage; spread offers were not graded.

