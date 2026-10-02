# Moneyline vs spread consistency, with a total-aware key-number margin model

Code: `scripts/research/ml_spread_consistency.py` (stages part1 -> part1b -> p2dev -> p3dev -> freeze -> holdout -> report), reusable model `scripts/research/margin_by_total.py`. Numbers: `ml_spread_consistency.json`; frozen rules: `ml_spread_consistency_frozen.json` (frozen 2026-10-01T20:53:57, before any 2023-25 moneyline / teaser or 2024-25 alt-line evaluation in this study). Caveat: the 2023-25 seasons were already used as holdout by earlier studies (teasers_v2, buy_points, edge_lab); the rules here were not tuned on them.

## Verdict

1. **Total-aware key numbers: no out-of-sample gain.** Kernel-raked weights w(|margin|; total) do pick up the classic
   pattern in the fit years (1999-2019: w3 = 3.33 at total 38 vs 2.96 at 50; w7 rises slightly with the total), but
   most of it is era (low totals = 1999-2011). On 2020-2025 the total-aware model is no better than the same model
   without the total (exact-margin log loss -0.0007 ± 0.0006 (2012-19 fit; 1999-2019 fit -0.0001 ± 0.0008), exact-3 -0.0004 ± 0.0003, exact-7 -0.0002 ± 0.0003, cover outcomes at ±0.5 around
   3/7/10/14 +0.0000 ± 0.0001 nats/game; positive = total-aware better). Actual 2020-25 exact 3s went the *other* way: high totals
   (>47) had 97 vs ~75 predicted, low totals (<=41) 40 vs 42-47. Exact 7 is over-predicted by every model fit before
   2020 (131 actual vs 155-163; only the old 2012-14 weights, 135, are close) -- a post-2020 drift no total
   adjustment fixes. LOSO picked sigma constant in the total (sigma slope 0) and no spread dependence.
2. **Moneyline vs spread: markets are consistent to ~0.5 pp.** ML-implied and spread-implied (total-aware) win
   probabilities have equal log loss at every window (diff ≤ 0.0012 ± 0.0008); the ML market prices favourites
   0.2-0.9 pp richer than the spread at every total (no total-dependent mispricing); each allowed book's ML sits within
   ~±1 pp of its own spread (up to 2 pp on 10+ point favourites; vs a 4-5% ML hold), so same-book inconsistency is never bettable on its own. What is
   real: when the sharp (LowVig/BetOnline) spread implies a different P(win) than the sharp ML, the ML moves toward
   the spread by close (slope 0.2-0.6 dev, 0.1-0.25 holdout, Sun t≈3). Combined with soft-book ML prices, that
   produced two rules that held up on 2023-25 by CLV (see table); the total>=44 version failed.
3. **Teasers / buys re-priced: unchanged verdicts.** Total-aware leg probabilities differ from the single-weight
   model by 0.2-1.4 pp, and not systematically by total (the difference is mostly sigma), so no teaser or buy flips
   sign. Frozen T1 (Fri Wong legs, total <= 44):
   65 teasers, ROI +6.3% ± 11.2, model EV at close -1.8%; T2 (pre-kick): 43 teasers, ROI -16.1% ± 13.9; B1 (buy onto 3/7 at the close, total <= 43.5, real alt prices): 86 bets, ROI -8.1% ± 9.4 -- model EV
   -3.9%. Historical 'low-total Wong dogs win more' (2006-2019 closes: 0.80 at totals <= 41 vs 0.70 at > 47) did not
   repeat in 2023-25 (0.71 vs 0.78) and the margin model does not reproduce it either way.

## Moneyline holdout (2023-2025, evaluated once)

| rule | dev 2020-22 bets / CLV / ROI | holdout bets | avg ML | win (close-implied) | ROI ± SE | CLV vs sharp ML close ± SE | CLV vs Pinnacle close (n) | CLV vs sharp spread close | per season CLV |
|---|---|---|---|---|---|---|---|---|---|
| ML1_sun_dog_blend1 | 68 / +2.4% ± 0.7 / +0.1% ± 18.1 | 125 | +275 | 0.2903 (0.2904) | +3.7% ± 15.2 | +1.0% ± 0.4 | +1.6% ± 0.4 (100) | +6.6% ± 0.5 | 2023: +0.9% (25), 2024: +0.4% (32), 2025: +1.4% (68) |
| ML2_early_dog_blend1_tot44 | 135 / +3.1% ± 0.7 / +8.0% ± 12.5 | 149 | +240 | 0.2973 (0.3003) | +6.9% ± 14.0 | -1.2% ± 0.8 | -0.9% ± 0.9 (119) | +3.5% ± 0.8 | 2023: -0.1% (30), 2024: -2.9% (44), 2025: -0.6% (75) |
| ML3_early_spread_gap2 | 102 / +2.4% ± 0.8 / -1.5% ± 13.9 | 163 | +270 | 0.3148 (0.2976) | +11.0% ± 13.4 | +1.6% ± 0.7 | +2.0% ± 0.8 (126) | +6.5% ± 0.7 | 2023: +2.1% (37), 2024: -0.8% (43), 2025: +2.6% (83) |

Rules (exact):

* **ML1_sun_dog_blend1** -- Sunday 14:10 UTC snapshot. Dog (consensus spread > 0) moneyline at an allowed book, price <= +400, with EV >= +1% vs fair = mean(sharp spread-implied P(win) [lowvig/betonlineag spread + juice through the total-aware margin model], sharp no-vig ML) (fallback: consensus spread-implied). Best-EV book, one bet per game.
* **ML2_early_dog_blend1_tot44** -- Same bet at the FIRST of the Tue 14:10 / Fri 21:40 / Sun 14:10 UTC snapshots where it qualifies, only when the consensus total at the snapshot is >= 44 (dog ML value in higher-scoring games).
* **ML3_early_spread_gap2** -- Pure ML-vs-spread inconsistency: first Tue/Fri/Sun snapshot where an allowed book's ML (either side) has EV >= +2% vs the SHARP SPREAD-implied P(win) (total-aware model) AND EV >= 0 vs the sharp no-vig ML.

Pre-registered test: CLV - 2·SE > 0 AND ROI > 0 = edge; CLV t >= 2 = paper-track. ML1 and ML3 pass the letter (CLV t 2.7 and 2.4, ROI positive but ±13-15%), ML2 (total >= 44) fails (CLV -1.2%). Context: the best-priced allowed-book dog ML at the same Sunday snapshot, unselected, has CLV -2.5% (holdout), so the selection lifts CLV ~3.5 pts; ML1/ML3 overlap on 88 games. CLV fell from dev (+2.4%/+2.4%) to holdout (+1.0%/+1.6%), and ML1 is a close cousin of the live moneyline v2/v3 tracks (soft price vs sharp no-vig); its new ingredient is the sharp spread-implied P(win) in the fair price. Dog CLV relies on proportional de-vig of the closing sharp ML; the Pinnacle close (2024-25) agrees. The 'vs sharp spread close' column is inflated for dogs (the ML market prices favourites 0.5-0.9 pp richer than the spread in 2023-25), so it is shown for completeness, not as evidence.

## 1. Margin model (fit 1999-2019 and 2012-2019; validated 2020-2025)

Chosen by leave-season-block-out log likelihood: 1999-2019 {'sigma0': 14.2, 'sigma_slope': 0.0, 'h': 4.0, 'smooth': 80.0, 'mode': 'abs', 'loso_loglik': -3.84633}; 2012-2019 {'sigma0': 13.8, 'sigma_slope': 0.0, 'h': 2.5, 'smooth': 80.0, 'mode': 'abs', 'loso_loglik': -3.87353}. Used for parts 2-3: **total_aware_2012** (lower 2020-2022 exact-margin log loss of total_aware (1999-2019) vs total_aware_2012).

Key-number weights of the 1999-2019 total-aware fit by total:

| total | 0 | 3 | 6 | 7 | 10 | 14 |
|---|---|---|---|---|---|---|
| 38.0 | 0.0736 | 3.334 | 1.237 | 2.063 | 1.451 | 1.48 |
| 41.0 | 0.075 | 3.201 | 1.271 | 2.128 | 1.411 | 1.453 |
| 44.0 | 0.0755 | 3.068 | 1.303 | 2.17 | 1.383 | 1.43 |
| 47.0 | 0.0746 | 2.977 | 1.317 | 2.201 | 1.373 | 1.423 |
| 50.0 | 0.0735 | 2.959 | 1.305 | 2.227 | 1.361 | 1.424 |
| 53.0 | 0.0733 | 3.01 | 1.285 | 2.22 | 1.352 | 1.436 |

Out-of-sample log loss by model (nats/game; lower is better):

| period | model | exact | exact3 pred/act | exact7 pred/act | e3 ll | e7 ll | cover ll (9 lines) |
|---|---|---|---|---|---|---|---|
| 2020-2022 | total_aware | 3.853 | 122/130 | 81/62 | 0.4317 | 0.2643 | 0.5939 |
| 2020-2022 | pooled_1999_2019 | 3.853 | 125/130 | 78/62 | 0.4306 | 0.2634 | 0.5942 |
| 2020-2022 | old_spread_rules | 3.845 | 105/130 | 66/62 | 0.4343 | 0.2612 | 0.5946 |
| 2020-2022 | teasers_v2_single | 3.842 | 114/130 | 76/62 | 0.4319 | 0.2628 | 0.5937 |
| 2020-2022 | total_aware_2012 | 3.849 | 114/130 | 77/62 | 0.4323 | 0.2631 | 0.5937 |
| 2020-2022 | pooled_2012_2019 | 3.848 | 114/130 | 76/62 | 0.4318 | 0.2628 | 0.5937 |
| 2023-2025 | total_aware | 3.822 | 127/127 | 82/69 | 0.417 | 0.2818 | 0.6093 |
| 2023-2025 | pooled_1999_2019 | 3.823 | 130/127 | 81/69 | 0.4166 | 0.2808 | 0.6093 |
| 2023-2025 | old_spread_rules | 3.828 | 109/127 | 69/69 | 0.4185 | 0.2796 | 0.6095 |
| 2023-2025 | teasers_v2_single | 3.82 | 118/127 | 79/69 | 0.4169 | 0.2804 | 0.6093 |
| 2023-2025 | total_aware_2012 | 3.823 | 118/127 | 78/69 | 0.4172 | 0.2806 | 0.6093 |
| 2023-2025 | pooled_2012_2019 | 3.822 | 118/127 | 79/69 | 0.4169 | 0.2805 | 0.6093 |
| 2020-2025 | total_aware | 3.837 | 249/257 | 163/131 | 0.4243 | 0.2731 | 0.6017 |
| 2020-2025 | pooled_1999_2019 | 3.837 | 255/257 | 159/131 | 0.4235 | 0.2722 | 0.6018 |
| 2020-2025 | old_spread_rules | 3.837 | 213/257 | 135/131 | 0.4263 | 0.2705 | 0.6021 |
| 2020-2025 | teasers_v2_single | 3.831 | 232/257 | 155/131 | 0.4243 | 0.2717 | 0.6016 |
| 2020-2025 | total_aware_2012 | 3.836 | 231/257 | 155/131 | 0.4247 | 0.2719 | 0.6016 |
| 2020-2025 | pooled_2012_2019 | 3.835 | 233/257 | 155/131 | 0.4243 | 0.2717 | 0.6016 |

Exact 3 / 7 by closing total, 2020-2025 (actual vs predicted):

| total | n | act3 | pred3_total_aware | pred3_total_aware_2012 | pred3_teasers_v2_single | act7 | pred7_total_aware | pred7_total_aware_2012 | pred7_teasers_v2_single |
|---|---|---|---|---|---|---|---|---|---|
| <=41 | 296 | 40 | 46.88 | 42.38 | 41.14 | 33 | 27.43 | 26.19 | 27.33 |
| 41.5-44 | 420 | 57 | 62.13 | 58.05 | 57.23 | 23 | 39.9 | 36.67 | 38.18 |
| 44.5-47 | 428 | 63 | 60.99 | 56.46 | 58.01 | 31 | 41.22 | 38.96 | 38.76 |
| 47.5-50 | 318 | 56 | 45.2 | 42.09 | 43.71 | 18 | 31.38 | 31.01 | 29.09 |
| >50 | 231 | 41 | 33.85 | 32.46 | 32.27 | 26 | 23.09 | 22.54 | 21.42 |

Same, fit period 1999-2019 (in-sample for the total-aware fit):

| total | n | act3 | pred3_total_aware | pred3_pooled | act7 | pred7_total_aware | pred7_pooled |
|---|---|---|---|---|---|---|---|
| <=41 | 2070 | 347 | 327.2 | 313.4 | 177 | 189.1 | 195.4 |
| 41.5-44 | 1322 | 198 | 195.7 | 198.3 | 138 | 126 | 123.9 |
| 44.5-47 | 1137 | 148 | 162.5 | 170 | 103 | 109.9 | 106.3 |
| 47.5-50 | 643 | 91 | 90.65 | 96.25 | 68 | 63.06 | 60.16 |
| >50 | 411 | 59 | 58.49 | 61.09 | 40 | 39.97 | 38.24 |

## 2. Moneyline vs spread, descriptive

Log loss of the home-win outcome by reference and window (dev 2020-22 | holdout 2023-25):

| window | n | p_ml_cons | p_sp_cons | p_sp1_cons | p_ml_sharp | p_sp_sharp | sp_minus_ml_logloss | sp_minus_ml_se |
|---|---|---|---|---|---|---|---|---|
| tue | 799 | 0.6194 | 0.6194 | 0.6196 | 0.6151 | 0.6107 | 0.0001 | 0.0006 |
| fri | 777 | 0.6114 | 0.6106 | 0.6107 | 0.6157 | 0.6179 | -0.0007 | 0.0006 |
| sun | 736 | 0.6135 | 0.6135 | 0.6136 | 0.6131 | 0.6124 | 0 | 0.0006 |
| kick | 834 | 0.6066 | 0.6061 | 0.6062 | 0.6057 | 0.606 | -0.0005 | 0.0006 |

| window | n | p_ml_cons | p_sp_cons | p_sp1_cons | p_ml_sharp | p_sp_sharp | p_pin | sp_minus_ml_logloss | sp_minus_ml_se |
|---|---|---|---|---|---|---|---|---|---|
| tue | 854 | 0.6131 | 0.6136 | 0.6133 | 0.6113 | 0.612 |  | 0.0005 | 0.0007 |
| fri | 790 | 0.6098 | 0.6109 | 0.6107 | 0.6079 | 0.6098 | 0.597 | 0.0012 | 0.0008 |
| sun | 745 | 0.6134 | 0.614 | 0.6139 | 0.6116 | 0.6138 |  | 0.0006 | 0.0009 |
| kick | 854 | 0.6088 | 0.609 | 0.6088 | 0.6085 | 0.6094 | 0.5997 | 0.0002 | 0.0008 |

Favourite P(win) at the last snapshot, ML (no-vig consensus) vs spread-implied, by number (dev | holdout):

| fav_by | n | p_ml | p_sp_total_aware | p_sp_single | actual | se |
|---|---|---|---|---|---|---|
| 0-1 | 50 | 0.514 | 0.5127 | 0.5137 | 0.5 | 0.0707 |
| 1.5-2 | 60 | 0.5388 | 0.5352 | 0.536 | 0.4667 | 0.0644 |
| 2.5 | 70 | 0.5645 | 0.561 | 0.5618 | 0.6232 | 0.0597 |
| 3 | 125 | 0.5939 | 0.5915 | 0.5925 | 0.52 | 0.0439 |
| 3.5 | 75 | 0.6264 | 0.6254 | 0.6266 | 0.5867 | 0.0559 |
| 4-6 | 142 | 0.6686 | 0.6649 | 0.6661 | 0.6071 | 0.0398 |
| 6.5 | 58 | 0.7152 | 0.7123 | 0.7134 | 0.7759 | 0.0593 |
| 7 | 53 | 0.7362 | 0.7356 | 0.7364 | 0.7692 | 0.0611 |
| 7.5 | 35 | 0.7503 | 0.7533 | 0.754 | 0.7714 | 0.0732 |
| 8-9.5 | 57 | 0.7745 | 0.7766 | 0.7772 | 0.8772 | 0.0554 |
| 10+ | 113 | 0.8352 | 0.8288 | 0.8295 | 0.8584 | 0.0349 |

| fav_by | n | p_ml | p_sp_total_aware | p_sp_single | actual | se |
|---|---|---|---|---|---|---|
| 0-1 | 46 | 0.5141 | 0.5123 | 0.5133 | 0.587 | 0.0737 |
| 1.5-2 | 79 | 0.5371 | 0.5331 | 0.5342 | 0.5823 | 0.0561 |
| 2.5 | 97 | 0.5667 | 0.562 | 0.5629 | 0.567 | 0.0503 |
| 3 | 129 | 0.5959 | 0.593 | 0.5939 | 0.6667 | 0.0432 |
| 3.5 | 84 | 0.6282 | 0.6242 | 0.6256 | 0.6905 | 0.0527 |
| 4-6 | 165 | 0.6765 | 0.668 | 0.6688 | 0.6909 | 0.0364 |
| 6.5 | 38 | 0.7198 | 0.7132 | 0.7142 | 0.8108 | 0.0738 |
| 7 | 47 | 0.7452 | 0.7368 | 0.7373 | 0.7447 | 0.0636 |
| 7.5 | 38 | 0.7625 | 0.754 | 0.7547 | 0.5789 | 0.069 |
| 8-9.5 | 49 | 0.7863 | 0.7771 | 0.7781 | 0.7551 | 0.0586 |
| 10+ | 83 | 0.8509 | 0.832 | 0.8328 | 0.8916 | 0.0391 |

By total (dev | holdout): the ML-minus-spread gap does not depend on the total.

| total | n | gap_ml_minus_sp | p_ml | p_sp | actual |
|---|---|---|---|---|---|
| <=41 | 110 | 0.0021 | 0.6515 | 0.6494 | 0.7037 |
| 41.5-44 | 182 | 0.0016 | 0.6732 | 0.6716 | 0.6319 |
| 44.5-47 | 215 | 0.0023 | 0.6822 | 0.68 | 0.6761 |
| 47.5-50 | 164 | 0.004 | 0.6649 | 0.6609 | 0.6768 |
| >50 | 167 | 0.0028 | 0.6531 | 0.6504 | 0.6168 |

| total | n | gap_ml_minus_sp | p_ml | p_sp | actual |
|---|---|---|---|---|---|
| <=41 | 201 | 0.0058 | 0.6541 | 0.6484 | 0.6318 |
| 41.5-44 | 230 | 0.0067 | 0.664 | 0.6573 | 0.6783 |
| 44.5-47 | 219 | 0.0078 | 0.6589 | 0.6512 | 0.7248 |
| 47.5-50 | 140 | 0.0086 | 0.6594 | 0.6507 | 0.7071 |
| >50 | 65 | 0.005 | 0.6458 | 0.6408 | 0.6769 |

## 3. Teasers and point buys re-priced

Wong legs (Fri + last snapshot, book numbers) win rate vs total-aware and single-weight predictions, dev 2020-22 | holdout 2023-25:

| key | n | actual | pw_sharp | pw_single |
|---|---|---|---|---|
| <=41 | 99 | 0.8283 | 0.7371 | 0.7397 |
| 41.5-44 | 140 | 0.7429 | 0.7351 | 0.7373 |
| 44.5-47 | 192 | 0.7917 | 0.7353 | 0.7395 |
| >47 | 260 | 0.7615 | 0.7371 | 0.7389 |

| key | n | actual | pw_sharp | pw_single |
|---|---|---|---|---|
| <=41 | 211 | 0.7014 | 0.7356 | 0.7386 |
| 41.5-44 | 228 | 0.7588 | 0.7359 | 0.7384 |
| 44.5-47 | 177 | 0.7232 | 0.7363 | 0.739 |
| >47 | 206 | 0.7233 | 0.7363 | 0.7381 |

Wong legs at nflverse closes, 2006-2019 | 2023-2025:

| key | n | actual | pw_sharp | pw_single |
|---|---|---|---|---|
| ('dog', '<=41') | 86 | 0.8023 | 0.7324 | 0.7328 |
| ('dog', '41.5-44') | 95 | 0.7579 | 0.7312 | 0.7314 |
| ('dog', '44.5-47') | 98 | 0.7959 | 0.7346 | 0.7352 |
| ('dog', '>47') | 95 | 0.6947 | 0.7364 | 0.7353 |
| ('fav', '<=41') | 44 | 0.7045 | 0.7274 | 0.7273 |
| ('fav', '41.5-44') | 38 | 0.7368 | 0.7294 | 0.7292 |
| ('fav', '44.5-47') | 56 | 0.8036 | 0.7274 | 0.7274 |
| ('fav', '>47') | 70 | 0.7 | 0.7279 | 0.7272 |

| key | n | actual | pw_sharp | pw_single |
|---|---|---|---|---|
| ('dog', '<=41') | 34 | 0.7059 | 0.7398 | 0.7402 |
| ('dog', '41.5-44') | 58 | 0.8448 | 0.7383 | 0.7382 |
| ('dog', '44.5-47') | 41 | 0.7805 | 0.7388 | 0.7393 |
| ('dog', '>47') | 55 | 0.7818 | 0.7356 | 0.7353 |
| ('fav', '<=41') | 18 | 0.6111 | 0.7343 | 0.7341 |
| ('fav', '41.5-44') | 16 | 0.625 | 0.7328 | 0.7324 |
| ('fav', '44.5-47') | 14 | 0.7143 | 0.7321 | 0.7319 |
| ('fav', '>47') | 18 | 0.5556 | 0.7371 | 0.7361 |

Frozen teaser / buy rules, holdout:

| rule | n | roi | model EV | desc |
|---|---|---|---|---|
| T1_fri_wong_le44 | 65 | +6.3% ± 11.2 | -1.8% | Friday 21:40 UTC: Wong legs AT THE BOOK's number (+1.5..+2.5 / -7.5..-8.5) at DK/FD/MGM/CZR, consensus total <= 44; best-EV same-book pairs (total-aware leg probs) at the book's fixed 2-team price, no EV threshold, up to 3 disjoint pairs per week. |
| T2_kick_wong_le44 | 43 | -16.1% ± 13.9 | -2.8% | As T1 at each game's last snapshot (~75 min pre-kick). |
| B1_close_buy_onto_3or7_tot_le43_5 | 86 | -8.1% ± 9.4 | -3.9% | ~75-min close: buy onto 3 or 7 (+2.5->+3, -3.5->-3, +6.5->+7, -7.5->-7) at the same book's alt price, consensus total <= 43.5; best allowed book by total-aware EV; no EV filter; one per game. |

Alt-line calibration, close, 2023 dev (low total = consensus <= 43.5): bought 1-2 pts in low totals won 0.583 vs 0.558 predicted -- the hint behind B1 -- but in 2024-25 the same cell was 0.558 vs 0.557 and the excess moved to high totals (0.572 vs 0.557); the total-aware and single models predict the same (±0.1 pt).

