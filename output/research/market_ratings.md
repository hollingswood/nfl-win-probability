# Market-implied power ratings: what moves them, where the market over/under-reacts

Script: `scripts/research/market_ratings.py` (dev / freeze / holdout / report). Raw numbers: `output/research/market_ratings.json`. Frozen rules: `output/research/market_ratings_frozen.json`.

## Verdict

- **The market's week-to-week updating is close to efficient.** Lines move about 0.05-0.06 points per point of last-game result surprise (margin minus closing spread). The weight that would have been correct is 0.03-0.05. The next-game ATS coefficient is -0.007 to -0.020 in every block, and never significant.
- **The classic hypothesis is not supported.** The market does not underweight efficiency. In the components-only regression, it slightly OVER-weights one-game EPA surprise. It already discounts return/defensive TDs and garbage-time points, and if anything it discounts garbage time too much. Turnover luck over-reaction is too small to bet.
- **All 3 frozen rules failed the 2023-25 holdout** (Bonferroni α = 0.0167 each). R1 (primetime under-reaction) and R2 (bounce after an ATS blowout) were positive in every block but not significant. R3 (look-ahead reversion, early-line CLV) was flat.
- **One structural fact replicated out of sample.** The part of the look-ahead line that deviates from the rating-implied line partly reverts by the close (slope about -0.10 to -0.13, t about 4 in both dev and holdout). It is too small to beat the price after vig.

## Method

- **Ratings.** A Kalman random-walk filter on 32 team ratings plus HFA, observing each week's closing lines (`spread_line`, 1999-2026). Tuned on one-step-ahead LINE prediction MSE, 2003-19 only: s_obs 1.5, q_week 1.0, season carry-over rho 0.6, q_off 2.0.
- **Prior.** For every game, `prior` is the line implied by ratings before either team's last result. From it: `adj = close - prior`, `ats = margin - close` and `true = margin - prior`, with **true = adj + ats**. Regressing all three on the same last-game components (home minus away) gives the market's weight, its error and the correct weight. Coefficients add exactly.
- **Components of each team's previous game.** All are team perspective. In the JSON, the M2/M3 regressions under `test1_dev_2003_2019` use 2012-19 only, because pbp starts in 2012:
  - ATS surprise.
  - Efficiency-EPA surprise: net pass/rush EPA, turnover plays removed, residualised on the line (2012+ pbp).
  - INT net and fumble-recovery luck (lost minus 0.5 × fumbles).
  - Non-offensive TDs net and garbage-time points net (Q4, home WP < 0.1 or > 0.9).
  - Close-game W/L, OT, ATS × primetime, ATS × public team, and the excess ATS beyond 14.
- **Sample.** Core sample excludes games where either starting QB changed from the team's last game, which is the 'injuries known' control. SEs are clustered by season-week.
- **Early lines (2020-25).** The look-ahead line is the last snapshot before either team's previous kickoff, from the main and openers files, within 4 days of the re-open. The early (re-open) line is the first snapshot after both previous games ended (+4h), median 158 h before kickoff. Both use the price-implied consensus mu. The close is `closing_fair` mu_close_all. CLV is valued at the best allowed-book offer.

## Test 1: what moves market ratings, and is the move the right size?

Coefficients are points of line per unit of (home minus away) last-game component. 'adj' is the market's move, 'ats' is the leftover error (negative means over-reaction) and 'true' is the correct weight. Format is coefficient ± clustered SE.

| block (n) | component | adj (market) | ats (error) | true |
|---|---|---|---|---|
| dev 2003-19 (3366) | d_ats | +0.060 ± 0.002 | -0.006 ± 0.013 | +0.054 ± 0.013 |
| dev 2020-22 (588) | d_ats | +0.050 ± 0.006 | -0.020 ± 0.029 | +0.030 ± 0.029 |
| HOLDOUT 2023-25 (617) | d_ats | +0.047 ± 0.005 | -0.011 ± 0.028 | +0.036 ± 0.028 |
| dev 2003-19 | d_close_wl | -0.040 ± 0.043 | -0.426 ± 0.269 | -0.466 ± 0.269 |
| dev 2003-19 | d_ats_prime | -0.008 ± 0.005 | +0.066 ± 0.031 | +0.058 ± 0.031 |
| dev 2003-19 | d_ats_big | -0.024 ± 0.009 | -0.013 ± 0.058 | -0.037 ± 0.058 |
| dev 2020-22 | d_close_wl | -0.084 ± 0.096 | +0.864 ± 0.574 | +0.780 ± 0.556 |
| dev 2020-22 | d_ats_prime | -0.008 ± 0.014 | +0.185 ± 0.077 | +0.177 ± 0.075 |
| dev 2020-22 | d_ats_big | -0.033 ± 0.025 | -0.041 ± 0.123 | -0.073 ± 0.130 |
| HOLDOUT 2023-25 | d_close_wl | -0.011 ± 0.078 | +0.919 ± 0.530 | +0.908 ± 0.552 |
| HOLDOUT 2023-25 | d_ats_prime | +0.000 ± 0.012 | +0.095 ± 0.069 | +0.095 ± 0.071 |
| HOLDOUT 2023-25 | d_ats_big | -0.040 ± 0.027 | -0.019 ± 0.119 | -0.060 ± 0.120 |

**Components only** (2012-19 dev uses pbp; same spec in 2020-22 and in the holdout):

| block | component | adj (market) | ats (error) | true |
|---|---|---|---|---|
| dev 2012-19 | d_eff_s | +0.061 ± 0.004 | -0.019 ± 0.022 | +0.042 ± 0.022 |
| dev 2012-19 | d_int_net | +0.189 ± 0.029 | +0.205 ± 0.159 | +0.394 ± 0.164 |
| dev 2012-19 | d_fum_luck | +0.136 ± 0.048 | -0.132 ± 0.298 | +0.004 ± 0.297 |
| dev 2012-19 | d_fum_cnt_net | +0.158 ± 0.053 | +0.537 ± 0.335 | +0.695 ± 0.344 |
| dev 2012-19 | d_ntd_net | +0.024 ± 0.054 | -0.009 ± 0.370 | +0.014 ± 0.374 |
| dev 2012-19 | d_garb_net | -0.015 ± 0.007 | +0.031 ± 0.046 | +0.016 ± 0.045 |
| dev 2020-22 | d_eff_s | +0.041 ± 0.006 | -0.010 ± 0.038 | +0.032 ± 0.038 |
| dev 2020-22 | d_int_net | +0.192 ± 0.046 | +0.094 ± 0.315 | +0.286 ± 0.315 |
| dev 2020-22 | d_fum_luck | +0.027 ± 0.089 | -1.069 ± 0.585 | -1.042 ± 0.601 |
| dev 2020-22 | d_fum_cnt_net | +0.239 ± 0.101 | -0.302 ± 0.518 | -0.063 ± 0.568 |
| dev 2020-22 | d_ntd_net | -0.003 ± 0.101 | -0.401 ± 0.805 | -0.404 ± 0.809 |
| dev 2020-22 | d_garb_net | +0.001 ± 0.012 | +0.059 ± 0.060 | +0.061 ± 0.062 |
| HOLDOUT | d_eff_s | +0.041 ± 0.005 | -0.043 ± 0.030 | -0.002 ± 0.030 |
| HOLDOUT | d_int_net | +0.191 ± 0.046 | +0.213 ± 0.255 | +0.403 ± 0.255 |
| HOLDOUT | d_fum_luck | +0.180 ± 0.098 | +0.294 ± 0.535 | +0.474 ± 0.541 |
| HOLDOUT | d_fum_cnt_net | +0.107 ± 0.097 | -0.213 ± 0.572 | -0.106 ± 0.592 |
| HOLDOUT | d_ntd_net | +0.093 ± 0.120 | +0.292 ± 0.644 | +0.385 ± 0.687 |
| HOLDOUT | d_garb_net | -0.002 ± 0.010 | +0.075 ± 0.059 | +0.073 ± 0.059 |

**How to read it:**
- **Efficiency.** Market weight on one-game efficiency surprise is 0.06 / 0.04 / 0.04 against a correct weight of 0.04 / 0.03 / 0.00. That is a mild over-reaction to single-game EPA, the opposite of the classic claim.
- **Interceptions.** INT net is moved 0.19 per INT in every era, while 0.29-0.40 would have been correct. That is a consistent under-reaction, but no block is significant.
- **Garbage time.** The market discounts garbage-time points (it reacts about -0.02 at the re-open), yet garbage-time points carry positive signal: ATS +0.03 / +0.06 / +0.08. It is a lead, not significant.
- **Non-offensive TDs.** These are already removed from the move: adj is -0.19 per return TD when holding the score fixed.

## Test 2: multi-week dynamics (ATS residual vs close)

| test | dev 2003-19 | dev 2020-22 | holdout |
|---|---|---|---|
| 3 straight rating rises (±1) | +0.257 ± 0.469 | -0.519 ± 1.006 | -0.993 ± 0.898 |
| rating change over 3 games | -0.037 ± 0.074 | -0.199 ± 0.154 | +0.300 ± 0.149 |
| extreme rating (beyond ±6) | -0.007 ± 0.202 | -0.287 ± 0.428 | +0.538 ± 0.498 |
| rating level | +0.093 ± 0.035 | -0.025 ± 0.076 | +0.164 ± 0.083 |
| season-to-date ATS (ctrl rating) | -0.054 ± 0.036 | -0.107 ± 0.074 | -0.002 ± 0.086 |
| season-to-date efficiency (ctrl rating) | -0.113 ± 0.057 | -0.081 ± 0.065 | -0.090 ± 0.073 |
| eliminated-proxy team, wk 12+ | -0.687 ± 0.479 | -0.671 ± 1.298 | -2.046 ± 1.314 |
| line adjustment (close - prior) | -0.098 ± 0.072 | -0.264 ± 0.120 | +0.296 ± 0.147 (holdout CONTAMINATED, see disclosure) |
| wk 1-4: offseason rating move | +0.096 ± 0.101 | +0.041 ± 0.140 | -0.322 ± 0.333 |
| wk 1-4: last-yr point diff beyond rating | +0.028 ± 0.090 | -0.055 ± 0.117 | +0.237 ± 0.269 |

**Findings:**
- Momentum, regression of extreme ratings and season-start priors show nothing stable. No preseason win totals are in the repo, so offseason rating moves and last season's point differential stand in for them.
- The eliminated-team proxy (cannot finish at least .500) is negative in every block, but its cover-rate rule loses in dev. Means and cover rates disagree.
- Fading the week's line adjustment looked good in dev (-0.10, -0.26) and reversed in 2023-25 (+0.30). It was not frozen; see the disclosure.

## Test 3: look-ahead → re-open → close (2020-25)

| quantity | dev 2020-22 | holdout 2023-25 |
|---|---|---|
| games / with look-ahead line | 751 / 295 | 768 / 629 |
| mean abs re-open (early - look-ahead), pts | 1.338 | 1.089 |
| mean abs drift (close - early), pts | 1.346 | 1.124 |
| re-open per pt of result surprise | +0.049 ± 0.004 (R² 0.3782) | +0.049 ± 0.003 |
| drift per pt of result surprise | +0.003 ± 0.003 (R² 0.001) | +0.006 ± 0.004 |
| correct weight (margin - look-ahead) | +0.045 ± 0.048 (R² 0.0044) | +0.040 ± 0.028 |
| ATS vs price close | -0.021 ± 0.029 (R² 0.0009) | -0.012 ± 0.028 |
| drift on look-ahead deviation from rating line | -0.126 ± 0.030 | -0.103 ± 0.028 |
| drift on efficiency surprise (M2) | +0.001 ± 0.006 | +0.019 ± 0.005 |
| re-open on garbage-time pts (M2) | -0.013 ± 0.011 | -0.022 ± 0.006 |

**Findings:**
- The adjustment to a result happens almost entirely at the re-open: about 0.049 per point, with R² about 0.4 from the result alone. It is the right size: the correct weight is 0.040-0.045. Little result-driven drift remains afterwards.
- What does drift is the look-ahead line's own deviation from the rating-implied line, which partly reverts. This replicated in the holdout (all three seasons negative).
- **New in the holdout only.** After the re-open the line keeps moving toward efficiency (EPA) surprise: drift +0.019 per EPA point, t about 3.7, against about 0 in dev. It was not pre-registered, so it is a lead for paper tracking only.

## Frozen rules and the one-time holdout

Frozen before the holdout (2026-10-01). α = 0.0167 each.

| rule | dev 2003-19 | dev 2020-22 | HOLDOUT 2023-25 | pass |
|---|---|---|---|---|
| R1 primetime under-reaction (S ≥ 12), closing ATS | 243-206-18 (54.1%), ROI +5.9% ± 4.5 | 40-28-1 (58.8%), ROI +14.4% ± 11.6 | 48-39-1 (55.2%), ROI +5.3% ± 10.1, p 0.301 | False |
| R2 bounce after ATS miss ≥ 21, closing ATS | 199-161-10 (55.3%), ROI +7.2% ± 5.0 | 37-23-0 (61.7%), ROI +20.4% ± 12.4 | 33-25-2 (56.9%), ROI +8.6% ± 12.1, p 0.2455 | False |
| R3 look-ahead reversion, early best price, CLV | n/a | LOSO: 30 bets, CLV +1.78% (t +1.04, p 0.148), cover 70.0%, ROI +34.7% ± 16.4 | 62 bets, CLV -0.37% (t -0.39, p 0.652), cover 46.7%, ROI -10.1% ± 12.1 | False |

**Same side at the early line** (best allowed price):
- R1 holdout: 88 bets, CLV -2.35% (t -2.88, p 0.998), cover 52.9%, ROI +1.3% ± 10.1.
- R2 holdout: 60 bets, CLV -3.12% (t -3.82, p 1.000), cover 58.6%, ROI +12.2% ± 12.1.
- Both signals pay at the close, not by moving the line. Betting them early costs about 2-3% CLV against a -4.5% blind baseline.

**Pooled across dev and holdout** (selection was made on dev, so this is optimistic):
- R1: 331-273, 54.8%, z = 1.19 against 52.4%.
- R2: 269-209, 56.3%, z = 1.71.

**Disclosure.** During development a per-season table of the coefficient of next-game ATS residual on the raw line adjustment (close - rating-implied prior) was printed for ALL seasons incl. 2023-25 (+0.33, +0.47, +0.12 vs dev about -0.1/-0.26). The 'fade the line adjustment' rule is therefore NOT frozen and its 2023-25 numbers are contaminated/descriptive only. No other holdout quantity was looked at before this freeze.

## Paper-track recommendation

Nothing qualifies as an edge, and nothing should get real money. R2, and less so R1, can go on a zero-stake paper track at the close with actual juice, exactly as frozen. Each produces 20-30 bets a season. That is monitoring, not a test that will resolve soon. Separating a true 55% from 52.4% at 2 SE needs about 1,400 bets, which is decades at this volume.

- **R1.** In a regular-season game where neither team's starting QB changed from its previous game: let s = (previous-game margin minus its closing spread) if that game was primetime (Thu, Mon, or Sun at 7pm ET or later), else 0. Bet home ATS if s_home - s_away ≥ 12. Bet away if it is ≤ -12.
- **R2.** Same QB filter. Back the team that missed its previous closing spread by 21 or more, unless both teams did.

Post-hoc leads to re-test on 2026+ only:
- Garbage-time points are over-discounted.
- In the 2023+ market, drift continues toward efficiency surprise.
- INT margin is under-weighted.
