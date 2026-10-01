# Factor combinations vs the closing market (spread and total)

Question: do combinations of situational/game factors predict where games finish relative to the closing spread (`y_side` = home margin − spread_line) or total (`y_tot` = total − total_line), even though the individual factors don't?

Code: `scripts/research/interactions.py` (stages dev → freeze → holdout once → report). Numbers: `output/research/interactions.json`. Frozen: `output/research/interactions_frozen.json` (frozen 2026-10-01T01:31:10+00:00; holdout run 2026-10-01T01:31:11+00:00).

## Verdict

**No. Combining factors adds nothing against the closing spread or total.** Nothing we could bet survived.

* **A. All combinations at once (GBM depth ≤ 3, ridge on all ~2,550 pairwise terms of 72 features):**
  * **CV picked the most heavily regularised settings.** In season-grouped CV it chose GBM depth 2 with 100 trees, and ridge at the largest alpha. CV R² was ≈ +0.001 (spread) and +0.004 (total), no better than predicting the training mean.
  * **Frozen holdout 2023-25, R² vs the market:**
    * spread: GBM −0.0035, ridge −0.0005
    * total: GBM +0.0036, ridge +0.0034
  * **The totals "gain" is the intercept.** Models refit on targets shuffled within season average +0.0035, the same as the real ridge. The gain comes from right-skewed totals residuals, not from the features.
  * **Betting the largest predicted residuals at the close lost.** The top 10% covered 39 / 43 / 46 / 41% (GBM spread, ridge spread, GBM total, ridge total); the top 5% covered 30 / 42 / 47 / 33%. That is at or below the shuffled-target null band.
  * **Early-line spread bets** at the best allowed-book price had negative CLV. In the top deciles, excess CLV (the directional test) was within ±1.5 SE. Betting the model's side in EVERY game shows a small directional excess (+0.6 to +1.0%, t 1.4–2.3), but raw CLV is −2.1 to −2.5%. That is not bettable, and it plausibly comes from the two not-ex-ante inputs (actual starting QB, recorded weather).
* **Early totals: the one apparent positive is recorded wind.** The totals GBM's early top decile shows excess CLV with t ≈ 3.8 on 2023-25, but all of it comes from games with RECORDED wind ≥ 12 mph, which a Tuesday bettor doesn't know. Excluding those games, CLV is negative (−1.4% holdout, −4.4% dev). H1 and H9 show the same thing (Tuesday CLV +8% and +5%, t ≈ 5 and 3, but only t 1.8 / 1.3 at the close). This is the already-known Tuesday wind-under effect (`totals.md`), not a new interaction.
* **B. Ten pre-registered combinations:** 0 of 10 passed the dev bar, so nothing was tested as a survivor. Only 1 of 10 was nominally |t| ≥ 1.96 in dev, and it had the WRONG sign:
  * H4 (rested team off a bye vs a team off MNF): t −3.98 on n 35. It then flipped to +0.78 on 2020-25, which is textbook small-sample noise.
  * H10 (fade new QB/coach early in the season) was also the wrong sign, but not significant (t −1.70).
  * On 2020-25 (all reported, none eligible) the best was H8 (late-season low-stakes overs), t 2.25, driven by 2020-22 only (2023-25 t 0.52, cover 52%).
* **C. False discoveries:**
  * **The pair sweep "won" in dev about as often as chance.** It covered 34 binary flags × 2 targets, 586 testable cells. 51 cells were nominally significant in dev (29 expected); the pure interaction term was significant 25 times (29 expected).
  * **7 cells "replicated" on 2020-25 (1.3 expected), but none is an interaction.** Every one is a single factor or totals skew: big home favourites, artificial turf → higher totals, late season. Their cover rates are 49–52%, not bettable at −110.
  * **Pure interaction terms: 25 dev winners → 1 replicated** (0.6 expected). The correlation of dev t with holdout t was +0.002.

**Paper track:** nothing new from combinations. The only signal that persists is the forecast-wind under, already tracked (`totals_wind_rules.json`). At most, log "forecast wind ≥ 15 × both offenses pass-heavy" as a pre-registered *sub-split* of that existing ledger, using logged Tuesday forecasts, never recorded wind. No separate bets.

## Data

Games with closing spread and total: train 2003-19 4539, dev 2020-22 838, holdout 2023-25 855 (REG + postseason).
72 pre-game features: the 22 Walters factor components (`walters_factors.build_factors`), rest days, travel km / tz shift / body-clock hour (`travel_features`), QB change / irregular starter / new QB / new coach flags, outdoor temp/wind/rain, turf, dome/warm team flags, EWMA (strictly prior games) points for/against and, 2012+, neutral pass rate, sec/snap, off/def EPA, pre-game win pct and conservative eliminated/clinched flags, plus the closing spread and total.
Not strictly ex-ante: recorded game-time weather and the actual starting QB (see module docstring).

## A. All combinations at once

Season-grouped expanding-window CV on 2003-19 (validation seasons 2009-19). R² is against the market (prediction 0 = the closing number is right): R² = 1 − SSE(model)/SSE(0). ΔMSE = model − market per game.

| target | model | spec chosen by CV | CV 2009-19 | dev 2020-22 |
|---|---|---|---|---|
| y_side | gbm | `{"max_depth": 2, "lr": 0.02, "n": 100, "leaf": 100, "l2": 10.0}` | R² +0.0009, ΔMSE -0.16 ± 0.41, corr +0.038, pred sd 0.82 | R² -0.0042, ΔMSE +0.66 ± 0.62, corr -0.017, pred sd 0.65 |
| y_side | ridge2 | `{"alpha": 300000.0}` | R² +0.0010, ΔMSE -0.18 ± 0.19, corr +0.031, pred sd 0.44 | R² -0.0012, ΔMSE +0.18 ± 0.29, corr -0.012, pred sd 0.33 |
| y_side | training mean only | – | R² -0.0002, ΔMSE +0.03 ± 0.04, corr -0.023, pred sd 0.05 | |
| y_tot | gbm | `{"max_depth": 2, "lr": 0.02, "n": 100, "leaf": 100, "l2": 10.0}` | R² +0.0044, ΔMSE -0.78 ± 0.51, corr +0.059, pred sd 0.90 | R² +0.0006, ΔMSE -0.10 ± 0.86, corr +0.037, pred sd 0.89 |
| y_tot | ridge2 | `{"alpha": 300000.0}` | R² -0.0000, ΔMSE +0.00 ± 0.47, corr +0.008, pred sd 0.51 | R² -0.0012, ΔMSE +0.21 ± 0.44, corr -0.017, pred sd 0.30 |
| y_tot | training mean only | – | R² +0.0010, ΔMSE -0.17 ± 0.32, corr +0.014, pred sd 0.12 | |

All CV configurations: y_side gbm {"max_depth": 2, "lr": 0.02, "n": 100, "leaf": 100, "l2": 10.0}: R² +0.0009; y_side gbm {"max_depth": 2, "lr": 0.02, "n": 100, "leaf": 300, "l2": 10.0}: R² +0.0006; y_side gbm {"max_depth": 2, "lr": 0.02, "n": 300, "leaf": 100, "l2": 10.0}: R² -0.0034; y_side gbm {"max_depth": 2, "lr": 0.02, "n": 300, "leaf": 300, "l2": 10.0}: R² -0.0028; y_side gbm {"max_depth": 3, "lr": 0.02, "n": 100, "leaf": 100, "l2": 10.0}: R² +0.0005; y_side gbm {"max_depth": 3, "lr": 0.02, "n": 100, "leaf": 300, "l2": 10.0}: R² -0.0011; y_side gbm {"max_depth": 3, "lr": 0.02, "n": 300, "leaf": 100, "l2": 10.0}: R² -0.0069; y_side gbm {"max_depth": 3, "lr": 0.02, "n": 300, "leaf": 300, "l2": 10.0}: R² -0.0068; y_side ridge2 {"alpha": 3000.0}: R² -0.1393; y_side ridge2 {"alpha": 10000.0}: R² -0.0395; y_side ridge2 {"alpha": 30000.0}: R² -0.0085; y_side ridge2 {"alpha": 100000.0}: R² +0.0001; y_side ridge2 {"alpha": 300000.0}: R² +0.0010; y_tot gbm {"max_depth": 2, "lr": 0.02, "n": 100, "leaf": 100, "l2": 10.0}: R² +0.0044; y_tot gbm {"max_depth": 2, "lr": 0.02, "n": 100, "leaf": 300, "l2": 10.0}: R² +0.0025; y_tot gbm {"max_depth": 2, "lr": 0.02, "n": 300, "leaf": 100, "l2": 10.0}: R² +0.0010; y_tot gbm {"max_depth": 2, "lr": 0.02, "n": 300, "leaf": 300, "l2": 10.0}: R² +0.0012; y_tot gbm {"max_depth": 3, "lr": 0.02, "n": 100, "leaf": 100, "l2": 10.0}: R² +0.0038; y_tot gbm {"max_depth": 3, "lr": 0.02, "n": 100, "leaf": 300, "l2": 10.0}: R² +0.0021; y_tot gbm {"max_depth": 3, "lr": 0.02, "n": 300, "leaf": 100, "l2": 10.0}: R² -0.0036; y_tot gbm {"max_depth": 3, "lr": 0.02, "n": 300, "leaf": 300, "l2": 10.0}: R² +0.0001; y_tot ridge2 {"alpha": 3000.0}: R² -0.1309; y_tot ridge2 {"alpha": 10000.0}: R² -0.0358; y_tot ridge2 {"alpha": 30000.0}: R² -0.0092; y_tot ridge2 {"alpha": 100000.0}: R² -0.0023; y_tot ridge2 {"alpha": 300000.0}: R² -0.0000

Betting only the largest predicted residuals at the close (−110, break-even 52.4%); thresholds = the 90th/95th percentile of |prediction| on 2020-22, frozen for 2023-25. Null = same pipeline refit on targets shuffled within season (60 permutations): mean [5%, 95%].

| target | model | period | top 10% | top 5% | null R² | null top-10% cover | null top-5% cover |
|---|---|---|---|---|---|---|---|
| y_side | gbm | dev 2020-22 | 84 bets, 38-46-0, cover 0.452 ± 0.054 (z vs 52.4% -1.31), ROI -0.136 | 42 bets, 17-25-0, cover 0.405 ± 0.076 (z vs 52.4% -1.54), ROI -0.227 | -0.0021 [-0.0081, +0.0025] | 0.491 [0.402, 0.574] | 0.494 [0.366, 0.610] |
| y_side | gbm | **holdout 2023-25** (R² -0.0035) | 61 bets, 24-37-0, cover 0.393 ± 0.062 (z vs 52.4% -2.04), ROI -0.249 | 27 bets, 8-19-0, cover 0.296 ± 0.088 (z vs 52.4% -2.37), ROI -0.434 | -0.0035 [-0.0069, +0.0012] | 0.499 [0.421, 0.595] | 0.502 [0.341, 0.657] |
| y_side | ridge2 | dev 2020-22 | 84 bets, 44-39-1, cover 0.530 ± 0.055 (z vs 52.4% +0.12), ROI +0.012 | 42 bets, 16-25-1, cover 0.390 ± 0.076 (z vs 52.4% -1.71), ROI -0.249 | -0.0006 [-0.0034, +0.0019] | 0.496 [0.416, 0.569] | 0.486 [0.374, 0.610] |
| y_side | ridge2 | **holdout 2023-25** (R² -0.0005) | 85 bets, 36-47-2, cover 0.434 ± 0.054 (z vs 52.4% -1.64), ROI -0.168 | 44 bets, 18-25-1, cover 0.419 ± 0.075 (z vs 52.4% -1.38), ROI -0.196 | -0.0015 [-0.0042, +0.0010] | 0.492 [0.412, 0.598] | 0.493 [0.360, 0.632] |
| y_tot | gbm | dev 2020-22 | 84 bets, 37-47-0, cover 0.441 ± 0.054 (z vs 52.4% -1.53), ROI -0.159 | 42 bets, 19-23-0, cover 0.452 ± 0.077 (z vs 52.4% -0.93), ROI -0.136 | -0.0023 [-0.0078, +0.0019] | 0.488 [0.393, 0.574] | 0.481 [0.356, 0.610] |
| y_tot | gbm | **holdout 2023-25** (R² +0.0036) | 70 bets, 32-37-1, cover 0.464 ± 0.060 (z vs 52.4% -1.00), ROI -0.113 | 39 bets, 18-20-1, cover 0.474 ± 0.081 (z vs 52.4% -0.62), ROI -0.093 | -0.0012 [-0.0068, +0.0049] | 0.491 [0.360, 0.575] | 0.493 [0.346, 0.629] |
| y_tot | ridge2 | dev 2020-22 | 84 bets, 36-47-1, cover 0.434 ± 0.054 (z vs 52.4% -1.64), ROI -0.170 | 42 bets, 15-27-0, cover 0.357 ± 0.074 (z vs 52.4% -2.16), ROI -0.318 | -0.0006 [-0.0024, +0.0016] | 0.447 [0.381, 0.530] | 0.435 [0.333, 0.524] |
| y_tot | ridge2 | **holdout 2023-25** (R² +0.0034) | 82 bets, 32-47-3, cover 0.405 ± 0.055 (z vs 52.4% -2.11), ROI -0.218 | 48 bets, 15-30-3, cover 0.333 ± 0.070 (z vs 52.4% -2.56), ROI -0.341 | +0.0035 [+0.0008, +0.0059] | 0.490 [0.383, 0.576] | 0.474 [0.359, 0.614] |

Early-line bets (first snapshot ≤ 9 days for spreads; Tuesday snapshot for totals) on the predicted side at the best allowed-book price (`my_books.json`), closing-line features replaced by the early consensus line. CLV is price-based (spreads: `edge_lab.closing_fair` mu_close_all via `line_move_model`; totals: sharp-close fair total, `totals.TotalDist`). Excess = CLV minus the mean CLV of both sides' best prices in that game (strips the vig level; its t is the directional test).

| target | model | period | baseline both sides | all games | top 10% | top 5% |
|---|---|---|---|---|---|---|
| y_side | gbm | dev 2020-22 | CLV -0.0317, ROI -0.032 (n 1664) | 832 bets, CLV -0.0254 (t -6.1), excess +0.0063 (t +1.5), ROI -0.002 ± 0.033 | 82 bets, CLV -0.0208 (t -1.7), excess +0.0123 (t +1.0), ROI -0.033 ± 0.104 | 41 bets, CLV -0.0311 (t -1.8), excess +0.0016 (t +0.1), ROI -0.207 ± 0.145 |
| y_side | gbm | **holdout 2023-25** | CLV -0.0305, ROI -0.032 (n 1708) | 854 bets, CLV -0.0210 (t -5.1), excess +0.0095 (t +2.3), ROI -0.004 ± 0.032 | 61 bets, CLV -0.0325 (t -1.9), excess -0.0036 (t -0.2), ROI -0.172 ± 0.121 | 27 bets, CLV -0.0369 (t -1.4), excess -0.0095 (t -0.4), ROI -0.437 ± 0.170 |
| y_side | ridge2 | dev 2020-22 | CLV -0.0317, ROI -0.032 (n 1664) | 832 bets, CLV -0.0235 (t -5.7), excess +0.0081 (t +2.0), ROI -0.048 ± 0.033 | 83 bets, CLV -0.0333 (t -2.7), excess -0.0011 (t -0.1), ROI +0.062 ± 0.104 | 44 bets, CLV -0.0269 (t -1.6), excess +0.0052 (t +0.3), ROI -0.216 ± 0.140 |
| y_side | ridge2 | **holdout 2023-25** | CLV -0.0305, ROI -0.032 (n 1708) | 854 bets, CLV -0.0249 (t -6.0), excess +0.0056 (t +1.4), ROI +0.018 ± 0.032 | 85 bets, CLV -0.0135 (t -1.2), excess +0.0162 (t +1.5), ROI -0.076 ± 0.103 | 44 bets, CLV -0.0119 (t -0.9), excess +0.0176 (t +1.4), ROI -0.148 ± 0.144 |
| y_tot | gbm | dev 2020-22 | CLV -0.0324, ROI -0.030 (n 1598) | 799 bets, CLV -0.0355 (t -11.4), excess -0.0031 (t -1.0), ROI -0.061 ± 0.034 | 75 bets, CLV -0.0039 (t -0.3), excess +0.0287 (t +2.1), ROI -0.095 ± 0.110 | 37 bets, CLV +0.0081 (t +0.4), excess +0.0420 (t +1.9), ROI -0.091 ± 0.158 |
| y_tot | gbm | **holdout 2023-25** | CLV -0.0319, ROI -0.035 (n 1710) | 855 bets, CLV -0.0242 (t -9.2), excess +0.0076 (t +2.9), ROI -0.026 ± 0.032 | 73 bets, CLV +0.0047 (t +0.5), excess +0.0344 (t +3.8), ROI -0.018 ± 0.110 | 39 bets, CLV +0.0183 (t +1.5), excess +0.0464 (t +3.8), ROI -0.020 ± 0.155 |
| y_tot | ridge2 | dev 2020-22 | CLV -0.0324, ROI -0.030 (n 1598) | 799 bets, CLV -0.0464 (t -14.9), excess -0.0140 (t -4.5), ROI -0.042 ± 0.034 | 74 bets, CLV -0.0544 (t -4.7), excess -0.0225 (t -1.9), ROI -0.175 ± 0.111 | 39 bets, CLV -0.0507 (t -2.8), excess -0.0199 (t -1.1), ROI -0.269 ± 0.150 |
| y_tot | ridge2 | **holdout 2023-25** | CLV -0.0319, ROI -0.035 (n 1710) | 855 bets, CLV -0.0415 (t -15.8), excess -0.0097 (t -3.7), ROI -0.069 ± 0.032 | 82 bets, CLV -0.0350 (t -4.6), excess -0.0049 (t -0.6), ROI -0.254 ± 0.102 | 48 bets, CLV -0.0341 (t -3.3), excess -0.0040 (t -0.4), ROI -0.360 ± 0.129 |

GBM dev permutation importance (MSE increase, 2020-22): y_side: home_pf_ewm +0.22, away_pf_ewm +0.13, week +0.07, home_pa_ewm +0.06, away_off_epa_ewm +0.03, off_ot +0.03, away_pa_ewm +0.02, spread_line +0.02; y_tot: wind_o +0.62, home_def_epa_ewm +0.53, off_ot +0.14, home_new_qb +0.12, temp_o +0.08, home_pass_rate_ewm +0.06, away_wpct +0.03, art_turf +0.02

Holdout by season (R² vs market): y_side gbm: 2023 -0.0029, 2024 -0.0039, 2025 -0.0037; y_side ridge2: 2023 +0.0003, 2024 -0.0016, 2025 -0.0003; y_tot gbm: 2023 +0.0210, 2024 -0.0167, 2025 +0.0050; y_tot ridge2: 2023 +0.0038, 2024 +0.0024, 2025 +0.0038

## B. Hypothesis-driven combinations (pre-registered before any result)

Dev bar (2003-19): n ≥ 40, one-sided t ≥ 2.576 (p < 0.05/10) on the mean residual in the stated direction, and cover > 52.4%. Survivors tested once on 2020-25. Residual = margin − spread_line (side) or total − total_line (totals), signed toward the bet. Weather = recorded conditions.

- **H1_wind_passheavy_under** (y_tot): outdoor/open roof, wind >= 15 mph, and the two teams' mean pre-game neutral-situation pass rate >= its 2012-19 median (pbp seasons only) -> UNDER. *Why:* the market lowers totals for wind on average, but wind should hurt pass-dependent offenses more than a generic adjustment allows.
- **H2_tnf_long_travel_home** (y_side): Thursday game with both teams on <= 5 days rest and the visitor travelling >= 1600 km (~1000 mi) base-to-venue -> HOME. *Why:* short preparation plus a long trip compounds fatigue/logistics for the road team.
- **H3_backup_qb_road_weather_home** (y_side): road team's starter started < 50% of its previous 8 games (>= 4 prior games), outdoor game with temp <= 40F or wind >= 15 mph -> HOME. *Why:* the market prices the backup, but an inexperienced QB on the road in bad weather is worse than either adjustment alone.
- **H4_bye_vs_off_mnf_rested** (y_side): regular season, one team has >= 13 days rest (off bye) and the other <= 6 days (off MNF) -> the RESTED team. *Why:* largest regular-season rest gap; bye and MNF are each priced individually.
- **H5_tnf_night_west** (y_side): Thursday game, kickoff >= 7pm ET, both teams on <= 5 days rest, team base zones differ by >= 2 (E vs M/P, or C vs P) -> the more WESTERN team. *Why:* night-game body-clock edge of the western team should be larger with no time to adjust.
- **H6_div_big_road_fav_dog** (y_side): divisional game with the road team favoured by >= 7 at the close -> HOME (the dog). *Why:* division familiarity compresses margins; big road favourites are popular with the public.
- **H7_warm_team_cold_late** (y_side): road warm-weather team (Walters' definition), outdoor game, temp <= 35F, week >= 13 or postseason -> HOME. *Why:* cold-weather effect on warm-climate teams should be largest late in the season.
- **H8_low_stakes_late_over** (y_tot): regular season week >= 15 and BOTH teams eliminated or clinched (conservative mathematical rules from prior results only) -> OVER. *Why:* reduced defensive intensity / backups in games with nothing at stake.
- **H9_dome_team_outdoor_wind_under** (y_tot): at least one team with a dome/retractable home plays at an outdoor/open venue with wind >= 15 mph -> UNDER. *Why:* dome offenses (built for and used to calm conditions) suffer more in wind.
- **H10_early_new_qb_or_coach_fade** (y_side): weeks 1-3 of the regular season, exactly one team has a new starting QB (not its most frequent starter of the previous season) or a new head coach -> FADE that team. *Why:* the market may overrate new-QB/new-coach teams before evidence accumulates.

| hypothesis | dev 2003-19 | passes dev bar | 2020-25 (all reported) | 2020-25 early-line CLV |
|---|---|---|---|---|
| H1_wind_passheavy_under | n 81, +1.77 ± 1.24 (t +1.43), cover 0.582 ± 0.056 | no | n 47, +3.47 ± 1.90 (t +1.83), cover 0.596 ± 0.072 | 46 bets, CLV +0.0826 (t +4.9), excess +0.1156 (t +6.6), ROI +0.349 ± 0.128 |
| H2_tnf_long_travel_home | n 40, -1.21 ± 2.05 (t -0.59), cover 0.474 ± 0.081 | no | n 36, -0.56 ± 1.91 (t -0.29), cover 0.556 ± 0.083 | 36 bets, CLV -0.0518 (t -2.0), excess -0.0195 (t -0.8), ROI +0.012 ± 0.162 |
| H3_backup_qb_road_weather_home | n 170, -0.05 ± 1.00 (t -0.05), cover 0.506 ± 0.039 | no | n 59, +1.75 ± 1.55 (t +1.13), cover 0.586 ± 0.065 | 59 bets, CLV -0.0281 (t -1.5), excess +0.0057 (t +0.3), ROI +0.167 ± 0.122 |
| H4_bye_vs_off_mnf_rested | n 35, -6.70 ± 1.68 (t -3.98), cover 0.294 ± 0.078 | no | n 16, +2.19 ± 2.81 (t +0.78), cover 0.562 ± 0.124 | 15 bets, CLV +0.0030 (t +0.2), excess +0.0344 (t +2.1), ROI +0.018 ± 0.255 |
| H5_tnf_night_west | n 16, +5.38 ± 3.02 (t +1.78), cover 0.733 ± 0.114 | no | n 13, +3.62 ± 2.80 (t +1.29), cover 0.538 ± 0.138 | 13 bets, CLV -0.0407 (t -0.8), excess -0.0088 (t -0.2), ROI +0.258 ± 0.252 |
| H6_div_big_road_fav_dog | n 129, -1.62 ± 1.12 (t -1.44), cover 0.488 ± 0.044 | no | n 64, +0.57 ± 1.71 (t +0.33), cover 0.587 ± 0.062 | 41 bets, CLV -0.0134 (t -1.0), excess +0.0201 (t +1.5), ROI +0.080 ± 0.151 |
| H7_warm_team_cold_late | n 65, +1.31 ± 1.39 (t +0.95), cover 0.547 ± 0.062 | no | n 37, +3.40 ± 2.10 (t +1.62), cover 0.611 ± 0.081 | 37 bets, CLV -0.0301 (t -1.7), excess +0.0040 (t +0.2), ROI +0.243 ± 0.152 |
| H8_low_stakes_late_over | n 159, +0.68 ± 1.09 (t +0.63), cover 0.471 ± 0.040 | no | n 50, +3.96 ± 1.76 (t +2.25), cover 0.620 ± 0.069 | 47 bets, CLV -0.0732 (t -6.8), excess -0.0428 (t -4.1), ROI +0.093 ± 0.139 |
| H9_dome_team_outdoor_wind_under | n 86, +1.47 ± 1.42 (t +1.03), cover 0.566 ± 0.054 | no | n 49, +2.48 ± 1.96 (t +1.27), cover 0.542 ± 0.072 | 48 bets, CLV +0.0463 (t +2.9), excess +0.0784 (t +4.9), ROI +0.097 ± 0.137 |
| H10_early_new_qb_or_coach_fade | n 366, -1.12 ± 0.66 (t -1.70), cover 0.444 ± 0.026 | no | n 127, -1.56 ± 1.15 (t -1.36), cover 0.472 ± 0.045 | 126 bets, CLV -0.0276 (t -3.0), excess +0.0030 (t +0.3), ROI -0.000 ± 0.085 |

Dev by era (2003-11 / 2012-19): H1_wind_passheavy_under: – / 1.43; H2_tnf_long_travel_home: 0.51 / -1.03; H3_backup_qb_road_weather_home: 0.14 / -0.27; H4_bye_vs_off_mnf_rested: -3.96 / -2.1; H5_tnf_night_west: 0.68 / 2.39; H6_div_big_road_fav_dog: -0.06 / -2.09; H7_warm_team_cold_late: 0.43 / 1.06; H8_low_stakes_late_over: 0.8 / 0.07; H9_dome_team_outdoor_wind_under: -0.08 / 1.83; H10_early_new_qb_or_coach_fade: -1.71 / -0.64 (t values)

2020-22 / 2023-25 split (t): H1_wind_passheavy_under: 1.87 / 0.4; H2_tnf_long_travel_home: -1.53 / 0.95; H3_backup_qb_road_weather_home: 0.91 / 0.73; H4_bye_vs_off_mnf_rested: 0.24 / 0.89; H5_tnf_night_west: -0.04 / 2.27; H6_div_big_road_fav_dog: 0.12 / 0.39; H7_warm_team_cold_late: 0.18 / 2.18; H8_low_stakes_late_over: 2.58 / 0.52; H9_dome_team_outdoor_wind_under: 1.77 / -0.05; H10_early_new_qb_or_coach_fade: -0.98 / -0.96

## C. False-discovery picture

**B scorecard:** 0 of 10 pre-registered combinations passed the dev bar; 0 held up on 2020-25.
Nominally 'significant' at p < 0.05 in dev (two-sided t ≥ 1.96 either direction): 1 of 10.

**Exhaustive sweep** of all pairwise AND-combinations of 34 binary flags (art_turf, away_2nd_tz_trip, away_3rd_road_in_4, away_bounce, away_fav, away_new_coach, away_off_bye, away_off_mnf, away_qb_change, away_qb_irreg, cold35, div, early_wk3, high_total48, home_bounce, home_fav7, home_new_coach, home_off_bye, home_off_mnf, home_qb_change, home_qb_irreg, late_wk13, low_total40, mnf, night_tz, nonconf, playoff, rain, short_trip, snf, tnf, travel_2000mi, tz_10am, wind15), both targets, cells with n ≥ 50 in 2003-19: 586 tests.
- dev 'winners' at |t| ≥ 1.96 (cell mean ≠ 0): 51 (expected by chance ≈ 29.3); pure interaction term (y ~ a + b + a·b) |t| ≥ 1.96: 25 of 586.
- Bonferroni (|t| ≥ 3.93): 1 cells, 0 interaction terms.
- Largest dev cells: art_turf×low_total40 y_tot n 386 mean +2.74 t +3.99; away_off_mnf×nonconf y_tot n 52 mean +5.44 t +2.99; home_off_mnf×late_wk13 y_tot n 59 mean +5.08 t +2.88; home_fav7×tz_10am y_side n 104 mean +3.46 t +2.73; art_turf×away_qb_change y_tot n 183 mean +2.81 t +2.71; away_new_coach×home_new_coach y_side n 196 mean -2.43 t -2.63
- Of the 51 dev winners, 36 had the same sign in 2020-25 and 7 were again |t| ≥ 1.96 with the same sign (≈ 1.3 expected if all were noise). Correlation of dev t vs 2020-25 t over all 586 tests: +0.152. Holdout cells at |t| ≥ 1.96 regardless of dev: 38.
- Bonferroni dev winners in 2020-25: art_turf×low_total40 y_tot dev t +3.99 → hold t +1.35 (n 77)
- Replicated at p < 0.05: art_turf×away_3rd_road_in_4 y_tot dev t +2.45 → hold t +2.20 (n 247, mean +1.94); art_turf×away_qb_irreg y_tot dev t +1.98 → hold t +1.97 (n 180, mean +1.89); art_turf×home_fav7 y_tot dev t +1.96 → hold t +2.62 (n 157, mean +2.74); cold35×home_fav7 y_side dev t +2.00 → hold t +3.09 (n 39, mean +4.99); home_fav7×late_wk13 y_side dev t +2.47 → hold t +2.33 (n 117, mean +2.68); home_fav7×low_total40 y_side dev t +2.48 → hold t +2.04 (n 34, mean +4.44); home_off_mnf×late_wk13 y_tot dev t +2.88 → hold t +2.13 (n 26, mean +5.98)

## Post-hoc diagnostics (after the one-shot holdout; select nothing, change no verdict)

- Totals GBM early-line top decile 2020-22: 75 bets, 77% unders, 41% in games with RECORDED wind ≥ 12 mph. Wind ≥ 12 only: 31 bets, CLV +0.0523 (t +2.3), excess +0.0866 (t +3.7), ROI +0.172 ± 0.170. Excluding them: 44 bets, CLV -0.0436 (t -3.2), excess -0.0121 (t -0.9), ROI -0.283 ± 0.139.
- Totals GBM early-line top decile 2023-25: 73 bets, 67% unders, 27% in games with RECORDED wind ≥ 12 mph. Wind ≥ 12 only: 20 bets, CLV +0.0550 (t +3.9), excess +0.0853 (t +6.0), ROI -0.045 ± 0.219. Excluding them: 53 bets, CLV -0.0143 (t -1.4), excess +0.0151 (t +1.5), ROI -0.008 ± 0.128.
- Sweep, pure interaction terms (y ~ a + b + a·b): 25 dev winners → 16 same sign in 2020-25, 1 replicated at p < 0.05 (≈ 0.6 expected by chance); corr(dev t, 2020-25 t) = +0.002.
- Sweep cell-t correlation by target: y_side +0.061, y_tot +0.260. Totals residuals are right-skewed: mean total − total_line {'2003-19': 0.651, '2020-25': 0.651} but over rate (ex push) {'2003-19': 0.4972, '2020-25': 0.4884} — mean-residual cell tests for totals pick up skew that does not pay at −110.
- Single flags behind the 'replicated' cells (cover rate is what pays): home_fav7:y_side: 2003-19 n 1095, +0.88 ± 0.41 (t +2.14), cover 0.492 ± 0.015, 2020-22 n 184, +1.30 ± 0.91 (t +1.42), cover 0.520 ± 0.037, 2023-25 n 156, +1.76 ± 1.04 (t +1.70), cover 0.516 ± 0.040; art_turf:y_tot: 2003-19 n 1906, +0.98 ± 0.31 (t +3.15), cover 0.511 ± 0.011, 2020-22 n 371, +1.19 ± 0.71 (t +1.68), cover 0.490 ± 0.026, 2023-25 n 374, +1.84 ± 0.71 (t +2.60), cover 0.519 ± 0.026; late_wk13:y_tot: 2003-19 n 1359, +0.43 ± 0.37 (t +1.16), cover 0.480 ± 0.014, 2020-22 n 262, +1.28 ± 0.82 (t +1.56), cover 0.488 ± 0.031, 2023-25 n 279, +2.08 ± 0.83 (t +2.49), cover 0.558 ± 0.030
