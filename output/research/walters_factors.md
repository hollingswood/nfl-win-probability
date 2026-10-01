# Walters 'Gambler' game factors: fixed-table out-of-sample test

Published factor values (end of 2022-23) applied without fitting; 1 unit = 0.20 pt; H = home minus visitor factor points. Residual = home margin − closing `spread_line` (nflverse). 2023-2025 is out-of-sample for the published table. All thresholds pre-specified; all reported.

## Interpretations

- Team codes normalised to franchises (OAK->LV, SD->LAC, STL->LA); each team's home stadium/base per season = its most-used non-neutral home venue (travel.team_bases); STADIUMS lat/lon/tz for distances and zones.
- Turf: team's home-turf type per season = majority of its non-neutral home games, grass/dessograss = grass, everything else artificial (blank surface -> nearest season of that team). Same type +1 visitor, opposite +1 home.
- Prime-time slots: Thursday/Sunday/Monday with kickoff >= 7pm ET (Thanksgiving afternoon games are not TNF). 'Coming off MNF' requires the MNF game to be the team's previous game within 7 days (not across a bye).
- Overtime: previous game (same season) went to OT and was <= 10 days ago (a bye in between cancels it); 'home' OT game = team was the non-neutral home team.
- 3rd away game in four: the visitor's current game plus its previous 3 games this season contain >= 3 non-home games (away or neutral).
- Bye: regular-season game, week > 1, >= 13 days since the team's previous game this season. Quality = tercile of the team's pre-game Elo (src features.elo_ratings parameters, re-run to rank vs all teams' current Elo at that date): bottom third 'below-average' 4 (5 away), middle 'average' 5 (6), top 'great' 7 (8). Both teams can get it.
- Playoff bye: Divisional-round team whose previous game was a regular-season game -> +1 home (bye teams host). Regular bye units are not applied in the playoffs; the 2-week Super Bowl gap is not a bye.
- Super Bowl: previous season's SB winner gets +4 in its 1st game and +2 in games 2-4 of the next season; SB loser's opponent gets the same.
- Travel: visitor's base-to-venue great-circle distance >= 2000 miles -> +1 home. Short-trip visitor bonuses (non-neutral games only): TB/JAX/MIA +1, DAL/HOU +1, ATL/CAR +1, IND/CIN +1, CHI/GB +1, LA metro pair (LA/LAC) +2 and Bay Area pair (SF/OAK, through 2019) +2 as the 'SF/LA area' analogue, LV vs LA/LAC +1, any pair of PHI/NYG/NYJ/WAS/NE/BAL/BUF +1 except NYG/NYJ +2 and BAL/WAS +2. Pairs keyed on bases, so e.g. SD-LA or OAK-LA pre-2020 are not bonused.
- Zones from base tz: East = New_York/Detroit/Indianapolis/Toronto, Central = Chicago, Mountain = Denver/Phoenix (ARI as Mountain), Pacific = Los_Angeles (incl. LV).
- 10 a.m. games: kickoff before 2pm ET at a venue in the Eastern/Central zone (incl. Mexico City) for a Pacific-base team -> +2 opponent, Mountain-base team -> +1 opponent.
- Night games (kickoff >= 7pm ET): penalties East 6, Central 3, Mountain 1, Pacific 0; net units = penalty(eastern team) - penalty(western team) to the more western team.
- Second consecutive game >= 2 time zones from home: visitor's |tz shift| >= 2 in this game AND in its previous game this season -> +2 home.
- Bounce back: previous game (same season, any rest) lost by >= 19 -> +2, >= 29 -> +4 (not cumulative).
- Warm-weather team = base latitude < 34.5 or Las Vegas (MIA TB JAX HOU NO ATL DAL ARI LA LAC SD LV); cold-climate dome team = home roof mostly dome/closed/retractable and not warm (DET, MIN except 2014-15, IND, STL). Only the VISITOR can be 'visiting'; cold OUTDOOR = roof outdoors/open with nflverse temp (recorded at game time, i.e. not strictly ex-ante). Warm: <=35F .25, <=30 .50, <=25 .75, <=20 1.00, <=15 1.25, <=10 1.75 to home. Dome: 20<t<=30 .25, 10<t<=20 .50, t<=10 .75 to home.
- Rain from play-by-play `weather` text (2012+ only; 2003-11 rain = 0): description before 'Temp:' mentions rain/showers/drizzle and not 'chance/possible/no rain/%' -> +0.25 visitor; heavy/hard/downpour/pouring -> +0.75. Outdoor/open roof only. Recorded conditions, not forecast.
- W values: main H treats them as W units (x 0.2 pt); H_wpts treats them as points (reported side by side).
- Skipped: 'Variable' items (matchups, snow, heavy wind), all E-factors (subjective) and QB-specific W-factors.
- Neutral-site games keep the schedule's designated home team for all factors.

## 1. Distribution of H

| period | games | mean | sd | p05 | p95 | |H|≥0.5 | |H|≥1 | |H|≥1.5 | |H|≥2 |
|---|---|---|---|---|---|---|---|---|---|
| 2003-2022 H | 5377 | +0.28 | 0.78 | -1.00 | +1.60 | 43.2% | 22.2% | 8.2% | 3.4% |
| 2003-2022 H_wpts | 5377 | +0.28 | 0.78 | -1.00 | +1.60 | 43.5% | 22.5% | 8.4% | 3.5% |
| 2023-2025 H | 855 | +0.36 | 0.83 | -0.80 | +1.80 | 47.7% | 26.0% | 11.6% | 5.1% |
| 2023-2025 H_wpts | 855 | +0.36 | 0.84 | -0.80 | +1.83 | 47.7% | 26.2% | 12.2% | 5.5% |

Share of games where each component is non-zero (2003-22 / 2023-25): turf 100%/100%, division 36%/34%, conference 25%/28%, home_tnf 4%/6%, home_snf 6%/7%, home_mnf 6%/8%, home_off_mnf 4%/5%, away_off_mnf 4%/6%, three_away 29%/32%, off_ot 10%/8%, bye 11%/10%, playoff_bye 1%/1%, super_bowl 3%/3%, travel_2000 10%/12%, short_trip 9%/10%, tz_10am 9%/11%, night_tz 9%/13%, two_tz 3%/3%, bounce_back 18%/18%, w_warm_cold 2%/2%, w_dome_cold 0%/0%, w_rain 3%/4%

## 2. Does H predict the result beyond the closing line?

Slope of residual (margin − spread_line) on H: 1 = market ignores the factors, 0 = fully priced.

| period | resid ~ H | resid ~ H_wpts | resid ~ H without night_tz | margin ~ H (no line) | b_H in margin ~ line + H |
|---|---|---|---|---|---|
| 2003-2022 (Walters in-sample) | +0.96 ± 0.23 (t +4.1, n 5377) | +0.93 ± 0.23 (t +4.1, n 5377) | +0.87 ± 0.24 (t +3.6, n 5377) | +0.71 ± 0.26 (t +2.8, n 5377) | +0.97 ± 0.23 |
| 2003-2012 | +1.25 ± 0.35 (t +3.5, n 2670) | +1.23 ± 0.35 (t +3.5, n 2670) | +1.10 ± 0.37 (t +3.0, n 2670) | +0.71 ± 0.40 (t +1.8, n 2670) | +1.29 ± 0.35 |
| 2013-2022 | +0.73 ± 0.30 (t +2.4, n 2707) | +0.69 ± 0.30 (t +2.3, n 2707) | +0.67 ± 0.32 (t +2.1, n 2707) | +0.72 ± 0.33 (t +2.2, n 2707) | +0.73 ± 0.30 |
| 2020-2022 | -0.19 ± 0.54 (t -0.3, n 838) | -0.20 ± 0.53 (t -0.4, n 838) | -0.38 ± 0.57 (t -0.7, n 838) | -0.41 ± 0.60 (t -0.7, n 838) | -0.20 ± 0.54 |
| 2023-2025 (out-of-sample) | +0.34 ± 0.48 (t +0.7, n 855) | +0.42 ± 0.48 (t +0.9, n 855) | +0.08 ± 0.54 (t +0.1, n 855) | -0.03 ± 0.55 (t -0.1, n 855) | +0.40 ± 0.48 |

Against the price-implied close (`edge_lab.closing_fair` mu_close_all): 2020-2022: -0.22 ± 0.54 (t -0.4, n 834); 2023-2025: +0.30 ± 0.48 (t +0.6, n 854)

ATS at the closing spread_line, betting the side H favours (−110; break-even 52.4%):

| period | |H|≥ | bets | /season | W-L-P | cover ± SE | z vs 52.4% | ROI ± SE |
|---|---|---|---|---|---|---|---|
| 2003-2022 (H) | 0.5 | 2322 | 116.1 | 1188-1078-56 | 0.524 ± 0.010 | +0.04 | +0.001 ± 0.020 |
| 2003-2022 (H) | 1.0 | 1195 | 59.8 | 641-526-28 | 0.549 ± 0.015 | +1.74 | +0.047 ± 0.027 |
| 2003-2022 (H) | 1.5 | 440 | 22.0 | 240-188-12 | 0.561 ± 0.024 | +1.53 | +0.069 ± 0.045 |
| 2003-2022 (H) | 2.0 | 182 | 9.1 | 97-83-2 | 0.539 ± 0.037 | +0.41 | +0.028 ± 0.070 |
| 2023-2025 (H) | 0.5 | 408 | 136.0 | 206-194-8 | 0.515 ± 0.025 | -0.35 | -0.016 ± 0.047 |
| 2023-2025 (H) | 1.0 | 222 | 74.0 | 118-100-4 | 0.541 ± 0.034 | +0.52 | +0.033 ± 0.063 |
| 2023-2025 (H) | 1.5 | 99 | 33.0 | 49-48-2 | 0.505 ± 0.051 | -0.37 | -0.035 ± 0.095 |
| 2023-2025 (H) | 2.0 | 44 | 14.7 | 22-22-0 | 0.500 ± 0.075 | -0.32 | -0.045 ± 0.146 |
| 2003-2022 (H_wpts) | 1.0 | 1210 | 60.5 | 647-534-29 | 0.548 ± 0.014 | +1.65 | +0.045 ± 0.027 |
| 2003-2022 (H_wpts) | 1.5 | 449 | 22.4 | 244-193-12 | 0.558 ± 0.024 | +1.45 | +0.064 ± 0.044 |
| 2003-2022 (H_wpts) | 2.0 | 186 | 9.3 | 98-86-2 | 0.533 ± 0.037 | +0.24 | +0.017 ± 0.070 |
| 2023-2025 (H_wpts) | 1.0 | 224 | 74.7 | 121-99-4 | 0.550 ± 0.034 | +0.78 | +0.049 ± 0.063 |
| 2023-2025 (H_wpts) | 1.5 | 104 | 34.7 | 52-50-2 | 0.510 ± 0.049 | -0.28 | -0.026 ± 0.093 |
| 2023-2025 (H_wpts) | 2.0 | 47 | 15.7 | 25-22-0 | 0.532 ± 0.073 | +0.11 | +0.015 ± 0.140 |

## 3. Line movement (early-week consensus → price-implied close), 2020-2025

First snapshot ≤ 9 days before kickoff (≥ 3 h). move_px = mu_close_all − price-implied consensus at that snapshot (home-signed); move_raw = spread_line − consensus point.

| period | games | median h before | move_px ~ H | move_raw ~ H | move_px ~ H_wpts |
|---|---|---|---|---|---|
| 2020-2022 | 834 | 162.4 | +0.06 ± 0.08 (t +0.8, n 834) | +0.06 ± 0.08 (t +0.7, n 834) | +0.07 ± 0.08 (t +0.8, n 834) |
| 2023-2025 | 854 | 211.3 | +0.11 ± 0.09 (t +1.3, n 854) | +0.04 ± 0.09 (t +0.4, n 854) | +0.12 ± 0.09 (t +1.4, n 854) |

Bet the H side at the first snapshot, best allowed-book price (price-based CLV vs honest close):

Excess CLV = H-side CLV minus the mean CLV of both sides' best prices in the same game (strips the ~3% vig/shopping level; its t is the directional test).

| period | |H|≥ | bets | CLV (t) | excess CLV (t) | beat close | ROI ± SE |
|---|---|---|---|---|---|---|
| 2020-2022 | baseline: both sides every game | 1664 | -0.0317 | 0 | | -0.0317 |
| 2020-2022 | 0.5 | 366 | -0.0335 (-5.2) | -0.0018 (-0.3) | 0.369 | -0.0225 ± 0.0497 |
| 2020-2022 | 1.0 | 202 | -0.0335 (-4.1) | -0.0022 (-0.3) | 0.371 | -0.0191 ± 0.0670 |
| 2020-2022 | 1.5 | 85 | -0.0157 (-1.4) | +0.0150 (+1.4) | 0.447 | -0.0078 ± 0.1032 |
| 2020-2022 | 2.0 | 33 | -0.0097 (-0.5) | +0.0227 (+1.2) | 0.424 | -0.0220 ± 0.1678 |
| 2023-2025 | baseline: both sides every game | 1708 | -0.0305 | 0 | | -0.0319 |
| 2023-2025 | 0.5 | 407 | -0.0189 (-3.4) | +0.0120 (+2.2) | 0.408 | +0.0320 ± 0.0471 |
| 2023-2025 | 1.0 | 221 | -0.0226 (-3.0) | +0.0088 (+1.2) | 0.398 | +0.0286 ± 0.0637 |
| 2023-2025 | 1.5 | 99 | -0.0223 (-1.9) | +0.0081 (+0.7) | 0.475 | -0.0692 ± 0.0959 |
| 2023-2025 | 2.0 | 44 | -0.0253 (-1.3) | +0.0052 (+0.3) | 0.477 | -0.0878 ± 0.1456 |

## 4. Does H improve our model or the closing line (2023-25)?

- residual (margin − mu_model) on H: 2020-2022: -0.27 ± 0.54 (t -0.5, n 834); 2023-2025: +0.50 ± 0.49 (t +1.0, n 854)
- fixed_add_H_to_model_2023_25: MSE 166.19 → 166.11 (Δ -0.075 ± 0.730, n 854)
- fixed_add_H_to_close_2023_25: MSE 161.9 → 161.68 (Δ -0.215 ± 0.722, n 854)
- fixed_add_H_to_close_2023_25_all_games: MSE 161.77 → 161.54 (Δ -0.230 ± 0.721, n 855)
- fitted_model_plus_H_2023_25: MSE 167.91 → 168.17 (Δ +0.258 ± 0.194, n 854, coef_H -0.28)
- fitted_model_line_plus_H_2023_25: MSE 162.43 → 162.59 (Δ +0.169 ± 0.150, n 854, coef_H -0.22)
- fitted_close_plus_H_2023_25: MSE 161.54 → 161.55 (Δ +0.009 ± 0.646, n 855, coef_H +0.97)
- fitted_close_plus_all_components_2023_25: MSE 161.54 → 162.3 (Δ +0.760 ± 1.056, n 855)

Model residual on each component (points), 2023-25: turf -1.40 ± 2.21 (t -0.6, n 854); division -1.44 ± 4.60 (t -0.3, n 854); conference -3.61 ± 4.89 (t -0.7, n 854); home_tnf -0.81 ± 4.51 (t -0.2, n 854); home_snf -0.92 ± 2.15 (t -0.4, n 854); home_mnf -3.02 ± 3.85 (t -0.8, n 854); home_off_mnf +0.10 ± 2.70 (t +0.0, n 854); away_off_mnf +0.69 ± 1.27 (t +0.5, n 854); three_away -0.20 ± 2.35 (t -0.1, n 854); off_ot +0.02 ± 2.53 (t +0.0, n 854); bye +2.27 ± 1.05 (t +2.2, n 854); playoff_bye +17.50 ± 33.08 (t +0.5, n 854); super_bowl +0.08 ± 2.72 (t +0.0, n 854); travel_2000 -13.54 ± 7.04 (t -1.9, n 854); short_trip -3.80 ± 7.14 (t -0.5, n 854); tz_10am -2.77 ± 4.41 (t -0.6, n 854); night_tz +2.42 ± 1.35 (t +1.8, n 854); two_tz -0.64 ± 5.03 (t -0.1, n 854); bounce_back +0.58 ± 1.89 (t +0.3, n 854); w_warm_cold +43.32 ± 14.33 (t +3.0, n 854); w_rain +23.79 ± 50.36 (t +0.5, n 854)

## 5. Individual factors

For games where the factor is active: Walters' average size (pts) vs the realised residual toward the favoured side (margin − spread_line), and the line's own lean toward that side. ~22 factors × 2 periods → treat |t| < 3 as noise (Bonferroni 5% ≈ |t| 3.0).

| factor | 03-22 n | Walters pts | resid toward ± SE (t) | line toward | 23-25 n | resid toward ± SE (t) | line toward |
|---|---|---|---|---|---|---|---|
| turf | 5377 | 0.2 | +0.47 ± 0.18 (+2.6) | 0.02 | 855 | -0.26 ± 0.44 (-0.6) | -0.0 |
| division | 1949 | 0.2 | +0.49 ± 0.29 (+1.7) | -2.2 | 293 | -0.88 ± 0.72 (-1.2) | -1.71 |
| conference | 1332 | 0.2 | -0.20 ± 0.37 (-0.5) | 2.22 | 243 | +0.38 ± 0.82 (+0.5) | 1.62 |
| home_tnf | 210 | 0.4 | +0.60 ± 0.86 (+0.7) | 1.83 | 51 | +0.15 ± 1.69 (+0.1) | 2.62 |
| home_snf | 342 | 0.8 | +0.93 ± 0.76 (+1.2) | 2.55 | 57 | +0.21 ± 1.67 (+0.1) | 1.12 |
| home_mnf | 294 | 0.4 | -0.02 ± 0.79 (-0.0) | 1.77 | 64 | -0.02 ± 1.47 (-0.0) | 1.37 |
| home_off_mnf | 232 | 0.8 | +0.08 ± 0.87 (+0.1) | -2.43 | 44 | -1.12 ± 2.10 (-0.5) | -2.74 |
| away_off_mnf | 213 | 1.27 | +0.99 ± 0.91 (+1.1) | 1.72 | 54 | +1.73 ± 1.69 (+1.0) | 0.82 |
| three_away | 1548 | 0.4 | -0.21 ± 0.33 (-0.6) | 2.38 | 272 | +0.56 ± 0.77 (+0.7) | 2.02 |
| off_ot | 548 | 0.6 | +1.13 ± 0.60 (+1.9) | 0.42 | 72 | +1.21 ± 1.61 (+0.8) | -0.17 |
| bye | 594 | 1.11 | +0.41 ± 0.53 (+0.8) | 0.82 | 86 | +1.72 ± 1.10 (+1.6) | 1.09 |
| playoff_bye | 74 | 0.2 | -0.86 ± 1.40 (-0.6) | 6.54 | 6 | +2.25 ± 7.17 (+0.3) | 7.75 |
| super_bowl | 158 | 0.51 | +2.22 ± 1.05 (+2.1) | 0.32 | 23 | -0.13 ± 1.69 (-0.1) | 0.48 |
| travel_2000 | 534 | 0.2 | -0.32 ± 0.59 (-0.5) | 2.25 | 104 | -1.37 ± 1.32 (-1.0) | 0.63 |
| short_trip | 469 | 0.21 | +0.45 ± 0.61 (+0.7) | -2.08 | 81 | -1.48 ± 1.41 (-1.1) | -1.41 |
| tz_10am | 486 | 0.34 | +0.73 ± 0.60 (+1.2) | 2.17 | 91 | +0.01 ± 1.45 (+0.0) | 1.25 |
| night_tz | 498 | 0.67 | +1.98 ± 0.59 (+3.4) | 0.18 | 113 | +2.20 ± 1.05 (+2.1) | 0.67 |
| two_tz | 139 | 0.4 | -1.25 ± 1.05 (-1.2) | 2.15 | 26 | +0.52 ± 2.00 (+0.3) | 2.4 |
| bounce_back | 951 | 0.51 | +0.58 ± 0.41 (+1.4) | -3.1 | 151 | -0.15 ± 1.00 (-0.1) | -3.52 |
| w_warm_cold | 95 | 0.12 | +0.50 ± 1.16 (+0.4) | 3.1 | 20 | +5.80 ± 2.66 (+2.2) | 4.35 |
| w_dome_cold | 22 | 0.07 | +1.84 ± 3.23 (+0.6) | 5.02 | 1 | – | – |
| w_rain | 139 | 0.05 | -2.37 ± 1.04 (-2.3) | -2.81 | 33 | +0.74 ± 2.37 (+0.3) | -0.53 |

## 6. POST-HOC: night-game time-zone factor (strongest in-sample factor)

Selected because it had the largest 2003-22 t; 2023-25 is the check. Cuts below are descriptive. 'toward' = residual (margin − spread_line) in favour of the more western team.

| period | resid toward west ± SE |
|---|---|
| 2003-2022 (Walters in-sample) | +1.98 ± 0.59 (n 498, cover 0.560) |
| 2003-2012 | +2.47 ± 1.05 (n 187, cover 0.578) |
| 2013-2022 | +1.69 ± 0.70 (n 311, cover 0.548) |
| 2020-2022 | +1.87 ± 1.23 (n 100, cover 0.560) |
| 2023-2025 (out-of-sample) | +2.20 ± 1.05 (n 113, cover 0.565) |

| cut | 2003-22 | 2023-25 |
|---|---|---|
| west_home | +2.09 ± 0.77 (n 283, cover 0.564) | +1.90 ± 1.32 (n 63, cover 0.597) |
| west_away | +1.85 ± 0.92 (n 215, cover 0.555) | +2.57 ± 1.70 (n 50, cover 0.522) |
| kick_ge_20:00 | +1.91 ± 0.59 (n 490, cover 0.556) | +2.10 ± 1.06 (n 110, cover 0.566) |
| units<=2 | +3.43 ± 1.38 (n 73, cover 0.614) | +3.20 ± 2.63 (n 10, cover 0.500) |
| units_3 | +1.95 ± 0.76 (n 313, cover 0.552) | +2.56 ± 1.40 (n 66, cover 0.581) |
| units>=5 | +1.15 ± 1.27 (n 112, cover 0.545) | +1.28 ± 1.90 (n 37, cover 0.556) |
| excl_SEA | +1.79 ± 0.63 (n 453, cover 0.546) | +2.06 ± 1.09 (n 104, cover 0.556) |
| regular_season | +1.84 ± 0.61 (n 468, cover 0.554) | +1.93 ± 1.06 (n 107, cover 0.559) |

By season: 2020: +4.52 ± 2.30 (n 30, cover 0.667); 2021: +0.47 ± 2.17 (n 35, cover 0.486); 2022: +1.00 ± 1.92 (n 35, cover 0.543); 2023: +2.20 ± 1.88 (n 42, cover 0.500); 2024: +0.54 ± 1.59 (n 36, cover 0.514); 2025: +3.90 ± 1.92 (n 35, cover 0.686)

- 2020-2022: early→close move toward west -0.212 ± 0.219 pts (n 100); bet west early at best price: CLV -0.0444 (t -3.5), excess -0.0125 (t -1.0), ROI +0.075 ± 0.095, n 100
- 2023-2025: early→close move toward west +0.211 ± 0.166 pts (n 113); bet west early at best price: CLV -0.0208 (t -2.1), excess +0.0100 (t +1.0), ROI +0.129 ± 0.089, n 113

## Verdict

**Mostly priced, at least by now.** On 2003-2022, the period the published values were built on, the residual slope on H is +0.96 ± 0.23. Back then the market ignored these factors almost completely, and the effect was strongest in 2003-12 (+1.25). The slope falls to +0.73 in 2013-22 and to −0.19 ± 0.54 in 2020-22. Out of sample (2023-25) it is +0.34 ± 0.48. That is too noisy to rule out either 0 or 1, but it is +0.08 once the night-game factor is removed. Using −110 at the close, betting the H side covers 54.1% ± 3.4 (|H| ≥ 1), 50.5% (≥ 1.5) and 50.0% (≥ 2) in 2023-25. None of these beats 52.4%. The early-week line drifts slightly toward H (6-11% of H, t ≤ 1.3). Betting the H side early at the best allowed-book price loses 1.9-3.4% CLV against the honest close. Its directional excess over the two-side baseline is about +1% in 2023-25 (t ≤ 2.2 at |H| ≥ 0.5) and about 0 in 2020-22. Adding H at its published weight changes the 2023-25 MSE by −0.08 ± 0.73 for our model and −0.22 ± 0.72 for the close; with a fitted weight the change is ≈ 0 or worse. **Do not add the composite H, or the components as a block, to the system.**

**One lead: night-game time zones (post-hoc).** night_tz is the only factor that passes a Bonferroni-style bar in sample (t +3.4). It replicates out of sample: the more western team in night games beats the closing spread by +2.20 ± 1.05 points (n 113, 56.5% cover). The effect is positive in every season from 2020 to 2025, for both home and away western teams, and with SEA excluded. However, it does not follow Walters' unit scale: the 1-2 unit gaps show the largest effect and the 5-6 unit gaps the smallest. The early line does not move toward it either (excess CLV t ±1.0). The only evidence is results-based, about 37 bets a season, so it is still unproven. If pursued, freeze a simple rule (back the more western team in ≥ 7pm ET games between different zones, at the best price) as a paper track and judge it on 2026+ results. The repo's earlier "body clock" test was a model log-loss test, not this ATS interaction.

**Other factors.** These are noise or failed to replicate: turf (in sample t +2.6, OOS −0.26), super_bowl (in sample t +2.1, OOS −0.13), off_ot, bye, and warm-weather-in-cold (OOS +5.8 ± 2.7 on only 20 games, in sample +0.5). Rain points the wrong way in sample (−2.37, t −2.3).

