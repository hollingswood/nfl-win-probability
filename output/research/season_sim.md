# Season simulator & futures test

Generated 2026-10-01 by `scripts/research/season_sim.py` (10,000 sims per run).

## Data

* Win totals 2013-2026 with over/under juice: `data/research_futures/win_totals_2013_2026.csv`, from nfelo's open `wt_ratings.csv` (github.com/greerreNFL/nfelosrs). Pre-2025 rows are a late-preseason (~Aug 30) consensus with real juice; 2025-26 are DraftKings at the start of the season. sportsoddshistory.com (now covers.com) was blocked by the egress proxy / returned 404.

* Division odds: only the top-2 favorites per division 2022-2025, extracted by an LLM page summarizer from sportsbettingdime.com (`division_odds_top2_2022_2025_partial.csv`), unverified, single book unknown, no complete market -> informational only. No playoff-odds or in-season futures archive found.

## Simulator calibration (fit on 2015-2019)

* Market ratings half-life 10.0 wk (RMSE vs future closing spreads: 3.0: 4.57, 6.0: 4.56, 10.0: 4.45)
* model: shrink k(h)=k0+k1*h  preseason 0.77-0.014h, in-season 1.02-0.032h; rating sd tau_a pre 3.86 / in 3.71 pts, random-walk tau_b pre 0.95 / in 0.92 pts/sqrt(wk), game noise 12.1/12.1
* market: shrink k(h)=k0+k1*h  preseason 1.85-0.031h, in-season 1.39-0.029h; rating sd tau_a pre 3.36 / in 3.75 pts, random-walk tau_b pre 1.08 / in 0.84 pts/sqrt(wk), game noise 12.0/12.2

Holdout (2020-25) game-level log loss of the sim's per-game win probability by weeks ahead:

| version | start | horizon | n | log loss | fav win% | mean p(fav) |
|---|---|---|---|---|---|---|
| model | pre | h0-3 | 380 | 0.6433 | 0.624 | 0.609 |
| model | pre | h4-9 | 514 | 0.6632 | 0.595 | 0.598 |
| model | pre | h10+ | 716 | 0.6748 | 0.581 | 0.588 |
| model | in | h0-3 | 1404 | 0.6389 | 0.640 | 0.636 |
| model | in | h4-9 | 1688 | 0.6513 | 0.626 | 0.621 |
| model | in | h10+ | 627 | 0.6595 | 0.624 | 0.605 |
| market | pre | h0-3 | 380 | 0.6428 | 0.608 | 0.603 |
| market | pre | h4-9 | 514 | 0.6624 | 0.591 | 0.596 |
| market | pre | h10+ | 716 | 0.6697 | 0.588 | 0.587 |
| market | in | h0-3 | 1404 | 0.6349 | 0.637 | 0.637 |
| market | in | h4-9 | 1688 | 0.6455 | 0.623 | 0.622 |
| market | in | h10+ | 627 | 0.6615 | 0.596 | 0.600 |

## 1. Preseason win totals: model sim vs posted totals

Model = production ridge trained on seasons < S; team inputs as of the day before week 1 (Elo with 1/3 regression, QB rating of the week-1 starter + qb_change vs the team's previous QBs, prior-season EWMAs), shrunk by k(h) fit on 2015-19. Market sim = ratings fitted to prior-season spreads (offseason-discounted) + week-1 lines. Bets at the listed juice; pushes refunded. SE = naive per-bet; SEc = season-clustered.

| set | n teams | RMSE line_adj | RMSE model | RMSE market-sim | corr(model-line, actual-line) | Brier P(over): no-vig / model / mkt-sim |
|---|---|---|---|---|---|---|
| dev_2015_19 | 160 | 2.639 | 2.803 | 2.692 | -0.003 | 0.2511 / 0.2557 / 0.2500 |
| holdout_2020_25 | 192 | 2.728 | 2.879 | 2.842 | +0.022 | 0.2477 / 0.2695 / 0.2609 |

| set | rule | bets | W-L-P | ROI | SE | SEc |
|---|---|---|---|---|---|---|
| dev_2015_19 | model |diff|>=0.5 | 90 | 46-40-4 | +0.048 | 0.103 | 0.051 |
| dev_2015_19 | model |diff|>=1.0 | 38 | 23-14-1 | +0.197 | 0.156 | 0.166 |
| dev_2015_19 | model |diff|>=1.5 | 15 | 7-8-0 | -0.094 | 0.265 | 0.213 |
| dev_2015_19 | model EV>=0.03 | 122 | 64-54-4 | +0.122 | 0.094 | 0.025 |
| dev_2015_19 | model EV>=0.08 | 103 | 54-46-3 | +0.122 | 0.103 | 0.034 |
| dev_2015_19 | market_sim |diff|>=0.5 | 90 | 49-36-5 | +0.104 | 0.100 | 0.069 |
| dev_2015_19 | market_sim |diff|>=1.0 | 42 | 24-16-2 | +0.170 | 0.150 | 0.118 |
| dev_2015_19 | market_sim |diff|>=1.5 | 16 | 10-5-1 | +0.314 | 0.243 | 0.207 |
| dev_2015_19 | market_sim EV>=0.03 | 122 | 62-55-5 | +0.072 | 0.092 | 0.045 |
| dev_2015_19 | market_sim EV>=0.08 | 96 | 48-43-5 | +0.084 | 0.104 | 0.056 |
| holdout_2020_25 | model |diff|>=0.5 | 121 | 55-61-5 | -0.107 | 0.085 | 0.090 |
| holdout_2020_25 | model |diff|>=1.0 | 61 | 30-30-1 | -0.067 | 0.121 | 0.132 |
| holdout_2020_25 | model |diff|>=1.5 | 24 | 11-12-1 | -0.115 | 0.190 | 0.248 |
| holdout_2020_25 | model EV>=0.03 | 152 | 68-77-7 | -0.054 | 0.081 | 0.056 |
| holdout_2020_25 | model EV>=0.08 | 130 | 56-68-6 | -0.080 | 0.088 | 0.059 |
| holdout_2020_25 | market_sim |diff|>=0.5 | 124 | 62-56-6 | -0.017 | 0.083 | 0.072 |
| holdout_2020_25 | market_sim |diff|>=1.0 | 70 | 35-32-3 | -0.029 | 0.111 | 0.127 |
| holdout_2020_25 | market_sim |diff|>=1.5 | 27 | 18-8-1 | +0.252 | 0.166 | 0.148 |
| holdout_2020_25 | market_sim EV>=0.03 | 153 | 73-73-7 | -0.022 | 0.079 | 0.046 |
| holdout_2020_25 | market_sim EV>=0.08 | 126 | 59-62-5 | -0.035 | 0.088 | 0.055 |

Information test: (actual - line_adj) = b x (model - line_adj). Dev b = 0.02 ± 0.22; holdout refit b = 0.07 ± 0.20. Holdout RMSE line 2.728 vs blend with dev weight 2.728.

Division odds sanity (partial top-2 list, n=63): Brier implied(with vig) 0.227 vs model 0.239; model EV>5% bets 10 (won 1), ROI -0.55.

## 2. In-season: QB injuries and futures

* Market: closing spread moves 23.5 ± 0.7 pts per 1.0 EPA/dropback of starter change (n=2544). Model coefficients: qb_diff 7.9, qb_change 15.7 pts per EPA/db.
* 119 starter absences of 3+ games (2015-25, start wk 3-14; 77 lasted the rest of the season; median 7 games). Market-sized downgrade -1.4 pts/game.
* Market-implied playoff probability: stale (no QB news) 0.231 -> informed 0.205 (mean move -0.026, mean |move| 0.042); refit with the next posted line only 0.221. Realized rate 0.176.
* Brier: stale 0.0841, informed 0.0703, next-line 0.0787; paired gain +0.0139 ± 0.0052.
* Contenders (stale p 15-85%, n=50): stale 0.383, informed 0.338, realized 0.278.
* Caveats: absence length uses hindsight (actual games missed), events include benchings, and there are no historical futures PRICES, so this measures how far a stale (pre-news) market-implied price is from an informed one, not a realized betting ROI. 'next_line' shows that refitting ratings with only the first post-news spread recovers ~1/3 of the move: rating systems built from past spreads lag QB news.

2026 sensitivity (market version, backup at replacement-level QB rating -0.10): for 25 teams with 20-80% playoff odds, starter out 4 games moves P(playoffs) by median -0.048, out for the season by median -0.205 (range -0.41 to -0.08); full table in season_sim_2026.json.

## 3. 2026 projections (as of 2026-10-01, next week 4, 48 games final)

| team | W | pre total | model wins | mkt wins | model P(div) | mkt P(div) | model P(PO) | mkt P(PO) | mkt P(#1) | QB out rest-of-season P(PO) |
|---|---|---|---|---|---|---|---|---|---|---|
| BUF | 3 | 10.5 | 12.0 | 12.1 | 0.73 | 0.78 | 0.89 | 0.91 | 0.33 | 0.61 |
| KC | 3 | 10.5 | 10.7 | 11.0 | 0.51 | 0.53 | 0.76 | 0.80 | 0.17 | 0.47 |
| BAL | 2 | 11.5 | 10.4 | 11.0 | 0.39 | 0.49 | 0.71 | 0.78 | 0.16 | 0.43 |
| SF | 3 | 9.5 | 11.8 | 10.9 | 0.55 | 0.43 | 0.85 | 0.74 | 0.19 | 0.33 |
| MIN | 3 | 8.5 | 11.8 | 10.8 | 0.59 | 0.39 | 0.86 | 0.74 | 0.18 | 0.54 |
| SEA | 2 | 10.5 | 10.3 | 10.6 | 0.27 | 0.37 | 0.66 | 0.70 | 0.14 | 0.59 |
| CIN | 2 | 10.5 | 10.2 | 10.3 | 0.33 | 0.32 | 0.66 | 0.68 | 0.10 | 0.37 |
| DET | 2 | 10.5 | 10.0 | 10.4 | 0.26 | 0.35 | 0.62 | 0.67 | 0.15 | 0.32 |
| PHI | 2 | 10.5 | 9.2 | 10.1 | 0.42 | 0.55 | 0.54 | 0.66 | 0.10 | 0.37 |
| DEN | 2 | 9.5 | 9.6 | 9.9 | 0.27 | 0.29 | 0.57 | 0.63 | 0.08 | 0.39 |
| JAX | 2 | 8.5 | 10.9 | 9.5 | 0.70 | 0.46 | 0.81 | 0.60 | 0.05 | 0.37 |
| LA | 1 | 11.5 | 9.6 | 9.3 | 0.17 | 0.18 | 0.59 | 0.53 | 0.06 | 0.23 |
| CHI | 2 | 9.5 | 8.8 | 9.0 | 0.12 | 0.15 | 0.43 | 0.46 | 0.05 | 0.38 |
| NE | 1 | 10.5 | 9.1 | 8.4 | 0.21 | 0.15 | 0.53 | 0.41 | 0.03 | 0.18 |
| PIT | 2 | 8.5 | 8.4 | 8.7 | 0.13 | 0.14 | 0.38 | 0.41 | 0.03 | 0.28 |
| GB | 1 | 9.5 | 7.2 | 8.7 | 0.04 | 0.12 | 0.19 | 0.40 | 0.03 | 0.14 |
| IND | 1 | 7.5 | 7.7 | 8.2 | 0.16 | 0.27 | 0.30 | 0.40 | 0.01 | 0.27 |
| NO | 1 | 7.5 | 8.3 | 7.7 | 0.37 | 0.33 | 0.45 | 0.38 | 0.01 | 0.25 |
| HOU | 0 | 9.5 | 7.3 | 8.0 | 0.12 | 0.23 | 0.24 | 0.36 | 0.01 | 0.21 |
| DAL | 1 | 9.5 | 7.9 | 8.0 | 0.20 | 0.21 | 0.33 | 0.34 | 0.02 | 0.11 |
| LV | 3 | 6.5 | 8.6 | 8.1 | 0.19 | 0.12 | 0.42 | 0.33 | 0.02 | 0.17 |
| CAR | 1 | 7.5 | 6.8 | 7.1 | 0.18 | 0.26 | 0.23 | 0.31 | 0.01 | 0.22 |
| TB | 0 | 8.5 | 6.4 | 6.7 | 0.16 | 0.24 | 0.21 | 0.27 | 0.01 | 0.13 |
| NYG | 2 | 7.5 | 8.2 | 7.6 | 0.25 | 0.16 | 0.36 | 0.27 | 0.02 | 0.15 |
| NYJ | 1 | 5.5 | 6.0 | 7.1 | 0.04 | 0.06 | 0.13 | 0.23 | 0.01 | 0.14 |
| ATL | 1 | 7.5 | 7.8 | 6.6 | 0.28 | 0.18 | 0.37 | 0.22 | 0.01 | 0.11 |
| LAC | 0 | 9.5 | 5.8 | 6.7 | 0.03 | 0.06 | 0.11 | 0.19 | 0.00 | 0.08 |
| CLE | 2 | 5.5 | 8.6 | 7.2 | 0.15 | 0.05 | 0.40 | 0.18 | 0.01 | 0.09 |
| WAS | 1 | 7.5 | 7.3 | 6.8 | 0.13 | 0.08 | 0.23 | 0.17 | 0.01 | 0.04 |
| ARI | 1 | 3.5 | 6.0 | 6.8 | 0.01 | 0.02 | 0.08 | 0.14 | 0.00 | 0.07 |
| TEN | 0 | 6.5 | 4.3 | 5.0 | 0.02 | 0.04 | 0.04 | 0.07 | 0.00 | 0.09 |
| MIA | 0 | 3.5 | 5.0 | 3.9 | 0.02 | 0.01 | 0.06 | 0.02 | 0.00 | 0.01 |

## Verdict

* Preseason win totals: the model's projection carries no information beyond the posted total (regression weight ~0 on dev and holdout; RMSE and P(over) Brier worse than the no-vig line). Dev ROI looked positive; every model rule lost on the 2020-25 holdout. Do not bet model-vs-total disagreements.
* The only rule positive in both dev and holdout is market_sim |diff|>=1.5 (prior-season spreads + week-1 lines vs the total; 28-13-2 combined), but it is 1 of 10 rules looked at on the holdout, n is small (~1.5 SE), and week-1 CLOSING lines post-date the win-total snapshot. Treat as a lead to paper-track with timestamped look-ahead lines, not an edge.
* In-season: properly pricing a starting-QB loss moves playoff odds by ~2-11 pts (4 games, median 5) and ~8-41 pts (season, median 20) for contenders, and the informed price beats a stale one on realized outcomes (paired Brier gain ~2.7 SE). The edge window exists only if a book's futures lag the news; that cannot be verified without timestamped futures prices (start logging them alongside history/odds_*.json).
* Caveats: pre-2025 win totals are an undated late-preseason consensus (source 'not_tracked'); the model preseason state uses the actual week-1 starter (known in August in almost all cases); ties not simulated; tiebreakers approximate (common games / strength of victory omitted).
