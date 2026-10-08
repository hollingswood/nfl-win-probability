# Props-implied team scoring vs team/game totals (round 3)

**Verdict: FAIL, and not close.** None of the four rules pass. Player props (anytime TD, or passing plus rushing yards) do
disagree with the team-total market. The gap has a standard deviation of about 1.3 points on the TD version and 2.0 points on the yards version. The disagreement does not predict
where the market closes. Betting toward the props loses about the vig: CLV is -3.2% to -4.1% in the 2024-25
holdout, the same as betting the sign of every gap (-3.3% to -4.0%). There is no dose-response.
The lead-lag test points the other way: the **props move toward the team-total market** (43% of the TD gap
closes from the props side by kickoff), while team totals do not move toward the props (slope 0.005 ± 0.012).
**There is no live rule and no data cost to pay.**

Script: `scripts/research/round3/props_implied_totals.py` (rules, calibration, split and pass criteria are declared
in its docstring, written before any outcome or CLV was computed). Data: `props_implied_totals_calib.json`
(2023 constants plus in-sample results) and `props_implied_totals.json` (holdout, run once).

## Design (pre-declared)

- **Props-implied team points, TD version.** Each anytime-TD Yes price gives lam_raw = -ln(1-p). The fair rate is
  lam = k_b · lam_raw^gamma. Here k_b is a per-book factor and gamma is one global exponent, both fit by maximum
  likelihood on 2023 TD outcomes. Six core books quote anytime TD in every season. A player's lam is the median
  over core books, and needs at least 2 books.
  The team value is the sum over the **top 9** players. The number of players listed fell from about 13.5 to 12.5
  per team between seasons, so the tail is cut to keep the level comparable.
  Points = PPTD · r · lam9 + (FG points as a linear function of the team total) + c0.
  PPTD and r are fit on 2023 outcomes; c0 is set so the 2023 mean gap vs the market is 0.
- **Yards version.** Starting-QB pass yards plus the top-2 rush-yards lines, each price-adjusted to a median
  (sigma 60 / 20). These map to points through an OLS fit of the team total on yards (2023).
- **Market.** The team total is the median implied mean over all books, using the key-number pmf from
  `derivatives.py`. The game total comes from `nflpred.totals.TotalDist`.
- **Rules.** All bets are placed at the early snapshot (Fri 21:40 UTC, or kickoff-24h), flat 1u, at the best-EV
  Arizona-legal quote.
  - P1: if the team gap is ≥ +1.5, bet the team-total OVER; if ≤ -1.5, bet the UNDER.
  - P2: if the sum of both teams' gaps is ≥ +2.5, bet the game-total OVER; if ≤ -2.5, bet the UNDER.
- **CLV.** The close price is the median no-vig price among close books at the **same number**. If no close book
  quotes that number, the close comes from the pmf at the close consensus. Results are graded on final scores.
- **Pass test.** Holdout n ≥ 100, CLV > 0 at one-sided p < 0.05/4 (clustered by game), and CLV > 0 in 2024 and in 2025.

2023 calibration:
- k_b ranges from 0.82 (BetRivers) to 0.98 (FanDuel), with gamma = 1.17. The mean raw implied TD probability is
  22.1%; after calibration it is 17.1%, against an actual rate of 17.1%.
- PPTD = 7.00 and r = 0.96. FG points are flat in the team total (slope 0.003).
- c0 = +0.75: the structural props estimate is 0.75 points below the market on average.
- Yards map: TT = -6.1 + 0.089 · yards.

## Rules: CLV and ROI (± clustered SE)

| Rule | Season | Bets | CLV | ROI | Overs |
|---|---|---|---|---|---|
| P1_TD team total | 2023 (in-sample) | 123 | -2.5% ± 0.4 | -7.1% ± 8.8 | 59% |
| | 2024 | 160 | -3.6% ± 0.3 | -15.0% ± 7.6 | 13% |
| | 2025 | 354 | -4.4% ± 0.2 | -10.9% ± 5.1 | 3% |
| | **2024-25** | **514** | **-4.1% ± 0.2 (p≈1)** | **-12.2% ± 4.2** | 6% |
| P1_YD team total | 2023 (in-sample) | 206 | -2.8% ± 0.3 | -3.9% ± 6.8 | 49% |
| | **2024-25** | **494** | **-4.1% ± 0.2 (p≈1)** | **-8.7% ± 4.5** | 16% |
| P2_TD game total | 2023 (in-sample) | 45 | -3.0% ± 0.7 | -2.0% ± 13.9 | 69% |
| | **2024-25** | **284** | **-3.3% ± 0.3 (p≈1)** | **-5.2% ± 5.7** | 2% |
| P2_YD game total | 2023 (in-sample) | 62 | -2.4% ± 0.6 | -12.2% ± 12.0 | 52% |
| | **2024-25** | **199** | **-3.2% ± 0.3 (p≈1)** | **-7.4% ± 6.8** | 0% |

Every rule has negative CLV in every season. Pass = false for all four.

Other variants, not tested:
- **Gap sign only (threshold 0), 2024-25.** CLV is -3.9% (P1_TD), -4.0% (P1_YD), -3.6% (P2_TD) and -3.3% (P2_YD).
  This is the vig baseline. The thresholded rules do no better than it.
- **Only the books in `my_books.json` (DK/FD/ESPN).** CLV is worse, from -3.6% to -5.3%. In 2024-25, DraftKings
  has no team totals in the feed.

## Lead-lag: who moves toward whom (early → close)

Each regression has an intercept, so its slope measures the cross-sectional relation and is not affected by the
level drift described below.

| Level / signal | Period | n | Market toward props (beta_mkt) | Props toward market (beta_props) | Difference (bootstrap) |
|---|---|---|---|---|---|
| team, TD | 2023 | 548 | 0.023 ± 0.020 | 0.328 ± 0.041 | -0.30 ± 0.05 |
| team, TD | 2024-25 | 1100 | 0.005 ± 0.012 (p 0.35) | 0.435 ± 0.026 | -0.43 ± 0.03 |
| team, yards | 2024-25 | 1014 | -0.001 ± 0.008 | 0.035 ± 0.013 | -0.035 ± 0.017 |
| game, TD | 2024-25 | 548 | -0.003 ± 0.019 | 0.436 ± 0.030 | -0.44 ± 0.04 |
| game, yards | 2024-25 | 457 | 0.054 ± 0.019 (p 0.002) | 0.132 ± 0.033 | -0.08 ± 0.04 |

The pre-declared test for "the market follows the props" needs beta_mkt > 0 **and** beta_mkt − beta_props > 0.
It fails in every cell. The one significant market slope is the game total on the yards signal. A 2.5-point gap
moves the total by about 0.14 points, which is far less than the vig, and that slope is also smaller than the
reverse one. Noise in the early consensus inflates both slopes equally, which is why the two directions are compared.

## Other findings

- **The props and team totals mostly agree.** Their correlation is 0.94-0.96 (TD version) and 0.80-0.87 (yards
  version). The team total is more accurate than either props-implied number: mean absolute error vs final points
  is 7.18 against 7.40 (TD) and 7.51 (yards) in 2024-25.
- **The gap does not predict final scores.** The slope of (points − team total) on the gap is -0.02 ± 0.20 for the
  TD version. For the yards version it is -0.20 ± 0.14, which has the wrong sign.
- **The level drifts.** With the 2023 anchor, the TD props imply 0.8 points (2024) and 1.8 points (2025) *less*
  than the team total. The yards version shows -0.9 and -1.5. The drift is also there at close. As a result the
  holdout rules were almost all UNDERs (94-100%), and they lost on results too: P1_TD ROI is -12% ± 4.
  Either anytime-TD prices got more expensive relative to 2023 or team totals rose; this test does not separate
  the two. A live version would need a rolling, market-only re-anchor. The level-free lead-lag slopes are about 0
  anyway, so re-anchoring would not create an edge.

## Notes

- **Inputs.** The script uses the matched props cache from `props_full.py prep` and the team-total cache from
  `derivatives.py build`, both in the session scratch directory. It stops if the props cache is missing.
  Game totals come from `data/historical_odds/totals` plus `dense`, and use the nearest snapshot within
  [-10, +70] min of the props request. Bet lists are saved to `scratch/round3/bets_*.csv`.
- **Team assignment.** A player's team is his team in this game if he played. Otherwise it is his team in his most
  recent *earlier* game, so there is no look-ahead. The players that cannot be assigned are mostly defenders,
  D/ST and "No Touchdown" entries, which the top-9 cut removes anyway.
- **Holdout run.** The first `holdout` call stopped with a KeyError while reading the calibration JSON, before any
  2024-25 data was loaded or computed. After a one-line key fix it ran once, and the holdout guard now refuses
  another run.
- **Data cost.** None is needed, since the test failed. For reference, a live version would cost about 1 Odds API
  credit per market per event per snapshot: anytime TD, plus pass_yds and rush_yds for the yards version, plus
  team_totals.
