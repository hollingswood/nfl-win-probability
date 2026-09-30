# NFL game totals (over/under): is there a bettable edge?

**Short answer: none that is proven.** One of the three frozen rules passed the 2023-25 bar: unders at the
Tuesday line when the game's wind turned out to be at least 15 mph. That rule knows the **recorded** game-day
wind, and a Tuesday bettor would only have a 5-day forecast. By Friday, when the forecast is good, the
edge is mostly gone (CLV +1.2%, not significant), and at kickoff it is fully priced in. Treat it as an
upper bound that still needs forward paper-trading with real Tuesday forecasts. The two rules that use
no future information (soft book vs sharp price, and a totals model that predicts line moves) failed.

Code: `scripts/research/totals.py` (stages `dev` → `freeze` → `holdout` (once) → `posthoc`).
Numbers: `output/research/totals.json`. Frozen rules: `output/research/totals_frozen.json`
(frozen 2026-09-30 22:45:20 UTC; holdout run once at 22:45:35 UTC).

## Method

* **Odds:** `data/historical_odds/totals/`, 2020-25, US region. Snapshots are Tue 14:10 UTC, Fri 21:40 UTC
  and about 75 min before each kickoff (the last one is our close). We dropped 2,049 rows with broken
  prices or off-market points (prices outside −250..+200, overround outside 1.00–1.12, or more than 5
  points from the snapshot median). Games are matched like `replay_early_lines.match_games`.
* **Total-points distribution:** Normal(σ = 13.43) on integer totals, times empirical key-number weights.
  Both are fit on 2012-19 finals against nflverse closing totals. Weights at 37/41/43/44/47/51 are about
  1.2–1.4. Calibration at the close: predicted push rate 1.42% vs 1.43% actual (dev), and 1.35% vs 1.29%
  (holdout).
* **Fair total:** For each book, the expected total E[T] whose no-vig P(over) / (P(over) + P(under)) matches
  its prices. We take the median over books, using all books and sharp books (lowvig, betonlineag, plus
  bookmaker in 2020 and circasports in 2022). The **closing fair total** comes from the last
  pre-kickoff snapshot, using sharp books (present for 100% of games), else all books.
* **CLV** of a bet at (side, point, price) = its EV under the closing fair distribution:
  P(win) × decimal price + P(push) − 1. PnL is graded on the actual total.
* **What we could bet:** allowed books present in the data were DraftKings, FanDuel, BetMGM, Caesars
  (williamhill_us) and BetRivers, plus Fanatics in 2025. One bet per game, locked at the first qualifying
  snapshot, at the best book there.
* **Leakage guards:** model features use strictly earlier games (EWMA with shift(1)). The league scoring
  level uses only games before the Tuesday of the game's week. A "Friday" snapshot counts only within
  3.5 days of kickoff; otherwise, for Thursday games, the model would know a result the Friday bettor didn't.

## 1. Data QA (2020-22 unless noted)

| | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| games matched / schedule | 269/269 | 285/285 | 284/284 | 285/285 | 285/285 | 285/285 |
| games with a Tue snapshot | 251 | 276 | 273 | 285 | 285 | 285 |
| median books per snapshot | 11 | 16 | 18 | 14 | 7 | 8 |
| close: median minutes before kickoff | 75 | 75 | 75 | 75 | 75 | 75 |

* Closing fair vs the nflverse `total_line`: they differ by 0.42 points on average (mean absolute).
  The median closing point equals `total_line` 54% of the time. The fair E[T] sits about 0.2 above the
  number because totals are right-skewed.
* **Movement from Tuesday to close** (fair E[T]):

  | | dev 2020-22 | holdout 2023-25 |
  |---|---|---|
  | mean move | −0.33 | −0.17 |
  | mean absolute move | 1.11 | 1.04 |
  | median point changed at all | 85% | 81% |
  | moved ≥ 1 point | 55% | 47% |
  | moved ≥ 2 points | 18% | 17% |
  | share of moves down / up | 50% / 34% | 45% / 39% |

  Totals drift **down** during the week, most for high totals (dev, ≥ 50: −0.73). Friday to close moves
  much less (mean absolute 0.65). The moves carry information: corr(move, final − Tuesday line) is +0.16.
* **Blind bets** at the best allowed price never have positive CLV:

  | snapshot | over | under |
  |---|---|---|
  | Tuesday, dev | −5.2% | −1.2% |
  | Tuesday, holdout | −4.3% | −2.0% |
  | close | about −3.1% | about −3.1% |

  At the close, −3.1% is just the vig. On Tuesday, unders are the cheaper side (overs get bet *down*),
  but neither side is positive.

## 2. Development (2020-22 only; 77 rule configurations tried, all in `totals.json` → `dev`)

**(a) Soft book vs sharp fair.** Allowed-book quotes whose EV under the *same snapshot's* sharp fair is
above 1%, bet Tue/Fri:

| n | CLV | t | ROI |
|---|---|---|---|
| 78 | +1.76% | 2.37 | +14% ± 11% |

The same filter at the close is circular (sharp close = the CLV reference). Against the all-book close,
it is about 0.

**(b) Wind.**
* At the close, the market *under*-reacts somewhat. Final − closing line for recorded wind 10–20 mph:
  −1.1 to −1.8 points in 2012-19, and −1.4 to −2.0 in dev.
* That can't produce CLV at the close, and close-time wind unders in the holdout were +4–6% ROI ± 9–13%:
  no edge.
* Unders at the **Tuesday** line with recorded wind ≥ 15 mph (dev):

  | n | CLV | t | ROI |
  |---|---|---|---|
  | 68 | +3.65% | 2.7 | +21% ± 11% |

  With a ≥ 15 mph threshold, the Friday line already prices it (CLV −0.9%).

**(c) Walk-forward totals model.**
* The model: ridge regression on EWMA team offensive/defensive EPA and success rate, neutral-situation
  pace (seconds per snap), plays, points, indoor, division, and league scoring level. It trains only on
  prior seasons (2013+). A weather variant uses recorded wind and cold.
* Out of sample, 2015-22: RMSE 13.73 vs the closing line's 13.33. The fitted blend weight on
  (model − line) is about 0.00–0.04, so the model adds nothing to the closing total.
* **But the model's disagreement with the Tuesday line predicts the move to the close.** The slope is
  0.105 (t = 6.8), and 0.22, 0.08 and 0.10 by season. From Friday the slope is 0.051. It predicts
  nothing beyond the close (slope 0.05, t = 0.35).

**(d) Timing.** See the blind bets above: Tuesday unders are cheaper than overs, but no blind timing
rule has positive CLV.

## 3. Frozen rules and 2023-25 results (run once)

Pass bar: mean CLV > 0 with one-sided p < 0.05/3 = 0.0167.

| rule | bets (per season) | CLV | t | p | ROI ± SE | by season CLV | pass |
|---|---|---|---|---|---|---|---|
| R1 soft vs sharp, Tue/Fri, EV > 1% | 74 (24.7) | −1.26% | −1.53 | 0.94 | +15.5% ± 10.9% | +0.2 / −2.8 / −0.7% | **no** |
| R2 Tue under, recorded wind ≥ 15 mph | 51 (17.0) | **+6.98%** | 5.14 | < 1e-6 | +20.2% ± 13.1% | +4.9 / +6.3 / +8.9% | **yes, with look-ahead** |
| R3 model predicts move (Tue/Fri, EV > 0) | 225 (75.0) | −0.11% | −0.22 | 0.59 | +6.9% ± 6.4% | +0.3 / −1.1 / +0.3% | **no** |

Post-hoc checks (labelled, `holdout_posthoc`; they change no verdict):
* **R2 depends on knowing the wind early.**

  | same wind rule | CLV | t |
  |---|---|---|
  | Tuesday, wind ≥ 12 | +2.4% | 2.6 |
  | **Friday**, wind ≥ 15 | +1.2% | 1.4 |
  | close | −2.6% (the vig) | – |

  The Tuesday CLV is the market pricing in wind *as the forecast firms up*, and the backtest gets the
  final answer for free.
* The Open-Meteo previous-runs archive, which would give real 5-day-ahead forecasts, was unreachable
  here (proxy 403), so the realistic Tuesday edge could not be measured. It lies somewhere between 0
  and +7%.
* **R3:** the model still predicts moves in the holdout (Tuesday slope 0.136, t = 9.1). But a
  0.1-point-per-point move is too small to beat the vig. The bets it triggered were mostly soft-book
  quotes that looked off-market, and like R1, they closed slightly worse.
* **R1:** in 2023-25, soft-book Tuesday quotes that looked better than lowvig/BetOnline did not hold up
  (CLV −1.0% to −1.8% across thresholds). On totals the sharp Tuesday price is not a reliable anchor.
* ROI figures are ±10–13% noise at these volumes. None of the rules' ROI is significant, and blind
  unders/overs lose about the vig.

## Verdict

* **Nothing to bet yet.** The market prices totals well, and pace and efficiency add nothing beyond the
  close.
* The only angle that passed (R2, Tuesday wind unders) assumes perfect knowledge of game-day wind. It is
  a promising lead, not a proven edge, and it is small: about 17 bets a season.
* **Next step:** start logging the Tuesday Open-Meteo kickoff forecast (`weather.py` already fetches it)
  alongside the Tuesday totals, and paper-trade "forecast wind ≥ 15 → under" for a season before
  staking. Judge it by CLV.
* Model disagreement is a real but small predictor of line moves. It could help *time* a totals bet the
  user already wants to make, but it is not a standalone edge.
