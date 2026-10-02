# NFL derivatives: 1H spreads, 1H totals and team totals vs the full-game market

Code: `scripts/research/derivatives_v2.py`. It imports the `fit` stage of `scripts/research/derivatives.py`, which was not modified. Numbers: `output/research/derivatives_v2.json`. Frozen rules: `output/research/derivatives_frozen.json`.

Protocol: the distributions and mean models were fit on nflverse 2012-2022 closing lines and results. Style tests used 2012-2022, with 2023 as a check. Rules were developed on 2023 only, three were frozen, and 2024-2025 was run once.

## Data and coverage
- The data has two snapshots per game: **early** (Fri 21:40 UTC for Sunday games, otherwise kickoff minus 24h) and **close** (kickoff minus 75 min).
- **Sharp derivative book:** betonlineag posts all three markets. lowvig posts none.
- **Allowed books that post these markets:** DK, FD, MGM, Caesars (williamhill_us) and BetRivers. ESPN BET, Fanatics and Hard Rock are not in the feed.
- **Coverage gets thinner after 2023.** In 2025, team totals came only from BetOnline, BetRivers, FanDuel and part of Caesars' slate.
- **Fair-value inputs:** the sharp full-game home margin (S) and total (T) at the same snapshot (lowvig/BetOnline, price-aware `margin_total`, `totals_dist`). These feed E[1H margin], E[1H total] and E[team points]. Key-number pmfs give P(win) and P(push).
- **1H ties:** 7.3% of 2012-22 first halves were tied. The model predicted 7.5% for 2023, and the actual rate was 7.0%.

## (a) Team style
These are walk-forward features over the previous 16 games, using only earlier games: scripted (first 15 plays) EPA, 1H-minus-2H EPA, 1H net EPA, 1H pace, and prior 1H residuals. None reached |t| ≥ 3 in 2012-22, which was the pre-set adoption bar.
- The best feature was 1H pace for the 1H total: t = 1.96 in 2012-22 and t = 2.1 in 2023, with an effect of about 0.3 points per SD.
- The joint R² was 0.1% for the 1H margin and 0.3% for the 1H total.
- **No style adjustment was used.** "Fast starters" are not a usable edge once the full-game line is known.

## Descriptive stats (allowed books plus BetOnline)

**Team totals vs the book's own full-game arithmetic, (total ± spread)/2**

| Measure | 2023 | 2024-25 |
|---|---|---|
| Line off by ≥ 0.5 | 43-65% | 43-62% |
| Line off by ≥ 1 | 6-31% | 6-28% |
| Price-implied mean vs the book's own full-game fair, off by ≥ 0.5 | 45-62% | 50-58% |
| Price-implied mean vs the book's own full-game fair, off by ≥ 1 | 7-23% | 4-18% |

- Most line gaps are offset by juice.
- The home and away team totals add up to within about 0.3 of the book's own total. DK and BetRivers sat about 0.3 low in 2023. Gaps of ≥ 1 were 0-6% at soft books and 13-18% at BetOnline.

**1H markets vs the book's own full-game fair**

| Market | 2023: off by ≥ 0.5 | 2023: off by ≥ 1 | 2024-25: off by ≥ 0.5 | 2024-25: off by ≥ 1 |
|---|---|---|---|---|
| 1H spread | 39-48% | 8-15% | 44-51% | 11-17% |
| 1H total | 27-35% | 5-7% | 19-34% | 1-4% |

- Every book prices the 1H home margin about 0.4 points below the model.
- At the close, home teams beat the derivative consensus by +0.66 (2023) and +0.36 (2024-25) points. This is not significant (t ≈ 1) and was not tested as a rule.

**Staleness at the early snapshot**
- Regressing the derivative's implied mean on the current and the previous (about Tuesday) full-game fair puts 0-25% of the weight on the old line in 2023. In 2024-25 it is 15-25%, for example FD, BetRivers and Caesars team totals at t ≈ 4.
- By the close, derivatives move 17-43% of the early model gap toward the model.
- This lag is real, but it is smaller than the vig. "Stale" rules had a 2023 CLV of -1.7% to -2.8%.

## Dev 2023 (≈ 300 candidate rules)
- **Model-based fairs** (sharp full-game → derivative, with or without a market level anchor) had negative CLV at every EV threshold from 0 to 6%: -1.2% to -3.5% across 75-680 bets. CLV does rise as model EV rises, so the model has some information, but not enough to beat soft-book vig.
- **Same-point outliers** worked: a soft-book price compared with the no-vig median of other books quoting the same line.
- **Converting prices across points with the key-number pmfs is noisy.** For example, at team total 10 the model and the market disagree by about 3% probability mass. So consensus EV uses same-point prices only.

## Frozen rules and the 2024-25 holdout (run once)
Bets per week are averaged over the 44 weeks with quotes. "±" values are standard errors.

| Rule | 2023 dev: n, CLV, ROI | 2024-25 n (per week) | CLV | ROI | Hit rate |
|---|---|---|---|---|---|
| **R1** close; EV ≥ 1% vs the leave-one-out same-point no-vig of ≥ 3 other books | 58, +0.96%, +21.5% | 48 (1.1) | **+0.86% ± 0.16** | +2.3% ± 14.5 | 51% |
| **R2** early; EV ≥ 1% vs same-point consensus, and the full-game model agrees (EV ≥ 0) | 31, +3.8%, +3.0% | 86 (2.0) | -0.06% ± 0.51 | +4.2% ± 10.6 | 53% |
| **R3** early; EV ≥ 8% vs the sharp-full-game model only | 29, +0.9%, +11% | 27 (0.6) | -0.27% ± 1.41 | +5.3% ± 18.8 | 54% |

- R1 held up in both seasons: CLV was +0.76% in 2024 and +1.01% in 2025.
- **R1's CLV is not real closing-line value.** It is measured against the consensus at the same moment (kickoff minus 75 min), so it shows a price advantage, not that the bet beat the true close.
- R2 and R3 fail. Pricing derivatives off the full-game line does not beat soft books out of sample.

## Recommendation
- **R1 is the only rule worth a cheap paper track.** Its edge is about 1-2% EV on roughly 1 bet per week, at books where derivative limits are low. It is not worth real money until a true kickoff-close CLV is positive over at least 100 bets.
- **R1 exactly:** at kickoff minus 75 min, take the best price at an allowed book on a 1H spread, 1H total or team total when it has EV ≥ 1%. EV is measured against the median no-vig of at least 3 other books quoting the same point. One bet per game-market.
- **Live data cost:** the per-event endpoint `/v4/sports/americanfootball_nfl/events/{id}/odds?markets=spreads_h1,totals_h1,team_totals&regions=us` costs 3 credits per call. The `us` region covers DK, FD, MGM, Caesars, BetRivers, BetOnline, Bovada and BetUS.
  - One bet call and one grading call near kickoff per game ≈ 2 × 16 × 3 ≈ **100 credits/week**.
  - Adding `us2` doubles that to about 200 per week.
  - Backfilling history costs 10 times as much.
- **Drop R2, R3, the style adjustments and the stale or own-line-inconsistency angles.**
