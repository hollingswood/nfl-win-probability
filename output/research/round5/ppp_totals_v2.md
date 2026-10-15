# College per-play totals v2 (pre-registered 2026-10-15)

Rules: `ppp_totals_v2_prereg.json` (three versions, so each must pass at p < 0.0167). Code: `scripts/research/round5/ppp_totals_v2.py`.

| version | holdout games | MAE projection / close | T2: gap predicts result (holdout) | T3: over/under rule (k from development) | 2026 so far, every game |
|---|---|---|---|---|---|
| v1 (as asked, for reference) | 3,158 | 13.16 / 12.58 | slope wrong way on dev, p 0.89 FAIL | 50.8% on 3,116 FAIL | 49.7% (173) |
| A opponent-adjusted (λ 1000) | 3,158 | 13.26 / 12.58 | slope +0.12, p 0.006 PASS | k 3: 52.6% on 1,894 FAIL | 44.5% (173) |
| B last-season blend (w 4 games) | 3,935 | 13.06 / 12.56 | slope +0.15, p 0.002 PASS | k 0: 51.9% on 3,883 FAIL | 47.5% (333) |
| C both | 3,935 | 13.16 / 12.56 | slope +0.16, p 0.0002 PASS | k 10: 52.5% on 318 FAIL | 53.5% (333) |

- What T2 means: in 2021–25, when the projection was above the closing total the game went over the close slightly more
  often; about 0.15 points per point of gap (10-point gap -> about 1.5 points). Seasons 2019, 2021, 2023, 2024, 2025 positive,
  2014–18 about zero, 2026 so far slightly negative (C: slope −0.04, n 333). The same with Bovada (2021–22) and DraftKings
  (2023–25) closes. Diagnostics after the fact (by season / line source) are information, not part of the test.
- What T3 means: the effect is too small to bet on its own at −110: the best rules win 52–53%, under the 52.4% needed
  plus significance. Against the opener (2021–25): A and C 53.0% on 1,117 / 1,269 bets at 5+ points, not significant.
- Verdict: no standalone bet. The projection carries a little information the closing total doesn't; the only plausible
  use is as a filter on bets that already have a price edge (e.g. shop-vs-sharp totals), which needs its own
  pre-registered test.
