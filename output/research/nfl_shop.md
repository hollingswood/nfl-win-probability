# NFL shop vs sharp (2022-25): no new edge

Same pre-declared design as the college screen that passed (script committed before running: scripts/research/nfl_shop.py). Prices at your books or all Arizona books vs Pinnacle's no-vig line at the same snapshot, at any number, with NFL key-number pricing fit on 2015-21 closes (nfl_dist.json). Graded against Pinnacle's last pre-kickoff line.

| Rule | Bets | CLV | p | ROI | Seasons CLV+ | Points beaten vs close | Verdict |
|---|---|---|---|---|---|---|---|
| your books spread EV>=2% | 278 | +0.4% | 0.189 | +5.5% ± 5.7% | 3/4 | +0.35 ± 0.08 | fail |
| your books total EV>=2% | 338 | -0.1% | 0.582 | -3.4% ± 5.2% | 2/4 | +0.63 ± 0.08 | fail |
| your books moneyline EV>=2% | 250 | +1.5% | 0.005 | -4.6% ± 7.0% | 2/4 | — | fail |
| your books spread EV>=4% | 49 | +2.6% | 0.023 | +17.0% ± 13.6% | 3/4 | +0.60 ± 0.24 | fail |
| your books total EV>=4% | 79 | -0.7% | 0.746 | +13.7% ± 10.6% | 2/4 | +0.49 ± 0.17 | fail |
| your books moneyline EV>=4% | 53 | +1.7% | 0.068 | -21.4% ± 15.1% | 3/4 | — | fail |
| all AZ books spread EV>=2% | 407 | +0.5% | 0.105 | +5.8% ± 4.7% | 3/4 | +0.44 ± 0.07 | fail |
| all AZ books total EV>=2% | 458 | +0.1% | 0.408 | -4.4% ± 4.5% | 3/4 | +0.67 ± 0.07 | fail |
| all AZ books moneyline EV>=2% | 345 | +1.6% | 0.000 | -5.3% ± 5.9% | 2/4 | — | fail |
| all AZ books spread EV>=4% | 77 | +2.5% | 0.012 | +26.2% ± 10.5% | 4/4 | +0.65 ± 0.19 | fail |
| all AZ books total EV>=4% | 115 | -0.2% | 0.570 | +11.7% ± 8.9% | 2/4 | +0.57 ± 0.15 | fail |
| all AZ books moneyline EV>=4% | 77 | +3.2% | 0.003 | -3.0% ± 13.3% | 3/4 | — | fail |

**None of the 12 rules passes.** NFL soft books copy Pinnacle far more closely than college books do: spreads and totals bought at better numbers still end up about even with Pinnacle's close once the price is counted (CLV −0.2% to +0.5% at the 2% bar). Moneylines beat the close by about 1.6% (p 0.0005) but lost money over 2022-25 and were positive in only 2 of 4 seasons; that part is already covered by the NFL moneyline v2-v4 tracks. Spreads at the 4% bar (+2.5% CLV, 77 bets, 4/4 seasons) are the closest call and are not frozen, because the bet count is below the pre-declared 100.

Contrast with college: same method, +1.7% to +7.6% CLV on every rule. College is where the soft-book lag is.

Not financial advice.
