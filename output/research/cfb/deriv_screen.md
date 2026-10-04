# College first halves and team totals (2023-25): no edge

Rules written and committed before the odds were downloaded (scripts/research/cfb/deriv_screen.py, commit 6a1f5c8). Quotes at Arizona books 48 hours before kickoff, priced two ways: against Pinnacle's own first-half or team-total line at the same number, and against Pinnacle's full-game spread and total converted with parameters fit on 2014-21 (cfb_deriv_params.json). Graded on closing line value vs Pinnacle's derivative close and on results (CFBD quarter scores).

| Rule | Bets | CLV (bets with a close at our number) | ROI | Seasons CLV+ | Verdict |
|---|---|---|---|---|---|
| D1 1H spread vs Pinnacle 1H | 6 | +1.9% (5) | -0.5% ± 45% | 2/3 | fail |
| D2 1H total vs Pinnacle 1H | 8 | +5.0% (4) | -25.2% ± 36% | 1/3 | fail |
| D3 team total vs Pinnacle team total | 19 | +0.1% (15) | -34.0% ± 23% | 1/3 | fail |
| D4 1H spread vs derived | 701 | -2.0% (273) | -3.8% ± 4% | 0/3 | fail |
| D5 1H total vs derived | 1387 | -2.7% (398) | -2.6% ± 3% | 0/3 | fail |
| D6 team total vs derived | 868 | -3.7% (181) | -2.5% ± 4% | 0/3 | fail |

Arizona books almost never hang a first-half or team-total price better than Pinnacle's at the same number (6-19 cases in three seasons). Converting the full-game line into these markets finds lots of apparent value, but those bets lost 2-4% against the closing line: the books price derivatives better than a simple conversion does. Not a paper track. Not financial advice.
