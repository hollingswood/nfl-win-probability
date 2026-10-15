# College totals from points per play × plays (pre-registered 2026-10-15): FAIL

Rule (Tyler): each team's points = average of (its offensive points/play × its plays/game) and (opponent's defensive
points allowed/play × its plays/game); total = both teams. Season-to-date numbers only (games before kickoff), both
teams 3+ games, FBS vs FBS. Rules: `ppp_totals_prereg.json`. Code: `scripts/research/round5/ppp_totals.py`.

| | games | MAE projection | MAE closing total | bet every game vs close | gap 5+ pts | gap 10+ pts |
|---|---|---|---|---|---|---|
| 2014–20 (development) | 3,968 | 14.06 | 13.27 | 49.1% | 46.8% (1,327) | 42.8% (243) |
| 2021–25 (holdout) | 3,158 | 13.16 | 12.58 | 50.8% | 50.1% (880) | 44.3% (122) |
| 2026 so far (info) | 173 | | | 49.7% | 51.6% (64) | 57.9% (19) |

- T2 (does the gap to the close predict the result?): development slope −0.10 (wrong way), holdout +0.07, p = 0.89 → FAIL.
- T3 (bet rule, gap threshold picked on development = every game): holdout 50.8%, needs 52.4% at p < 0.05 → FAIL.
- Against the opening total (2021–25 only; CFBD has no openers earlier): 51.5% every game, 52.5% on 80 bets at 10+ pts → no edge.
- Why: the projection is less accurate than the closing total and runs ~1 point high; the bigger its disagreement
  with the market, the worse it does. Season-to-date per-play rates over 3–6 games are noisy, and the market already
  prices pace and efficiency (with opponent adjustment, which this formula lacks).
