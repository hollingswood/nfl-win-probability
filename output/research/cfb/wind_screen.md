# College wind unders: pre-declared test (FAILS)

Rules written and committed on 2026-10-03, before any weather data existed (scripts/research/cfb/wind_screen.py). Weather = the GFS forecast for the kickoff window as it stood 1 or 2 days before the game (Open-Meteo Previous Runs), not the wind that actually blew. That archive only has wind forecasts from 2024 on, so the test covers 2024-25 (2,511 open-air games with odds); 2026 is out of sample.

| Rule | Bets 2024-25 | Under won | ROI | CLV vs Pinnacle close | Closing total vs our number | 2026 | Verdict |
|---|---|---|---|---|---|---|---|
| W1: 2-day forecast ≥ 15 mph, bet ~48 h out | 80 | 61% | +17.2% ± 10% | -1.0% | 0.78 lower | 9 bets, CLV -0.2% | fail |
| W2: 1-day forecast ≥ 15 mph, bet ~24 h out | 51 | 58% | +10.4% ± 13% | -1.4% | 0.70 lower | 6 bets, CLV +0.9% | fail |

The totals did keep falling after the bet (the close was about 0.7-0.8 points lower than our number), but books already charge extra on windy unders, so after the price our bets were worth about −1% against Pinnacle's close. The 61% win rate is within luck on 80 bets (about 1.6 standard errors), and the pre-declared bar required positive closing line value in most seasons (0 of 2). At the close, unders hit 54% in 15-20 mph forecasts (46 games) against 50% in calm games: little left once the closing total is set.

Not a paper track. The live forecast still shows on each college card. Not financial advice.
