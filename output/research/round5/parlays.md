# Parlays: straight and round robin (pre-registered 2026-10-15)

Rules: `parlays_prereg.json`. Code: `scripts/research/round5/parlays.py`. Parlay odds = product of leg odds (DK, FanDuel, theScore).

## P1 — ordinary legs (closing spreads at −110, 2021–25)
Cover rate of a random side: college 48.9% (3,944 games), NFL 49.4% (1,424). Singles −6.7% / −5.7%.

| | 2-leg | 3-leg | 4-leg | 5-leg | 6-leg | RR 4 by 2 | RR 5 by 3 | RR 6 by 3 |
|---|---|---|---|---|---|---|---|---|
| expected at 50% legs | −8.9% | −13.0% | −17.0% | −20.8% | −24.4% | −8.9% | −13.0% | −13.0% |
| college realized | −13.2% | −13.6% | −19.4% | −18.7% | −18.0% | −12.5% | −16.7% | −18.5% |
| NFL realized | −9.9% | −14.7% | −21.5% | −22.5% | −31.3% | −10.0% | −15.1% | −14.4% |

The vig compounds: every leg added makes the bet worse. A round robin is just several small parlays, so it loses at the
rate of its parlay size.

## P2 — legs with a real price edge (college shop vs sharp at your books, research bets 2021–25, 655 legs, ~7 a week)
Per 1 unit of total weekly stake, holdout 2023–25:

| format | realized ROI | CLV-implied ROI | losing weeks | largest drawdown (weeks of stake) |
|---|---|---|---|---|
| singles (all legs) | −4.1% | +0.7% | 61% | 3.8 |
| straight 2-leg (all pairs) | −15.0% | +1.9% | 67% | 10.2 |
| straight 3-leg | −22.1% | +5.0% | 74% | 14.6 |
| straight 4-leg | −25.7% | +7.5% | 78% | 18.6 |
| top 6 singles | +1.7% | +3.0% | 47% | 3.3 |
| top 6 RR by 2s | +8.6% | +6.1% | 56% | 5.6 |
| top 6 RR by 3s | +21.7% | +9.3% | 64% | 8.2 |
| top 5 RR by 3s (singles 3.1% CLV) | −3.5% | +9.8% | 75% | 17.2 |

- With a real edge, parlays multiply it (CLV-implied ROI roughly doubles per extra leg), but they multiply the swings
  more: drawdowns 2–5x singles and most weeks lose. No format met the pre-registered bar (better CLV, better results AND
  drawdown <= 1.5x singles): FAIL for all 8 formats. The +21.7% for "top 6 by 3s" comes from 36 weeks with a drawdown of
  8 weeks' stake; development 2021–22 showed −9% for the same format.
- Practical: parlaying edge bets is a bankroll-volatility choice, not extra edge per dollar risked at sensible sizing;
  parlaying ordinary bets just adds vig.

## P3 — same-game correlation (information only; standard parlays don't allow these)
College games where the spread is 30%+ of the total: favorite + over hit 29.1% (2014–20) and 30.7% in 2021–25 (906 games)
vs 27.5% that a +264 parlay needs (+11.7% at standard pricing, p 0.017); underdog + under 29.3%. Books know: same-game
parlays are priced with this correlation (much shorter than +264) and standard parlays block same-game legs. NFL: best
dev pairing (underdog + under) had only 57 holdout games, p 0.29.
