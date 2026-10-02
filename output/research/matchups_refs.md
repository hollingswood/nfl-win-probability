# Positional matchups & referee crews (2026-10-02)

Code: `scripts/research/matchups_refs.py` (stages `build`, `dev`, `model`, `freeze`, `holdout`).
Numbers: `matchups_refs.json`; frozen rules: `matchups_refs_frozen.json` (frozen before the single 2023-25 run).

## What was built

**Unit ratings** (per team, walk-forward): decayed sums over the team's strictly earlier games (half-life 8 games),
shown as a deviation from the league rate and shrunk toward it with a pseudo-count (60-200 plays). The league rate
comes from the last 256 team-games before the Tuesday of game week, so every rating is known at the Tuesday
snapshot. Units:
- Pass protection vs pass rush: sacks, hits, and pressure per dropback. Pressure comes from nflverse
  participation `was_pressure`, available 2016+.
- Passing: pass EPA per dropback; EPA when kept clean vs when pressured.
- Running: rush EPA and success rate, split into inside runs (guard/middle) and outside runs (tackle/end), plus each
  offense's outside-run share.
- Big plays: deep-pass value (air yards 20+), deep-attempt rate, explosive-play rate, YAC EPA.
- No-huddle rate.
- Blitz rate and EPA vs the blitz, from FTN `n_blitzers` (2022+).

**Matchup features.** For each offense vs the opposing defense: additive (`_add` = o+d), interaction (`_int` = o×d),
compounding mismatch (`_mm`, `_bad` = elite rush vs bad protection), plus four style matchups:
- `gap_style`: outside-run share × defense's outside-minus-inside weakness
- `press_sens`: defense pressure rate × offense's clean-minus-pressured EPA
- `blitz_sens`: blitz rate × offense's EPA vs the blitz
- `deep_style`: deep-attempt rate × deep yards allowed

Margin features use home minus away; totals use the sum. That gives 75 features in all.

**Referee crews.** These are the head referee's walk-forward ratings (half-life 48 games): penalties, penalty yards,
DPI/defensive holding/illegal contact, offensive holding, home penalty share, and the total-points residual. They
are shrunk hard: k=16 games for penalties, 80 for the residual.

Referee assignments are **assumed public on Thursday**. Early-week referee tests therefore use only the Friday
21:40 UTC snapshot and drop Thursday games.

## Results

**Do the unit ratings work?** Yes, they measure something real. The pre-game rating correlates with the same
game's stat: offense hits 0.30, pressure 0.22, outside-run share 0.42; defense blitz rate 0.40, pressure 0.10.
Referee crews are also real and stable: the rating correlates with that game's penalties at 0.16 and with
offensive holding at 0.19.

**Residual vs the closing line (2013-19, n=1,869):**
- Only 2 of the 75 features had |t| > 1.96, which is what chance alone would produce.
- None kept the same sign and significance across 2020-22 and the holdout.
- On the holdout, 4 of 75 had |t| > 1.96, again chance level.

Examples, in points per 1 SD (dev 13-19 / dev 20-22 / holdout 23-25):

| Feature | Dev 2013-19 | Dev 2020-22 | Holdout 2023-25 |
|---|---|---|---|
| `t_sack_int` | −0.66±0.32 | −0.48±0.42 | **+0.73±0.46** (sign flipped) |
| `m_press_bad` | +1.05±0.58 | +0.20±0.42 | −0.02±0.46 |
| Unit index, margin | +0.09±0.30 | +0.02±0.45 | +0.51±0.42 |

Referee penalties vs the total residual: +0.35±0.31 / −0.22±0.46 / −0.24±0.44. The referee's own total residual:
−0.16±0.32 / −0.09±0.46 / −0.84±0.43.

**Early-week moves: the one real effect.** The additive unit features predict the move from the Tuesday sharp
line to the closing price-based fair line:
- Unit index: **+0.34±0.06 points per SD in dev, +0.39±0.05 in the holdout (t = 8)**.
- It still holds after controlling for production-model disagreement (+0.31±0.05).
- The Tuesday line under-reacts to these unit stats, and the close catches up.
- The move (~0.4 point per SD) is smaller than the vig. Even at |index| ≥ 2.5, the best-price CLV is −0.8% (dev)
  and −1.5% (holdout), compared with about −3.2% for a random side.
- Totals moves and Friday totals moves show nothing from matchups or referees: all |t| < 2 apart from FTN blitz
  (2022 only).

**Production margin model** (`experiment.py` protocol): every matchup set made validation worse.

| Set | Val 2015-19 | Holdout 2020-25 |
|---|---|---|
| Baseline | 0.6192 | 0.6209 |
| index4 | 0.6206 | 0.6201 |
| All additive | 0.6239 | 0.6200 |
| Mismatch core | 0.6235 | 0.6199 |
| All interactions | 0.6276 | 0.6204 |

Not adopted.

## Frozen rules (3) and the single holdout run (bar: p < 0.05/3)

| Rule | Dev | Holdout 2023-25 | Verdict |
|---|---|---|---|
| R1 Tue spread, unit index ≥2.5, best allowed price | CLV −0.8% (n=87) | CLV **−1.5%** (n=88, t=−1.8), ROI −6% | fail |
| R2 Under at close, low-penalty crew (z ≤ −1) | ROI +8.0% (13-19), +9.7% (20-22); Fri CLV −3.6% | ROI **−7.2%** (n=94), Fri CLV −2.9% | fail |
| R3 Under at close, `t_sack_int` z ≥ 1 | ROI +1.9% / +11.6% | ROI **−24.7%** (n=90) | fail |

Note: the frozen file's `pass_bar` text says "R3: ROI > 0". It should read "R2/R3": both rules are graded on ROI
at the actual closing juice, and the code applies that to both.

## Verdict

There is no bettable matchup or referee edge. Closing lines already price unit-vs-unit matchups. Interaction terms
add only noise. Referee crews have stable penalty habits, but those habits don't move totals.

**Paper-track:** none. The only durable finding is a timing tool: Tuesday lines lag the unit stats by about
0.4 pt per SD. If you are already betting a side for another reason, bet early when the index agrees with you and
late when it disagrees. This doesn't qualify as a stand-alone rule.
