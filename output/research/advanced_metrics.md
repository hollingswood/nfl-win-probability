# Advanced team metrics: havoc, PROE/pace, downs, red zone, drives, OL, luck, 4th downs (2026-10-02)

Code: `scripts/research/advanced_metrics.py`, run in stages: `build`, `stability`, `model`, `dev`, `freeze`, `holdout`.
Numbers are in `advanced_metrics.json`. The rules were frozen in `advanced_metrics_frozen.json` before the single
2023-25 run.

## What was built

All ratings are walk-forward. Each one is a decayed sum (half-life 8 games, the same as production) over the team's
**strictly earlier** games. It is expressed as a deviation from the league rate over the last 256 team-games before
the game-week Tuesday, and shrunk toward that rate with a pseudo-count. This is the same machinery as
`matchups_refs.py`. Every team has an offense rating (`o_`) and a defense-allowed rating (`d_`). Margin features use
home net minus away net; totals features use the sum of all four.

Leakage test: I replaced every stat from date D onward with random noise and erased the scores. The ratings for games
on D did not change (max diff 0.0, checked on 3 dates). The helper models are fit only on earlier seasons. Those are
expected conversion by down × distance (Laplace-smoothed table) and the 4th-down go probability (gradient-boosted
model on the previous 4 seasons). In 2012 these helpers are fit on 2012 itself, so 2012 is a warm-up season.

| Family | Metrics |
|---|---|
| Havoc | (TFL or sack or forced fumble or INT or pass defensed) per play; offense allowed, defense created |
| Style | PROE (nflverse `pass_oe`), neutral PROE (1st/2nd down, WP 20-80%, not the last 2 min of a half), neutral pace (seconds between snaps in the same drive, Q1-3, score within 7), no-huddle rate |
| Downs | Early-down (1st/2nd) EPA and success rate; late-down (3rd/4th) conversion rate, conversion over expected (`ld_cx`), and late-down EPA; `ldgap` = z(ld_cx) − z(early-down SR) |
| Drives | Points per drive, drive success rate (first downs per series), available-yards %, starting field position, 3-and-out rate, red-zone TD rate; `rzgap` = z(RZ TD%) − z(available yards %) |
| OL rushing | Stuff rate, adjusted line yards (FO weights: losses 120%, 0-4 yards 100%, 5-10 yards 50%, 11+ yards 0%), second-level yards, open-field yards (designed runs only) |
| Luck | Own-fumble recovery rate (net of opponents'), INT / (INT + passes defensed), close-game (≤8 points) win surplus (decayed and season-to-date), Pythagorean residual (wins − Pythagorean wins with exponent 2.37; decayed and season-to-date), `luck5` composite |
| Other | Schedule strength faced (opponents' pre-game point-differential rating), 4th-down go rate over expected, FTN interception-worthy throws (2022+, descriptive only) |

**Persistence** is measured as the correlation between the pre-game rating and the same stat in that game
(2013-25).

- Style metrics persist: no-huddle 0.60, PROE 0.39, pace 0.27.
- Drive and down efficiency are moderate: 0.23-0.29. Havoc is 0.23 and 4th-down aggressiveness is 0.18.
- OL metrics are 0.12-0.14 and late-down conversion over expected is 0.14.
- Red-zone TD rate (0.07), INT share (0.03) and fumble recovery (0.02) are essentially luck.
- Record luck adds nothing to the next margin once the point-differential rating is known:
  - Pythagorean residual: +0.15±0.17 points per SD
  - Close-game surplus: +0.02±0.18 points per SD
- Schedule strength does add: +0.48±0.17 points per SD.

## (a) Production margin model (`experiment.py` protocol; adoption bar: validation AND holdout both improve ≥ 0.001)

| Group added | Val 2015-19 | Δ | Holdout 2020-25 | Δ |
|---|---|---|---|---|
| Baseline | 0.61915 | | 0.62085 | |
| Havoc (1) | 0.61860 | −0.00055 | 0.62013 | −0.00072 |
| Havoc split off/def (2, follow-up) | 0.61840 | −0.00075 | 0.62026 | −0.00059 |
| OL rushing (4) | 0.61859 | −0.00056 | 0.62176 | +0.00091 |
| Havoc + OL (5, follow-up) | 0.61882 | −0.00033 | 0.62091 | +0.00006 |
| Fourth down (1) | 0.61913 | −0.00002 | 0.62160 | +0.00075 |
| PROE/pace (4) | 0.61952 | +0.00037 | 0.62102 | +0.00017 |
| Early/late downs (4) | 0.61943 | +0.00028 | 0.62113 | +0.00028 |
| SOS (1) | 0.61972 | +0.00057 | 0.62102 | +0.00017 |
| Record season-to-date (3) | 0.61980 | +0.00065 | 0.62089 | +0.00004 |
| Drives (6) | 0.61967 | +0.00052 | 0.62425 | +0.00340 |
| Luck composite (1) | 0.62074 | +0.00159 | 0.62201 | +0.00116 |
| Luck (6) | 0.62106 | +0.00191 | 0.62318 | +0.00233 |
| All (31) | 0.62605 | +0.00690 | 0.62764 | +0.00679 |

Havoc is the only consistent small gain: about −0.0006 in both periods. It falls short of the 0.001 bar, so **nothing
is adopted.**

## (b) Market tests

The residuals are home margin − closing `spread_line` and total − `total_line`. Each coefficient is in points per
1 SD of the feature, with HC0 standard errors. The Tuesday move runs from the Tuesday 14:10 sharp line to the
price-based closing fair line (`edge_lab.closing_fair`).

**Multiple comparisons** (57 features in dev 2012-19, 58 on the holdout):
- Chance level throughout: 2 features with |t| > 1.96 in 2012-19, 0 in 2020-22 and 1 in 2023-25.
- No feature survives Bonferroni or Benjamini-Hochberg (q = 0.10).
- None of the 16 pre-specified luck hypotheses is supported. All one-sided p > 0.07, and most have the wrong sign.

| Feature (margin unless `t_`) | Close 2012-19 | Close 2020-22 | Close 2023-25 (holdout) | Tue move 2020-22 | Tue move 2023-25 |
|---|---|---|---|---|---|
| Late-down conversion over expected | +0.21±0.28 | −0.22±0.45 | −0.08±0.43 | +0.14±0.06 | **+0.24±0.04** |
| Late-down gap (late − early) | +0.05±0.29 | −0.63±0.44 | −0.55±0.40 | −0.12±0.06 | **−0.19±0.05** |
| Red-zone TD rate | +0.12±0.28 | −0.25±0.43 | +0.58±0.45 | +0.20±0.06 | +0.19±0.05 |
| Fumble-recovery luck | −0.07±0.28 | −0.24±0.43 | +0.67±0.44 | −0.02±0.06 | +0.01±0.05 |
| INT share | +0.21±0.29 | +0.06±0.43 | +0.16±0.43 | +0.22±0.06 | +0.08±0.05 |
| Close-game surplus (decayed) | −0.19±0.28 | +0.04±0.42 | +0.75±0.43 | +0.06±0.06 | +0.10±0.05 |
| Pythagorean residual (season-to-date) | +0.24±0.27 | −0.30±0.40 | +0.04±0.45 | −0.04±0.06 | −0.01±0.05 |
| Luck composite (`luck5`) | −0.01±0.28 | −0.20±0.43 | +0.75±0.44 | +0.12±0.06 | +0.18±0.05 |
| Havoc | +0.33±0.28 | −0.03±0.45 | +0.83±0.44 | +0.18±0.06 | **+0.27±0.04** |
| Early-down EPA | +0.35±0.28 | +0.02±0.45 | +0.76±0.43 | **+0.37±0.06** | **+0.36±0.05** |
| Points per drive | +0.55±0.27 | −0.14±0.46 | +0.71±0.44 | **+0.31±0.06** | **+0.38±0.05** |
| Starting field position | +0.76±0.28 | +0.03±0.46 | +0.95±0.43 | +0.13±0.07 | +0.23±0.05 |
| Adjusted line yards | +0.15±0.28 | +0.20±0.43 | +0.55±0.43 | +0.15±0.06 | +0.23±0.04 |
| 4th-down go over expected | +0.24±0.30 | +0.46±0.46 | −0.06±0.45 | −0.09±0.06 | +0.08±0.05 |
| `t_` neutral pace (slower) | −0.30±0.29 | −0.09±0.48 | −0.53±0.42 | −0.04±0.05 | −0.03±0.05 |
| `t_` neutral PROE | −0.23±0.29 | −0.19±0.47 | −0.28±0.44 | +0.09±0.05 | +0.04±0.05 |
| `t_` offensive luck (late-down + RZ) | −0.13±0.28 | +0.05±0.46 | −0.30±0.43 | +0.04±0.05 | +0.01±0.05 |

**The closing market does not overrate luck.** Luckier teams did not under-cover in either dev period. On the holdout,
most luck coefficients are *positive*: luckier teams covered more, which is noise in the opposite direction from the
hypothesis.

**What does replicate is a Tuesday-line under-reaction.** Efficiency metrics predict the move from Tuesday to the close
at t ≈ 5-8 in both 2020-22 and 2023-25. Examples are early-down EPA/SR, points per drive, available yards %, havoc and
ALY. The close also moves *toward* teams with good late-down and red-zone luck, but *away from* late-down overperformance
relative to early downs (`ldgap`, −0.19, t −3.6). In other words, the sharp close already discounts unsustainable
third-down success. These moves are 0.2-0.4 points per SD, which is smaller than the vig. That matches what
`matchups_refs` found.

**Early-line CLV, 2020-22.** A random side at the best Tuesday spread price has CLV −3.2%. A random Tuesday under has
−1.2%.
- Fading luck at Tuesday: CLV −0.7% to −6.8% across all luck metrics and thresholds (all negative). This is worse than random for RZ
  and INT share, because the close moves toward those teams.
- Backing efficiency at Tuesday: −2.9% to +0.3%.

## (c) Frozen rules (chosen on dev only) and the single 2023-25 run. Bar: one-sided p < 0.05/3

| Rule | Dev | Holdout 2023-25 | Pass |
|---|---|---|---|
| R1: fade fumble-recovery luck at the close, \|z\| ≥ 1.5 (ATS at closing juice) | 2012-19: 312 bets, ROI +2.8%; 2020-22: 141 bets, ROI +9.7% | 121 bets, cover 48.8%, ROI −6.8% (p 0.78) | No |
| R2: UNDER at the close when combined neutral pace is ≥ +1 SD (slow) | 2012-19: 317 bets, ROI +10.1% (t 1.86); 2020-22: 94 bets, ROI +7.3% | 62 bets, 54.1% vs break-even 52.4%, ROI +3.2% (p 0.40). Tuesday CLV −2.1% | No |
| R3: back early-down EPA, \|z\| ≥ 2, at the Tuesday best price | 2020-22: 44 bets, CLV +0.3% | 48 bets, CLV −2.4% (p 0.97), ATS at close ROI −4.9% | No |

**Verdict.** None of these metrics adds information beyond the closing line. "Luck" metrics (late-down conversion,
red-zone TD rate, fumble recovery, INT share, close-game record, Pythagorean residual) are almost pure noise game to
game, and the closing market does not overrate them. Havoc is the best model addition but falls below the adoption bar.
The one robust market pattern is the Tuesday under-reaction to efficiency. It is too small to beat the vig, so treat it
as a known structural fact, not an edge. Do not adopt anything; do not paper-trade R1-R3.
