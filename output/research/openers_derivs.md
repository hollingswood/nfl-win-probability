# Derivatives and alternate spreads at the Tuesday open

Code: `scripts/research/openers_derivs.py` (imports, does not modify, `derivatives.py`, `derivatives_v2.py`,
`buy_points.py`, `src/nflpred/margin_total.py`). Numbers: `openers_derivs.json`. Frozen rules:
`openers_derivs_frozen.json`. Protocol: descriptives + ~200 derivative and ~200 alt-spread candidate rules on 2023 only;
3 rules frozen; 2024-2025 run once. Pass bar per rule: pooled CLV > 0 with one-sided p < 0.05/3 (game-clustered SE)
AND CLV > 0 in 2024 and in 2025.

Snapshots: **open** = Tuesday 14:10 UTC (new), early = Fri 21:40 UTC / kick-24h, close = kick-75 min.
Fair at open = sharp (lowvig/BetOnline) full-game spread+juice and total at the SAME Tuesday snapshot (gap 0 h for
every open quote; sharp lines present for 99%). Derivative CLV = EV vs the close of the same derivative (median no-vig
of all books at the same point; else close consensus through the fitted pmf). Alt-spread fair = sharp main spread+juice
under the total-aware margin model (posted total of the same snapshot); alt CLV = EV vs the median two-way no-vig of
all books quoting the same (side, point) at the close (100% coverage), with the model close as a second view.

## Test 1 -- 1H spreads / 1H totals / team totals at the open

| 2023 (same book at all three snapshots) | open | Friday | close |
|---|---|---|---|
| mean abs dev of implied mean vs anchored sharp-FG fair: 1H spread / team total / 1H total | 0.42 / 0.58 / 0.41 | 0.38 / 0.53 / 0.40 | 0.39 / 0.53 / 0.37 |
| mean abs dev vs other books (same snapshot) | 0.21 / 0.26 / 0.18 | 0.20 / 0.27 / 0.19 | 0.21 / 0.27 / 0.22 |
| best allowed book, EV >= 1% vs same-point other-book no-vig (share of game-markets) | 1-4% | 3-10% | 4-12% |

* Opening derivative prices are only marginally softer vs fair (+0.02-0.05 pts) and are **not** more dispersed across
  books; with fewer books posting on Tuesday, shopping value is lower than at the close. 2024-25 looks the same.
* Open deviations do partly close: share of the open gap to the FG-implied fair closed by kickoff (controlling for the
  full-game move) = 1H total 0.69 / 1H spread 0.43 / team total 0.26 (2023); 0.56 / 0.26 / 0.23 (2024-25)
  -- larger than the Friday-to-close 0.17-0.43 found before. But the open fair itself is noisier (Tuesday lines move),
  and the EV calibration shows it does not cover the vig: open offers with model EV 0-2 / 2-4 / 4-7% had CLV
  -3.9 / -3.3 / -1.0% (2023) and -2.8 / -2.8 / -2.3% (2024-25); only EV >= 7% was ~0.
* Team totals vs the same book's own full-game lines at open: rules on gaps >= 0.5-1 pt had CLV -1.4% to -2.6% (n 108-353, 2023); >= 1.5 pts +0.9-1.4% on only 12-13 bets.

## Test 2 -- alternate spreads at the open

* Not mispriced more at the open. Best allowed-book alt line mean EV: open -5.1% / Fri -4.0% / close -3.9% (2023);
  -4.5 / -3.8 / -3.8% (2024-25). Same-line cross-book no-vig range is narrower at the open (median 2.1 vs 2.8 probability pts, 2023).
* **Alt dog +7 / +7.5 / +10 / +10.5 at open: no +EV region.** EV -4.1% to -5.9% (2023), -4.8% to -6.7% (2024-25);
  0-1.4% of lines positive; CLV -5% to -6%.
* Lines with positive model EV at open do get re-priced (same book/line, EV +3.4% at open -> -0.9% at close in
  2023; +4.0% -> -0.1% in 2024-25), but most of that is fair-value noise at the open.
* margin_total calibration at close (2023): bought alt lines win 1-3.5 pts more than predicted, sold lines 1-3 pts
  less at 6-7 pts off the main line (same direction as buy_points.md).

## Frozen rules and the one-time 2024-2025 holdout

| rule | 2023 dev n, CLV ± SE | 2024-25 n (bets/wk) | CLV ± SE | one-sided p | 2024 / 2025 CLV | ROI ± SE | pass |
|---|---|---|---|---|---|---|---|
| O1 1H total, EV >= 4% vs anchored sharp-FG fair, open | 17, +3.2 ± 1.8% | 56 (1.3) | +0.50 ± 0.90% | 0.29 | -3.6% / +1.4% | +12.8 ± 12.7% | no |
| O2 cross-book same-point EV >= 1% (>= 3 ref books), open, all derivs | 11, +2.7 ± 2.2% | 22 (0.5) | -1.11 ± 1.68% | 0.75 | +2.2% (n=2) / -1.4% | -16 ± 21% | no |
| O3 alt spread, model EV >= 3% (within 7 pts), open | 56, +1.9 ± 1.5% | 207 (4.7) | +0.87 ± 0.66% | 0.094 | +0.2% / +1.3% | +21.7 ± 11.0% | no |

* O3 is 206/207 **sold** lines (alt favourite laying ~6.4 more points at a median +230), 69% FanDuel. Wins 78 vs 66.2
  expected; the ROI is that luck, not CLV (+0.9%). The model over-rates exactly these lines.

## Verdict

The Tuesday open does not make derivatives or alt spreads beatable. Derivative opens drift toward the full-game-implied
fair, but the opening fair is too noisy and the move too small to clear the 4-5% hold. Opening alt ladders are no softer
than later ones. The +7/+10 alt dog has no +EV region. **No rule passes.** The only live derivative idea is still
derivatives_v2 R1 (cross-book shopping 75 min before kickoff). Optional zero-stake paper track: **O3**. Stop it
unless CLV t > 2.4 after about 150 more bets. It needs the alt-spread call at the Tuesday snapshot, which costs about
1-2 credits per game.
