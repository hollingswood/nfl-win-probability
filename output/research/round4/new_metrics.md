# Round 4: new advanced metrics (2026-10-12)

Pre-registered before running: `new_metrics_prereg.json` (NFL) and `cfb_havoc_prereg.json` (college), both
committed before any number was computed. Code: `scripts/research/round4/new_metrics.py`, `cfb_havoc.py`.
Numbers: `new_metrics.json`, `cfb_havoc.json`.

**Already tested before this round (2026-10-02):**
- **Havoc:** `advanced_metrics.md`. Log loss improved about −0.0006 in both periods, short of the 0.001 bar.
- **Pressure rate:** nflverse participation data, 2016+. Tested together with sacks, QB hits, clean-vs-pressured EPA
  and FTN blitz rate (`matchups_refs.md`). Every set made validation worse, and market tests were at chance level.

## Results: everything fails its pre-registered bar

| Test | Dev | Holdout | Verdict |
|---|---|---|---|
| A. Team CPOE (offense − allowed), production model | log loss +0.00046 (2015-19) | +0.00013 (2020-25) | fail (worse) |
| A. Starting-QB CPOE (EPA+CPOE composite), production model | +0.00046 | +0.00003 | fail (worse) |
| A. Team CPOE vs closing-line residual | +0.11±0.28 pts/SD (2012-19) | +0.07±0.31 (2020-25) | fail |
| A. QB CPOE vs closing-line residual | +0.39±0.28 | +0.04±0.30 | fail |
| B. FTN INT-worthy throws | −0.50±0.80 (2023) | −0.01±0.51 (2024-25) | fail |
| B. FTN catchable-ball rate | −0.45±0.78 | −0.33±0.52 | fail |
| B. FTN drop rate (luck, fade) | −1.41±0.84 | +0.96±0.51 (sign flipped) | fail |
| B. FTN QB-fault sacks | +0.81±0.89 | +0.38±0.47 | fail |
| C. Fade teams off a bye (arXiv 2408.10867) | 49.8% covers, 215 games (2012-19) | 52.9%, 153 games, p 0.44 (2020-25) | fail |
| College havoc, model | MAE −0.023 pts (2021-22) | +0.009 (2023-25) | fail |
| College havoc vs closing-line residual | +0.06±0.36 pts/SD (2020-22) | +0.21±0.30 (2023-25) | fail |

What the numbers mean:
- **College havoc** is a real, persistent trait: the pre-game rating correlates 0.23 with that game's net havoc.
  Spreads already account for it.
- **Bye week:** the bye team's margin vs the close was −0.5±0.8 pts (dev) and −0.1±0.9 pts (holdout). That matches
  the paper's finding that the bye barely helps since 2011, but the market isn't over-pricing it enough to bet.

## Literature checked
- [nfelo: best QB metrics](https://www.nfeloapp.com/analysis/what-are-the-best-metrics-for-nfl-quarterbacks/) found
  EPA predicts game margins best. CPOE and PFF grades are more stable but less predictive. Our CPOE result agrees.
- [STRAIN (arXiv 2305.10262)](https://arxiv.org/pdf/2305.10262): a pass-rush metric from tracking data that predicts
  future pressure better than pressure rate does (r 0.32 vs 0.10). It needs player tracking, which isn't public.
- [Bye-Bye, Bye Advantage (arXiv 2408.10867)](https://arxiv.org/pdf/2408.10867) is tested above as C.
- [Pythagorean vs machine learning, Frontiers 2025](https://public-pages-files-2025.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1638446/text)
  is season-level win-percentage modelling only, with nothing game-level to test.
