# Recency-weighted training (2026-10-05)

Question: should the win-probability model weight recent seasons more, to catch newer trends in football and betting?
Script: `scripts/research/recency_weighting.py` (variants fixed before running). Walk-forward: every season is predicted
by a model fit only on earlier seasons. Log loss, lower is better.

| Training data | Tune 2015-19 | Holdout 2020-25 | 2024-25 only |
|---|---|---|---|
| All seasons since 2013, equal weight (production) | 0.6192 | **0.6209** | **0.6144** |
| Last 6 seasons only | 0.6192 | 0.6228 | 0.6202 |
| Last 4 seasons only | 0.6196 | 0.6241 | 0.6197 |
| Last 3 seasons only | 0.6203 | 0.6269 | 0.6248 |
| Season weights, half-life 1 season | 0.6195 | 0.6269 | 0.6276 |
| Season weights, half-life 2 seasons | 0.6189 | 0.6233 | 0.6209 |
| Season weights, half-life 4 seasons | 0.6189 | 0.6216 | 0.6171 |
| Season weights, half-life 8 seasons | 0.6190 | 0.6211 | 0.6155 |

Verdict: no. Every recency variant is worse on the holdout, including on the two most recent seasons, and the more
weight on recent years the worse it gets. Football outcomes are noisy, so dropping or down-weighting old seasons
throws away more signal than it gains. The model already reacts to the present through its inputs (team stats with
an 8-game half-life, Elo, the current starting QB). Recent-trend risk matters more for betting rules, because markets
adapt: those are checked season by season (e.g. the college moneyline track's CLV shrank in 2024-25) and live in the
paper tracks.
