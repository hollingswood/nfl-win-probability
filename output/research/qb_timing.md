# Does the market misprice quarterback changes?

Script: `scripts/research/qb_timing.py` (dev run; `--holdout` runs 2023-2025 once).
Outputs: `qb_timing_dev.json`, `qb_timing_frozen.json`, `qb_timing_holdout.json` (all in this folder).

**Verdict: the closing line prices QB changes correctly, and no rule built on public data at a known time beats the close.**
The early-week line does move about 2 points after a QB change it hasn't yet priced, and a bettor with perfect early
news would have beaten the close by about 5-8% (confirmed in 2023-25). But in our data that value only shows up when
we look at who actually started, which is hindsight. Every rule built from public signals at a known timestamp (the
starter left last week's game, the final report says Out, the final report says Questionable or Doubtful) landed at or
below zero CLV. Any edge depends on hearing the news before the books do, and this data can't test that.

## Definitions
- **QB change**: a team's listed starter (nflverse `*_qb_id`) differs from its previous game's starter in the same season. Week 1 is excluded.
- **new**: the starter did not start for this franchise in its previous 17 games. It is split three ways:
  - *injury_listed*: last week's starter is Out, Doubtful or Questionable on the final report.
  - *prev_left_early_unlisted*: last week's starter took less than 50% of the team's dropbacks in that game and isn't on the report.
  - *not_listed*: benching, IR, COVID list or resting. IR players are not on the report, so these reasons can't be separated.
- **return**: the returning QB was the established starter (most starts that season) when the replacement stint began. It is split into *after 1 missed game* and *after 2+*. When an earlier backup gets another turn, that is *other_qb_back_in*.
- **backup_continuing**: the same starter as last week, but not the season's established starter.
- **rookie debut**: a first career start (1999+) in the QB's rookie season.
- **gap**: the new starter's pre-game rating minus last week's starter's rating. Ratings are the model's EPA-per-dropback QB ratings (`features.qb_ratings`), using earlier games only. Multiply by about 38 to get points.
- **Close-ATS**: graded against the nflverse closing spread. Cover rates are from the named team's side. Games where both teams are in the same bucket are dropped. Pushes are excluded.
- **Early CLV**: the bet uses the best allowed-book price (from `my_books.json`) at the first snapshot at least 4h after both teams' previous games kicked off. The median is 148h before kickoff (Monday). CLV is valued at the price-implied close `closing_fair().mu_close_all`, with the sharp close shown alongside.

## 1. At the close (2012-2022): no systematic over- or under-reaction
Cover rate of the team that changed (± SE, n):

| bucket | 2012-19 | 2020-22 | pooled |
|---|---|---|---|
| new QB, all | 46.4% ±4.2 (140) | 58.3% ±5.8 (72) | 50.5% ±3.4 (212) |
| new, injury listed | 42.0% ±5.9 (69) | 62.5% ±7.7 (40) | 49.5% ±4.8 (109) |
| new, not listed (bench/IR/COVID/rest) | 50.9% ±6.7 (55) | 53.6% ±9.4 (28) | 51.8% ±5.5 (83) |
| rookie debut | 44.1% ±8.5 (34) | 58.8% ±11.9 (17) | 49.0% ±7.0 (51) |
| starter returns, after 1 game | 45.8% ±7.2 (48) | 42.3% ±9.7 (26) | 44.6% ±5.8 (74) |
| starter returns, after 2+ | 49.3% ±5.9 (71) | 42.9% ±9.4 (28) | 47.5% ±5.0 (99) |
| backup continuing | 45.5% ±2.8 (323) | 52.2% ±4.3 (134) | 47.5% ±2.3 (457) |
| new QB, big rating drop (gap < -0.15) | 47.1% (34) | 68.2% (22) | 55.4% ±6.6 (56) |

- **The effect changes sign between periods.** In 2012-19, backups under-covered. In 2020-22, they over-covered. None of these rows is significant against 50%.
- **The ATS residual doesn't track our QB gap.** Regressing it on the net QB gap (in points) gives a slope of +0.22 ±0.13 for 2012-19, -0.28 ±0.16 for 2020-22 and +0.04 ±0.10 pooled. The close already reflects the size of the downgrade.
- **One bin looked strong.** When the rating gap is lateral (|gap| ≤ 0.05), the changing team covered only 34%. That held in both periods (n=83 and n=30), so it was frozen as R1. It reversed out of sample (below).

## 2. Early week (2020-2022, 834 games)
Line move from the early snapshot to the price-implied close, in points toward the team (± SE):

| group | n | mean move | mean \|move\| |
|---|---|---|---|
| teams with no QB change | 1334 | +0.15 ±0.05 | 1.25 |
| new QB starts, all | 75 | **-2.15 ±0.29** | 2.56 |
| … last week's starter left that game | 13 | -0.13 ±0.37 | 1.14 |
| … last week's starter listed Out | 16 | -2.29 ±0.48 | 2.29 |
| … last week's starter listed Q/D | 16 | -3.28 ±0.55 | 3.38 |
| … not listed (benching/IR/COVID) | 30 | -2.33 ±0.53 | 2.89 |
| starter returns | 55 | +0.61 ±0.24 | 1.39 |

An in-game injury is priced by Monday morning. News that arrives midweek (benchings, IR moves, game-time decisions) moves the line afterward. Returning starters move the line much less.

Spread CLV at the best allowed book against the price-implied close:

| bet | n | mean CLV ± SE | beat close |
|---|---|---|---|
| null: every game, both sides, early | 1668 | -3.2% ±0.3 (the vig) | 35% |
| **hindsight**: against a team that will start a new QB, early | 75 | **+7.8% ±1.7** | 68% |
| hindsight: for a team whose better starter returns, early | 37 | +1.5% ±1.7 | 51% |
| early, against the team whose starter left last game | 32 | -4.9% ±1.9 | 28% |
| early, for the team whose starter left last game | 32 | -1.9% ±1.9 | 41% |
| after final report Out: against / for | 36 | -2.9% / -3.0% (±0.9) | 22% |
| after final report Q/D: against / for | 53 | -1.2% ±1.3 / -4.7% ±1.3 | 38% / 23% |

Once the Out designation is public, the line is fully priced: both sides lose only the vig. The early mispricing is real, but you can only capture it by knowing about the change before the books do.

## 3. Frozen rules and the 2023-2025 holdout (run once)
The corrected significance threshold is 0.05/3 = 0.0167.

| rule | dev | 2023-25 | pass |
|---|---|---|---|
| R1: at the close, bet against a team making a lateral QB change (\|gap\| ≤ 0.05) | 65.5% cover, n=113, p=0.003 | **38.5%** (15-24), ROI -26.6%, p=0.97 | no |
| R2 (hindsight diagnostic, not bettable): early, against a team that will start a new QB | +7.8% CLV, n=75 | **+5.5% ±1.5**, n=77, p=0.0003; sharp close +5.0%; every season positive | yes |
| R3: after the final report lists the starter Q/D, bet against his team | -1.2% CLV, n=53 | -1.2% ±1.2, n=47, p=0.85 | no |

## Implications
- Don't add a close-based QB adjustment. The market is already right on average.
- The only lever is news speed. `news.py` (the Sleeper/ESPN feeds) could act as a trigger: bet the moment a starter is ruled out, benched or put on IR midweek. It can't be backtested here because we have no timestamps for when news broke. The next step is to log the time each news item is first seen next to the odds snapshots, then measure CLV live.

## Caveats
- The injury table keeps only the last weekly row, so `date_modified` is an upper bound on when news was known.
- IR players are absent from the injury reports.
- The nflverse starter is wrong in a few games (4 found in 2022).
- Close-ATS results assume -110.
- Subgroup sizes are small, and the early snapshots fall on a fixed schedule rather than following the news.
