# Round 3: rule-change lag, option unders, conference/division spreads, early season, big favorites

Pre-declared 2026-10-07 in `scripts/research/round3/common.py` (the 12-rule list) and in each script's docstring, then each
script was run once. No thresholds were changed after the results. `x_exploratory.py` was written after the results and is
labelled exploratory.

**Method.** All bets are at the closing line. College lines come from CFBD (a real book's close where available) and have
no prices, so -110 is assumed. NFL totals use the nflverse closing over price. Pushes are left out of cover % and count as 0
in ROI. Break-even at -110 is 52.38%. p-values are one-sided exact binomial tests against 52.38% (can the rule beat the
vig?) and against 50% (is there any bias at all?).

**Pass/lead/fail.** There are 12 rules, so the Bonferroni cutoff is p < 0.05/12 = 0.00417 against 52.38%.
- **Pass:** p is below that cutoff, and both eras (2014-19 and 2020-25) cover above 52.38%.
- **Lead:** p against 52.38% is below 0.05, or p against 50% is below 0.00417, and both eras are above 50%.
- **Fail:** anything else.

**CLV.** A bet placed at the close has zero CLV by construction. For college 2021-25, each rule was also run at the CFBD
opening number, and the CLV shown is how many points the close then moved in the bet's favor. Per-season tables are in the
`t*.md` files next to this one.

## Summary

| rule | bets | W-L-P | cover | ROI | p vs 52.38% | verdict |
|---|---|---|---|---|---|---|
| R1 NFL over, wk 1-4, 2024+25 | 128 | 62-66-0 | 48.4% | -7.4% | 0.84 | fail |
| R2 CFB under, wk 1-4, 2023 (clock rule) | 407 | 199-204-4 | 49.4% | -5.7% | 0.90 | fail |
| R3 CFB over, wk 1-4, 2024 (2-min warning) | 423 | 196-224-3 | 46.7% | -10.8% | 0.99 | fail |
| R4 Academies under | 397 | 189-206-2 | 47.8% | -8.6% | 0.97 | fail |
| R5 All triple-option under | 616 | 303-311-2 | 49.3% | -5.8% | 0.94 | fail |
| R6 Power vs G5, back power | 1124 | 569-520-35 | 52.2% | -0.2% | 0.55 | fail |
| R7 Power team as favorite | 1020 | 518-472-30 | 52.3% | -0.1% | 0.53 | fail |
| R8 Power team as underdog | 103 | 51-47-5 | 52.0% | -0.6% | 0.57 | fail |
| R9 FBS vs FCS, back FBS | 1192 | 579-596-17 | 49.3% | -5.8% | 0.99 (two-sided vs 50%: 0.64) | fail |
| R10 CFB wk 0-2 under, total >= 55 | 905 | 485-414-6 | 53.9% | +3.0% | 0.18 (vs 50%: 0.010) | fail (near miss) |
| R11 CFB wk 0-2 over, total <= 45 | 104 | 50-54-0 | 48.1% | -8.2% | 0.84 | fail |
| R12 CFB dog, spread >= 21 | 2597 | 1295-1249-53 | 50.9% | -2.8% | 0.94 | fail |

**No rule passes, and none meets the declared lead bar.** Nothing new goes into paper-tracking from this round. Below the
tests is a short exploratory section with two slices worth pre-registering for 2027, if you want to.

## Test 1: Rule-change scoring lag (R1-R3)

Rows from the NFL table, regular season, closing totals:

| season | weeks | games | mean close | mean actual | actual - close | over % |
|---|---|---|---|---|---|---|
| 2018-23 pooled | 1-4 | 381 | - | - | - | 47.5% |
| 2024 | 1-4 | 64 | 43.6 | 43.1 | -0.5 | 46.9% |
| 2024 | 5-9 | 74 | 44.6 | 47.3 | +2.7 | 57.5% |
| 2024 | 10-18 | 134 | 44.6 | 46.3 | +1.8 | 55.3% |
| 2025 | 1-4 | 64 | 45.1 | 46.5 | +1.4 | 50.0% |
| 2025 | 5-9 | 71 | 45.5 | 46.9 | +1.5 | 56.3% |
| 2025 | 10-18 | 137 | 44.5 | 45.3 | +0.9 | 51.1% |
| 2026 (to date) | 1-4 | 63 | 44.9 | 45.9 | +0.9 | 49.2% |

- **R1 fails.** The overs went 62-66 (48.4%, ROI -7.4% at actual prices): 2024 was 30-34 and 2025 was 32-32. The 2026
  out-of-sample weeks 1-4 went 31-32.
- **Early 2024 and 2025 don't stand out.** From 1999 to 2023, the weeks 1-4 over rate averaged 48.9% with a season-to-season
  SD of 6.2 points. The 2024 (47%) and 2025 (50%) numbers sit inside that normal range.
- **2024 weeks 1-4 actually went under.** The kickoff rule did not produce early overs.
- **R2 fails** (2023 college unders, 199-204). The market moved first: weeks 1-4 closing totals fell from 55.1 in 2022 to
  52.7 in 2023, and games still landed 0.8 points over the lower numbers.
- **R3 fails** (2024 college overs, 196-224).
- **The general rule fails.** Pooled over the three rule changes it went 457-494 (48.1%, ROI -8.2%). All three went the
  wrong way. Don't pre-register "bet weeks 1-4 after a rule change". The market moves totals before the season, sometimes
  too far.

## Test 2: Service academies and triple-option teams, unders (R4, R5)

| slice | bets | W-L-P | cover | ROI |
|---|---|---|---|---|
| R4 academies, 2014-2025 | 397 | 189-206-2 | 47.8% | -8.6% |
| 2014-19 | 200 | 94-106 | 47.0% | -10.3% |
| 2020-25 | 197 | 95-100-2 | 48.7% | -6.9% |
| 2026 to date | 11 | 4-7 | 36.4% | -30.6% |
| vs each other | 36 | 28-7-1 | 80.0% | +51.3% |
| vs everyone else | 361 | 161-199-1 | 44.7% | -14.6% |
| by team: Army / Navy / Air Force | 143 / 148 / 142 | 77-64 / 71-76 / 69-73 | 54.6 / 48.3 / 48.6% | |
| R5 all option teams, 2014-2025 | 616 | 303-311-2 | 49.3% | -5.8% |
| 2014-19 / 2020-25 | 392 / 224 | 191-201 / 112-110 | 48.7 / 50.5% | |

- **Academies against other opponents go over more often than under** (161-199). Across all academy games, the mean close was 51.5 and the mean actual total was 53.6.
- **The market moves toward the under during the week.** Bet at the open in 2021-25, CLV averaged +0.8 points (the close
  was lower in 59% of games), but the bets only went 87-79 (+0.1% ROI).
- **R4 and R5 fail.** The only strong slice is academy vs academy. It was a reporting split, not a declared rule, and it is
  covered below.

## Test 3: Power vs Group of 5, and FBS vs FCS (R6-R9)

| slice | bets | W-L-P | cover | ROI |
|---|---|---|---|---|
| R6 power side, 2014-19 | 634 | 324-289-21 | 52.9% | +0.9% |
| R6 power side, 2020-25 | 490 | 245-231-14 | 51.5% | -1.7% |
| R6 power side, 2026 to date | 62 | 32-29-1 | 52.5% | +0.1% |
| R7 power favored | 1020 | 518-472-30 | 52.3% | -0.1% |
| R8 power underdog | 103 | 51-47-5 | 52.0% | -0.6% |
| R9 FBS side, 2014-19 | 579 | 265-303-11 | 46.7% | -10.7% |
| R9 FBS side, 2020-25 | 613 | 314-293-6 | 51.7% | -1.2% |
| R9 FBS side, 2026 to date | 119 | 73-46 | 61.3% | +17.1% |

- **Power teams beat Group of 5 lines by about a point on average** (+1.2 points over the close). That is not enough to
  beat -110: 52.2% overall, and p = 0.07 against 50%.
- **Spread buckets are just noise.** The 7.5-14 bucket covered 58.4%, but the 14.5-24 bucket covered 48.0%.
- **FBS vs FCS has no stable side.** The FCS team covered 53.3% in 2014-19, then the FBS team covered 51.7% in 2020-25, for
  50.7% FCS overall. The 2026 FBS run of 73-46 is a single partial season after 12 seasons with no edge. All four rules fail.

## Test 4: College early season (R10, R11)

Closing-line accuracy by week (FBS vs FBS only):

| era | weeks | spread MAE | total MAE | actual - close total | over % |
|---|---|---|---|---|---|
| 2014-19 | 1-2 | 12.43 | 12.92 | -1.16 | 47.3% |
| 2014-19 | 3+ | 12.2-12.6 | 13.1-13.4 | +0.3 to +0.6 | 48% |
| 2020-25 | 1-2 | 11.94 | 12.64 | -1.97 | 41.3% |
| 2020-25 | 3+ | 12.2-12.3 | 12.4-12.9 | +0.6 to +0.9 | 50-52% |

- **Week 0-2 lines are no less accurate than later lines.** Mean absolute error is about the same.
- **Week 0-2 totals do run high.** The actual total came in 1-2 points under the close, most clearly in 2020-25.
- **R10 is a near miss but still fails.** The rule was unders in weeks 0-2 when the total is 55 or more. Overall it was
  53.9%: 2014-19 went 50.7% and 2020-25 went 56.6%. The p-value against 50% was 0.010, short of the 0.0042 cutoff. The
  2026 out-of-sample games went 53-51.
- **R10 looks better at the open.** In 2021-25, with the threshold applied to the opening total, it went 210-152 (58.0%),
  but the close moved our way only 40% of the time (CLV +0.06 points). So the market isn't fixing this during the week. It
  is either a real early-season bias or noise.
- **R11 fails.** The opposite rule (overs when the total is 45 or less) went 50-54.

## Test 5: College big favorites, back the dog at 21+ (R12)

| slice | bets | W-L-P | cover | ROI |
|---|---|---|---|---|
| 2014-16 | 525 | 262-249 | 51.3% | -2.1% |
| 2017-19 | 625 | 324-286 | 53.1% | +1.4% |
| 2020-22 | 604 | 291-301 | 49.2% | -6.0% |
| 2023-25 | 843 | 418-413 | 50.3% | -3.9% |
| 2026 to date | 213 | 97-116 | 45.5% | -13.1% |
| FBS v FBS / FBS v FCS | 1336 / 904 | 667-636 / 460-431 | 51.2 / 51.6% | |
| spread 21-27.5 / 28-34.5 / 35+ | 1243 / 660 / 694 | | 51.0 / 51.5 / 50.2% | |

- **R12 fails.** The dog's edge was about 3 points of cover rate in 2017-19 and has been gone since 2020. This matches the
  earlier FBS-only test in `factor_screen.md` (F7).

## Exploratory (post-hoc, not evidence on its own; `x_exploratory.md`)

- **X1: academy vs academy unders went 28-7-1 (80%).** That was 16-2 in 2014-19 and 12-5-1 in 2020-25. At the Pinnacle
  close with real under prices, the 15 games from 2021 on went 11-4 (+43%). The 2026 Navy-Air Force game was another under.
  - The market has adjusted: these closes fell from about 50 (2014-16) to about 32-40 (2020-24), and 2024-25 went 3-3.
  - This is a widely publicized trend with only about 3 games a year, so it can never be validated statistically.
- **X2: the NFL kickoff-rule "lag" showed up after week 4, if anywhere.** In 2024, weeks 5-18 overs went 115-90 (56.1%); in
  2025 they went 110-98 (52.9%). For comparison, 2018-23 weeks 5-18 went 47.3%. The pooled p against 52.38% is 0.21.
- **X3: college weeks 0-2 unders, all totals, FBS vs FBS** went 569-454 (55.6%) for 2014-2025: 52.7% in 2014-19, 58.7% in
  2020-25. The 2026 out-of-sample games went 52-48 (-0.7%).

## What to paper-track in 2026

No pass, so nothing new is required. Two optional pre-registrations, to be judged only on 2027+ data:

1. **College early-season under, FBS vs FBS** (from R10 and X3):
   - **Rule:** in CFBD regular-season weeks 1-2 (which include week 0), bet the under at the best allowed-book price, from
     the closing snapshot, in FBS vs FBS games.
   - **Judging:** on results (cover rate against 52.38%, one-sided p < 0.05) after at least 400 bets, which is about 4
     seasons. CLV is informational only, because this showed no CLV.
   - **Note:** the 2026 out-of-sample result (52-48) already weakens it.
2. **Academy vs academy unders:**
   - Log only, about 3 games a year. Treat it as a curiosity, not a track.

Don't track the rule-change lag, option unders, the power/G5 or FBS/FCS sides, or big-favorite dogs.

Files:
- Scripts: `scripts/research/round3/` (`common.py`, `t1`-`t5`, `x_exploratory.py`).
- Results: `output/research/round3/*.json` and `*.md`.
