# College shop-vs-sharp: where the edge is strongest and how to scale it (round 3)

Script: `scripts/research/round3/cfb_shop_segments.py`. Its docstring set out all 13 segment tests, the metric and the pass rules on 2026-10-07, before anything was computed, and the script was run once. Full tables are in `cfb_shop_segments_tables.md` and the numbers in `cfb_shop_segments.json`. One extra check was added after the results were in. It is marked "post-hoc" and lives in `cfb_shop_segments_sensitivity.py`.

**What is being sliced.** The rule is the frozen `cfb_shop` rule:
- Arizona soft-book quote vs Pinnacle's no-vig price at the same snapshot, priced at any number with the key-number distributions.
- Thresholds: spreads EV ≥ 4%, totals and moneylines EV ≥ 2%, price −200..+200, 1 h to 7 days before kickoff.
- One bet per game per market, at the first snapshot that qualifies.

From 2021 to 2025 that is **1,430 bets with an average CLV of +2.4% ± 0.2%**. In 2026 so far (out of sample) it is +3.2% ± 0.6% on 89 bets. The data reproduces the research bet counts exactly (130 / 972 / 417).

**Metric.** CLV is the expected profit per unit staked, priced against Pinnacle's last pre-kickoff line. Each test regresses CLV on market fixed effects plus the segment, with standard errors clustered by game. The Bonferroni-corrected bar across the 13 tests is **p < 0.0038**.

A segment counts as "DIFFERS" only if it clears that bar **and** the best level beats the rest in at least 4 of 5 seasons. Realized ROI is shown for completeness. At these sample sizes it carries ±3–9% of noise, so it is never used as a test.

## Bottom line

| # | Question | Result | Verdict |
|---|---|---|---|
| T1 | Hours before kickoff | >120 h **+3.2%**, 72–120 h +2.0%, 48–72 h +0.5%, 24–48 h +1.1%, 1–24 h +2.5% (p 4e-4, 4/5 seasons) | **DIFFERS**: early week best, midweek worst |
| T2 | Snapshot weekday (Saturday games) | Sun–Mon **+3.3%**, Tue–Wed +1.8%, Thu–Fri +1.2%, Sat +2.2% (p 0.004, 4/5) | suggestive (just misses the bar) |
| T3 | Conference tier | P4 vs P4 +3.1%, G5 vs G5 +2.0%, FCS involved +3.2%, **P4 vs G5 +0.7%** (p 0.004, 5/5) | suggestive |
| T4 | Conference vs non-conference game | +2.9% vs +1.6% (p 0.007) | suggestive |
| T5 | Season phase | wk 0–2 +1.5%, wk 3–8 +2.9%, wk 9+ +2.5%, bowls +0.5% (n 58) (p 0.025) | suggestive |
| T6 | Book | p 4e-4, but the best book wins in only 3/5 seasons. ESPN Bet +0.3%, Caesars +0.5%; others +1.7% to +2.8% | significant but inconsistent |
| T7 | Your 3 books vs the 6 others | +2.1% vs +1.8% (p 0.41) | no difference |
| T8 | Favorite vs underdog | +2.5% vs +3.2% (p 0.25) | no difference |
| T9 | Over vs under (totals) | **under +2.8%** vs over +1.1% (p 0.004 < 0.0038, 4/5) | **DIFFERS** |
| T10 | Home vs away | +2.4% vs +3.6% (p 0.07) | no difference |
| T11 | Does EV at bet predict CLV? | slope **1.02 ± 0.07**, with a fixed haircut: CLV ≈ EV − 1.9% (totals), EV − 1.3% (spreads), ≈ EV (moneyline) | **yes**: size on the haircut EV |
| T12 | First vs later qualifying snapshot | first +3.0% vs last +1.3% in the same game (p 8e-5). Only 25% of opportunities are still there at the next snapshot (~7 h later) | **DIFFERS**: bet at first sight |
| T13 | Trend 2021–25 | −0.50% CLV per season (p 0.0015): 3.3, 3.2, 2.1, 2.3, 1.4%. 2026 so far 3.2% | **shrinking**, but fragile (see S8) |

The edge is real in every slice. Every level of every segment that has more than about 60 bets has positive CLV, except a few small cells: P4-vs-G5 totals, bowl games, ESPN Bet and Caesars. So the useful answers are about **when to look, which books to open, and how to size bets**, not about which games to skip.

---

## S1 Timing: early week is richest, midweek is thinnest (T1 DIFFERS, T2 suggestive)

All qualifying opportunities, best quote per game, market and snapshot (2,097 opportunities on 1,295 games):

| Hours before kickoff | Opportunities | EV at bet | CLV | Totals CLV | Moneyline CLV | Spread CLV |
|---|---|---|---|---|---|---|
| >120 h (Sun/Mon for Sat games) | 549 | +4.3% | **+3.2%** | +2.8% | +4.0% | +4.1% |
| 72–120 h | 601 | +3.4% | +2.0% | +0.8% | +3.9% | +3.5% |
| 48–72 h | 224 | +2.9% | +0.5% | −0.9% | +2.5% | +3.8% |
| 24–48 h | 239 | +3.1% | +1.1% | −0.1% | +3.0% | +3.9% |
| 1–24 h | 484 | +3.5% | +2.5% | +2.3% | +2.2% | +4.1% |

By snapshot slot (UTC; Arizona time is UTC−7 all year):

| Slot | Arizona time | Bets that first qualified here | Their CLV |
|---|---|---|---|
| **Sun 23:10 UTC** | Sun 4:10 pm | **278** (19% of all bets) | **+5.3%** |
| Mon 16:10 UTC | Mon 9:10 am | 164 | +2.2% |
| Mon 23:10 UTC | Mon 4:10 pm | 138 | +1.3% |
| Tue 16:10 UTC | Tue 9:10 am | 114 | +3.1% |
| Wed–Fri (6 slots) | | 382 | −0.0% to +1.4% |
| Sat 13:10–23:10 UTC | Sat 6 am–4 pm | 257 | +2.1% to +4.0% |

**Pinnacle's college lines first appear Sunday afternoon.** For Saturday games, 51% are first seen in the Sun 23:10 UTC snapshot and 33% at Mon 16:10 UTC (median 137 h before kickoff). Barely any are up at Sun 16:10.

The single best moment is therefore **the first look after Pinnacle posts, on Sunday afternoon or evening**. That is when soft books' openers are furthest from the sharp number.

The research snapshots do not include Monday 9–11 am ET (13–15 UTC). The nearest slot, Monday noon ET, is middling (+2.2%). So Pro Spanky's window can't be confirmed or ruled out; what the data shows is that Sunday evening ET beats Monday midday.

Wednesday through Friday is the weakest stretch, and for totals it is essentially zero CLV. Saturday, in the hours before kickoff, is good again.

**Verdict:** Timing matters (T1 passes; T2 points the same way and just misses the bar).
- The first Pinnacle-post window is the richest.
- Midweek totals are the weakest cell. This is descriptive, not separately tested.

## S2 Conference tier: no reliable difference (T3, T4 suggestive)

| Tier | Bets | CLV | Totals CLV |
|---|---|---|---|
| P4 vs P4 | 622 | +3.1% | +3.0% |
| P4 vs G5/independent | 173 | +0.7% | −0.1% |
| G5 vs G5 | 545 | +2.0% | +1.5% |
| FCS or lower involved | 90 | +3.2% | +3.1% |

Conference games: +2.9%. Non-conference games: +1.6%.

The weak spot is P4-vs-G5 non-conference **totals**. That is consistent across seasons (P4 vs P4 is best in 5 of 5) but does not clear the corrected bar (p 0.004 against 0.0038). The FCS cell is small.

**Verdict:** Not proven. Keep betting all tiers. "Skip P4-vs-G5 totals" is at most a hypothesis for a new version.

## S3 Season phase: no reliable difference (T5 suggestive)

Weeks 0–2 come in at +1.5% (totals +0.6%), weeks 3–8 at +2.9%, weeks 9+ at +2.5%, and bowls/postseason at +0.5% (only 58 bets, ±1.2%).

The early-season and bowl dips match the idea that sharp prices are less informative when there is little current-season data or many opt-outs. The difference is not significant (p 0.025).

**Verdict:** No change.

## S4 Books: your 3 vs the rest is no different, but the extra books are where volume comes from (T6 inconsistent, T7 no difference)

Here each book is judged on its own first qualifying quote (BOOKQ):

| Book | Bets/season alone | CLV | Share of rule bets (2021–25) |
|---|---|---|---|
| FanDuel (yours) | 80 | +2.6% | 285 |
| BetRivers | 90 | +2.4% | 276 |
| BetMGM | 78 | +1.7% | 267 |
| Caesars (williamhill_us) | 63 | **+0.5%** | 198 |
| DraftKings (yours) | 53 | +2.2% | 170 |
| Hard Rock | 23 | +2.8% | 85 |
| Bally | 34 | +2.2% | 82 |
| ESPN Bet / theScore (yours) | 21 | **+0.3%** | 61 |
| Fanatics | 2 | +0.8% | 6 |

Your books and the no-account books produce the same quality of bet (+2.1% vs +1.8%, p 0.41). **More books add volume, not better bets.** Books do differ overall (p 4e-4), but which book is best changes from season to season. ESPN Bet and Caesars quotes that look +EV mostly fail to hold up against the close.

Which book to open next (flat $100, expected profit per season, 2021–25 average):

| Book set | Bets/season | Expected profit/season |
|---|---|---|
| Your 3 books (DK, FD, ESPN/theScore) | 131 | $334 |
| + **BetRivers** | 197 | $526 (+$192) |
| + **BetMGM** | 242 | $625 (+$99) |
| + Hard Rock | 255 | $670 (+$45) |
| + Caesars, Bally, Fanatics | 286 | $690 (+$20 total) |

**Verdict:** Open **BetRivers first, then BetMGM**. Together they add about 110 bets and about $290 of expected profit a season. Hard Rock is a distant third, and the last three are not worth the effort. (Check that each operates in Arizona; this uses the repo's Arizona book list.)

## S5 Market × side: unders beat overs; favorite/dog and home/away don't matter (T9 DIFFERS; T8, T10 no difference)

| Segment | Bets | EV at bet | CLV | Share of EV kept at close |
|---|---|---|---|---|
| Totals, under | 489 | +3.4% | **+2.8%** | 0.83 |
| Totals, over | 425 | +4.3% | +1.1% | **0.26** |
| Spread/ML, underdog | 396 | +3.6% | +3.2% | 0.91 |
| Spread/ML, favorite | 120 | +3.8% | +2.5% | 0.66 |
| Spread/ML, away (non-neutral) | 253 | +3.8% | +3.6% | 0.96 |
| Spread/ML, home (non-neutral) | 214 | +3.5% | +2.4% | 0.68 |

Over value that looks real at a soft book keeps only a quarter of its EV by the close. Under value keeps most of it. College totals tend to drift down toward kickoff, which helps unders and hurts overs.

**Verdict:** Unders are the stronger half of the totals rule, and overs at a 2% threshold are marginal (+1.1%, p 0.008). A **separate over threshold (for example EV ≥ 4%) is a candidate for a new paper version**, not a change to v2.

## S6 Sizing: EV predicts CLV one-for-one, after a fixed haircut (T11 passes)

Bets at EV ≥ 2% in all three markets, 2021–25 (1,873 bets):

| EV at bet | Bets | CLV | Share of EV kept |
|---|---|---|---|
| 2–3% | 1,081 | +1.1% | 0.47 |
| 3–4% | 446 | +2.0% | 0.58 |
| 4–6% | 235 | +3.2% | 0.69 |
| 6–10% | 69 | +5.0% | 0.70 |
| ≥10% | 42 | +17.7% | 0.94 |

Fitted per market (in-sample):
- totals: CLV ≈ −1.9% + 1.04·EV
- spreads: CLV ≈ −1.3% + 0.91·EV
- moneylines: CLV ≈ −0.4% + 1.10·EV

Post-hoc check without the EV ≥ 10% bets: slopes 0.85–1.21, intercepts −0.7% to −1.5%. Same picture.

Moneyline EV is close to unbiased. Spread and total EV is overstated by about 1.3–1.9 points, so a total bet at EV 2.5% is really worth about +0.6%. Quarter-Kelly on the raw EV over-stakes those near-threshold totals.

Sizing on the haircut EV ("calibrated Kelly" below) cuts the amount wagered by about 40% for about the same expected profit, and with a lower chance of a losing season.

The ≥10% bets are 79% FanDuel, mostly 2022 totals 3–13 points off Pinnacle. They held their CLV (+17.7%) and won 62%, but quotes that far off are the kind books void as obvious errors or limit. Keep the 2% bankroll cap and eyeball them before betting.

**Verdict:** Size on calibrated EV. As a sizing change this needs a new version of the rule's `sizing` block; grading is in flat units, so validation is unaffected.

## S7 Persistence: bet the first time a game qualifies (T12 differs)

- 25% of game-markets qualify at two or more snapshots.
- An opportunity has a 25% chance of still being there at the next research snapshot (median gap 7 h).
- When a game qualifies more than once, the first snapshot's CLV is +3.0% and the last one's is +1.3% (paired difference +1.6% ± 0.4%).
- Second and later opportunities are still worth having on their own (+1.7%, p 0.002).

**Verdict:** The current "first qualifying snapshot" rule is right. Waiting costs about 1.6 points of CLV. Opportunities are short-lived, so **checking more often should find more of them**. These data (about 2 snapshots a day) can't say how many more. The live hourly Thu–Sat cadence will show it.

## S8 Decay: some shrinkage, less than it first looks (T13 shrinking, fragile)

CLV by season: 2021 +3.3%, 2022 +3.2%, 2023 +2.1%, 2024 +2.3%, 2025 +1.4%. 2026 so far: +3.2% on 89 bets.

- By market: moneylines are steady (+1.7% to +5.2%) and spreads too (+2.6% to +5.0%). Totals carry the decline, falling to +0.8% in 2025.
- The pre-declared trend is −0.50% per season (p 0.0015).
- Post-hoc, without the 42 bets at EV ≥ 10% (the 2022 FanDuel totals), it is **−0.29% per season, p 0.055**. That is not significant.

**Verdict:** Plan on **about 1.5–2% CLV a season, not 2.4%**. Watch totals in particular: 2025 was its worst year, but 2026 has started at +2.3%.

## Scale: what it is worth (estimate)

Expected profit = sum of stake × CLV, taking each bet's CLV as its expected return. 2021–25 averages, research cadence of about 2 snapshots a day. This is **an estimate**, not a promise.

| Book set | Sizing | Bets/season | Avg stake | Amount wagered/season | **Expected profit/season** | Season SD | P(losing season) |
|---|---|---|---|---|---|---|---|
| Your 3 books | flat $100 | 131 | $100 | $13.1k | **$334** | $1,169 | 39% |
| Your 3 books | ¼-Kelly, $10k | 131 | $90 | $11.7k | **$454** | $1,105 | 34% |
| Your 3 books | calibrated ¼-Kelly | 131 | $57 | $7.5k | **$409** | $849 | 32% |
| All 9 AZ books | flat $100 | 286 | $100 | $28.6k | **$690** | $1,714 | 34% |
| All 9 AZ books | ¼-Kelly, $10k | 286 | $88 | $25.1k | **$819** | $1,575 | 30% |
| All 9 AZ books | calibrated ¼-Kelly | 286 | $53 | $15.1k | **$668** | $1,132 | 28% |

- Your 3 books by market (flat $100): totals 84 bets, $215; moneylines 39 bets, $102; spreads 8 bets, $17.
- If CLV stays at 2025's level (about 60% of the five-year average), scale these down: roughly **$200 a season with your 3 books and $400 with all Arizona books** at flat $100.
- Realized profit in 2021–25 was −$360 a season at flat $100 with your 3 books and +$257 with all AZ books. Both are within one SD of expectation, which shows how noisy a single season is.
- The edge is real, but at $100 stakes it is worth hundreds of dollars a year, not thousands. Scaling comes from more books, more frequent early-week checks and bigger stakes, until books start limiting the account.

## Recommendations

**Operational (no rule change needed):**
1. **Open BetRivers, then BetMGM.** Together about +110 bets and +$290 of expected profit a season at flat $100, roughly doubling what your 3 books produce. Hard Rock is optional. Caesars, Bally and Fanatics add almost nothing.
2. **Check hourly Sunday 16:00–24:00 UTC (9 am–5 pm Arizona) and Monday 13:00–17:00 UTC (6–10 am Arizona).** Pinnacle posts in that window and it is the richest of the week; the Sun 23:10 UTC slot alone gave 19% of all bets at +5.3% CLV. The live job currently runs only every 3 h on Sun–Wed. A denser Monday-morning check also tests Pro Spanky's 9–11 am ET claim, which the research snapshots couldn't.
3. **Bet when a game first qualifies.** Opportunities usually disappear within hours, and waiting cost about 1.6 points of CLV.
4. **Hand-check any quote at EV ≥ 10%** (usually a soft-book total several points off) before betting it, because of the void/limit risk.
5. **Expect about 1.5–2% CLV**, and judge the track on CLV, not on one season's profit (a losing season has roughly a 30–40% chance even with the edge).

**Hypotheses for a new paper-track version (`cfb_shop` v3). Not adopted, because they come from this in-sample slicing:**
- **Overs need EV ≥ 4%; unders stay at 2%.** This follows T9, the only side test that passed.
- **Calibrated sizing:** quarter-Kelly on EV minus 1.9% (totals) or 1.3% (spreads) instead of raw EV. Moneylines unchanged.
- Weaker leads, which did not pass and should only be logged: skip totals 24–72 h before kickoff (midweek), skip P4-vs-G5 non-conference totals, and exclude ESPN Bet and Caesars quotes (inconsistent across seasons).
- **Do not lower thresholds.** Bets at EV 2–3% keep only about half their EV, and spreads at 2–4% average +0.9% to +1.9% CLV, which is a thin margin for the limits they would use up.

A v3 needs its own pre-declared pass rule on 2026+ data (for example, at least 150 bets and CLV above v2's on the same games). Until then v2 stays the live rule.

**Caveats.**
- CLV here comes from the key-number pricing model.
- Moneylines in this research remove the vig with the multiplicative method; live v2 uses Shin, which makes a negligible difference inside −200..+200.
- Research snapshots run about twice a day, and the live job runs more often, so live volume should come in higher.
- Not financial advice.
