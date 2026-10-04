# College round 2: how pros actually beat college lines (2021-2025)

Every rule below was written down and committed (f33671d) before any result was seen. Grading uses Pinnacle's last pre-kickoff line (closing line value, CLV): the standard pros judge themselves by, because realized profit needs thousands of bets to separate skill from luck. 2026 is kept separate as a fresh test.

## 1. Shop vs sharp: PASSES

Price every quote at your books against Pinnacle's line at the same moment, at any number. College key numbers (3, 7, 10, 14, 17, 21, 24, 28) are built in from 2014-21 results; on 2022-25 the pricing was within 0.5% of actual cover rates at ±1 point. Bet only quotes that beat the sharp price.

| Rule | Bets | EV when bet | CLV | p | ROI | Seasons CLV+ | 2026 CLV (bets) | Verdict |
|---|---|---|---|---|---|---|---|---|
| your 3 books spread EV>=2% | 220 | +3.4% | +1.2% | 4e-04 | -3.5% ± 6.5% | 4/5 | +3.2% (30) | CLV yes, ROI noise |
| your 3 books total EV>=2% | 418 | +4.5% | +2.6% | 3e-08 | -4.2% ± 4.7% | 5/5 | +0.6% (24) | CLV yes, ROI noise |
| your 3 books moneyline EV>=2% | 197 | +3.0% | +2.6% | 1e-04 | -0.3% ± 8.7% | 5/5 | -5.2% (2) | CLV yes, ROI noise |
| your 3 books spread EV>=4% | 40 | +6.7% | +2.1% | 4e-02 | +0.7% ± 15.3% | 4/5 | +9.4% (5) | too few |
| your 3 books total EV>=4% | 114 | +9.6% | +7.6% | 2e-11 | +5.7% ± 8.9% | 5/5 | +1.3% (2) | **pass** |
| your 3 books moneyline EV>=4% | 30 | +5.2% | +4.4% | 1e-03 | +32.8% ± 23.6% | 4/5 | -15.0% (1) | CLV yes, ROI noise |
| all AZ books spread EV>=2% | 556 | +3.3% | +1.7% | 4e-10 | -3.7% ± 4.1% | 5/5 | +3.9% (66) | CLV yes, ROI noise |
| all AZ books total EV>=2% | 914 | +3.8% | +2.0% | 4e-13 | +0.2% ± 3.1% | 5/5 | +2.3% (54) | **pass** |
| all AZ books moneyline EV>=2% | 403 | +3.0% | +2.9% | 2e-11 | +2.6% ± 5.9% | 5/5 | +3.4% (13) | **pass** |
| all AZ books spread EV>=4% | 113 | +5.9% | +3.8% | 3e-10 | +0.1% ± 9.0% | 5/5 | +6.1% (17) | **pass** |
| all AZ books total EV>=4% | 218 | +7.6% | +5.6% | 5e-16 | +4.7% ± 6.4% | 5/5 | +3.7% (5) | **pass** |
| all AZ books moneyline EV>=4% | 69 | +5.0% | +4.3% | 8e-07 | +11.4% ± 14.9% | 5/5 | -1.2% (3) | CLV yes, ROI noise |

**All 12 rules beat the closing line**, and every quote used had been updated by the book within 3 hours of the snapshot. The edge is biggest 3-7 days before kickoff (soft books are slow to copy Pinnacle early in the week) and holds in Power-4 and smaller-conference games alike. Realized ROI is noisy at these sample sizes (±3-9%); CLV is the reliable signal. Frozen as the **cfb_shop** paper track (cfb_shop_rules.json): spreads at EV ≥ 4%, totals and moneylines at EV ≥ 2%, across Arizona books, quarter-Kelly stakes.

**Books matter.** With only your 3 books, the totals rule found 418 bets in five seasons; with all Arizona books, 914. BetRivers, BetMGM and Hard Rock showed the most consistent CLV. Caesars showed none.

## 2. Openers vs power ratings: FAILS

| Rule | Bets | CLV | ROI | Verdict |
|---|---|---|---|---|
| O1 market ratings vs your books' opener, gap ≥ 3 | 1944 | -3.5% | -3.0% ± 2.1% | fail |
| O2 market ratings, gap ≥ 5 | 1337 | -3.4% | -3.2% ± 2.6% | fail |
| O3 our model vs opener, gap ≥ 3 | 1339 | -0.9% | +0.1% ± 2.6% | fail |
| O4 our model, gap ≥ 5 | 698 | +0.4% | -2.5% ± 3.6% | fail |
| O5 fade moves of 2.5+ pts, bet late | 547 | -4.0% | -2.3% ± 4.0% | fail |

Soft-book openers are not stale copies of last week's ratings: lines move *away* from ratings built on earlier closing lines (CLV −3.5%, about the vig). Our stats model is break-even against openers. Fading big moves loses too. Beating college openers takes information the market doesn't have yet, not better math on public stats.

## 3. What is being first on QB news worth?

965 games where a team's starting QB changed. From the Sunday/Monday line to kickoff, Pinnacle moved only **+0.24 points** against that team on average. 7% of these games moved 3+ points (all games: average absolute move 1.17 pts; QB-change games: 1.48 pts). Most QB changes are known and priced by Sunday. The AI news reader now covers suspensions, eligibility, opt-outs, interim coaches, kickoff and venue changes and betting-market reports, and logs first-seen times. It stays logging-only until it shows it beats the line.

Not financial advice.
