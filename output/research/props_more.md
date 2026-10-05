# Defensive and more QB/RB props, 2023-25 (pre-declared screen)

Script: `scripts/research/props_more.py` (committed 2026-10-04 before the odds were downloaded). Data: Odds API
historical props, US books, Friday-early and 75-min-close snapshots; 159k quotes. Results: nflverse weekly stats.
Two fixed rules per market: **S** = the receptions shop rule (EV >= 2% vs the median no-vig of >= 3 other books at the
same number, CLV test) and **U** = blind unders at the best allowed-book number (ROI test). Pass needs p < 0.05/8 and
consistency across seasons.

**Verdict: nothing passes. No new paper track.**

| Market | Books per player | Over rate (market says) | S bets / CLV | U bets | U ROI ± SE | Pass |
|---|---|---|---|---|---|---|
| Sacks | 1.3 | 39.7% (44.4%) | 0 (too few books) | 260 | +5.5% ± 5.5 | no |
| Solo tackles | 1.5 | 44.3% (50.1%) | 0 (too few books) | 1,358 | +5.0% ± 2.6 (p = 0.03) | no (needs p < 0.006) |
| Tackles + assists | 2.6 | 46.9% (49.8%) | 17 / +0.1% | 3,835 | +0.4% ± 1.5 | no |
| Defensive INTs | not in the feed | | | | | |
| Pass TDs | 5.8 | 49.3% (48.4%) | 23 / +1.0% | 1,202 | −7.0% ± 2.7 | no |
| INTs thrown | 5.1 | 48.8% (50.1%) | 20 / +1.1% | 1,521 | −2.7% ± 2.5 | no |
| Completions | 5.4 | 49.2% (49.9%) | 28 / −0.8% | 1,491 | −3.4% ± 2.4 | no |
| Rush attempts | 3.8 | 47.0% (49.7%) | 45 / +0.3% | 2,253 | +0.5% ± 2.0 | no |

What it means:
- **Shopping doesn't work on defensive props because almost nobody posts them.** Sacks and solo tackles have 1-2
  books per player, so there is no "other books' price" to beat. QB/RB markets are well covered, but shop
  opportunities are rare (20-45 in three seasons) and their CLV is about zero.
- **The QB props are efficient.** Overs hit at almost exactly the rate the prices imply; blind unders lose the vig.
- **Defensive unders lean the right way but aren't proven.** Overs hit 40-44% on sacks and solo tackles against
  44-50% implied. Solo-tackle unders made +5% (p = 0.03), but only 2024-25 have enough lines, and with 8 markets tested
  that isn't significant. It is a lead to re-test, not a bet.

Data note: the first run mapped tackles wrongly. nflverse splits official solo tackles into `def_tackles_solo` and
`def_tackles_with_assist`; checked against Zaire Franklin's official 2024 line (173 total, 93 solo = 75 + 18 + 80 in
nflverse). With the wrong mapping, tackle unders showed a fake +13-19% ROI. Results above use the corrected mapping
(the rules themselves were not changed).
