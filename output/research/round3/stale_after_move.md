# Stale soft book after a Pinnacle move, and the Kambi cluster

**Verdict: FAIL as a new angle.** None of the 8 pre-declared rules passes. The NFL version fails outright. Its only lead (R1b, 76 bets) disappears if you fill one hour later. The college leads come from the soft-book gap that the existing college shop track (`cfb_shop_rules.json`) already bets. Bets placed right after a Pinnacle move are not measurably better than bets placed when Pinnacle has not moved, and the price is gone by the next college snapshot. No live rule and no new paper track are recommended.

**Kambi cluster: true only for part of the period.** In 2024, BetRivers, Bally Bet and betPARX posted the same moneyline on 100% of co-quoted snapshots, and they lagged Pinnacle together. In 2025 BetRivers split off: its moneyline matched Bally's only 3% of the time in the NFL and 24% in college. **Bally Bet and betPARX are still one feed** (100% identical in 2025 and 2026). Over 2024-25 combined, BetRivers and Bally match on 51% of NFL snapshots, below the pre-declared 80% bar, so "BetRivers + Bally lag as one" fails as stated.

Script (rules in the docstring, written before any confirm-year data was loaded): `scripts/research/round3/stale_after_move.py`. Raw output: `stale_after_move_{nfl,cfb}.json`, bets in `stale_after_move_{nfl,cfb}_bets.parquet`.

## Design (pre-declared)

- **Pinnacle move.** Between consecutive Pinnacle snapshots of a game, either of these:
  - the Shin no-vig home win probability moves ≥ 3 pts, or
  - the spread-implied mean margin moves ≥ 1.0 pt (a point move or a price move both count).
  The move side is the side that got stronger. The snapshot after the move (t1) must be 1 h to 7 days before kickoff.
- **Bet.** At t1, any Arizona book quote on the move side (the stale side) with EV ≥ 2% vs Pinnacle's new fair price, priced between -200 and +200.
  - Books: DraftKings, FanDuel, ESPN Bet, Barstool, BetMGM, Caesars, BetRivers, Fanatics, Hard Rock, Bally.
  - Pricing is the same as the earlier shop screens: Shin for moneylines; the key-number margin model for spreads (`nfl_dist.json` for NFL, `cfb_dist.json` for college).
  - One bet per game: the first qualifying move, at the best-EV book.
- **Grading.**
  - CLV against Pinnacle's last pre-kickoff snapshot. Bets where that snapshot is not after t1 are dropped.
  - ROI against actual results.
  - Latency check: the same book and side, filled at its next snapshot (NFL: one hour later; college: the next scheduled snapshot, 2-17 h later).
- **Data split.** NFL hourly data exists only for 2022-25, so explore = 2022 and confirm = 2023-25. College: explore = 2021-22, confirm = 2023-25, and 2026 is reported separately.
- **Pass criteria (confirm years only).** All of:
  - ≥ 100 bets;
  - CLV > 0 at one-sided p < 0.05/8 (Bonferroni over 8 rules);
  - ROI ≥ 0;
  - CLV positive in ≥ 2 of 3 seasons;
  - NFL only: CLV > 0 when filled one hour later.
  A rule that fails these but has confirm CLV > 0 at unadjusted p < 0.05 is a **lead**.
- **Amendment, made before any confirm data was loaded.** A debugging run on the explore seasons gave only 4 / 7 (NFL) and 22 / 40 (college) bets under the suggested thresholds. That is too few to ever reach 100. So looser "b" variants were frozen: move ≥ 2 pts / ≥ 0.5 pt, EV ≥ 1%. The Bonferroni divisor went from 4 to 8. The explore CLVs of those handfuls were visible at that point; they were not used.

## Results (confirm 2023-25)

| Rule | Move / EV bar | Bets | EV at bet | CLV ± SE | p (1-sided) | ROI ± SE | Seasons CLV+ | Fill 1 snapshot later: CLV | No-move baseline CLV (bets) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 NFL moneyline | 3 pts / 2% | 6 | +3.3% | +2.8% ± 2.4% | 0.118 | -11% ± 40% | 2/3 | +0.2% ± 2.1% | +0.7% (258) | FAIL |
| R2 NFL spread | 1.0 pt / 2% | 17 | +3.2% | -0.5% ± 1.8% | 0.608 | -32% ± 23% | 1/3 | -2.8% ± 2.2% | +0.4% (298) | FAIL |
| R1b NFL moneyline loose | 2 pts / 1% | 76 | +2.1% | +1.9% ± 1.1% | 0.036 | +14% ± 11% | 3/3 | **-0.1% ± 0.9%** | +0.2% (423) | LEAD (weak) |
| R2b NFL spread loose | 0.5 pt / 1% | 212 | +1.9% | -0.3% ± 0.5% | 0.725 | -7% ± 7% | 1/3 | -1.6% ± 0.5% | +0.1% (544) | FAIL |
| R3 college moneyline | 3 pts / 2% | 26 | +2.9% | +1.4% ± 1.6% | 0.198 | -3% ± 21% | 3/3 | -1.3% ± 1.4% | +1.6% (225) | FAIL |
| R4 college spread | 1.0 pt / 2% | 85 | +3.1% | +1.7% ± 0.7% | 0.0065 | +22% ± 10% | 3/3 | -2.1% ± 0.7% | +1.3% (284) | LEAD |
| R3b college moneyline loose | 2 pts / 1% | 172 | +1.9% | +2.2% ± 0.6% | 0.0001 | -6% ± 8% | 3/3 | -1.2% ± 0.5% | +1.1% (494) | LEAD |
| R4b college spread loose | 0.5 pt / 1% | 410 | +1.9% | +0.5% ± 0.3% | 0.044 | -3% ± 5% | 3/3 | -2.8% ± 0.3% | +0.4% (627) | LEAD (weak) |

The **no-move baseline** is informational, not a pass criterion. It uses the same books, EV bar and pricing, but bets snapshots *without* a qualifying Pinnacle move. If the move-conditioned CLV is not clearly above the baseline, the move is not what drives the edge.

Each cell below is bets / CLV / ROI by season.

| Rule | explore | 2023 | 2024 | 2025 | 2026 (partial) |
|---|---|---|---|---|---|
| R1 NFL moneyline | 4 / +5.4% / +116% | 4 / +4.4% / +34% | 1 / +2.2% / -100% | 1 / -3.0% / -100% | n/a |
| R2 NFL spread | 7 / -0.3% / +35% | 6 / +0.9% / -36% | 5 / -1.6% / +15% | 6 / -1.0% / -69% | n/a |
| R1b NFL moneyline loose | 28 / +3.4% / +3% | 35 / +0.6% / +19% | 25 / +2.1% / -10% | 16 / +4.5% / +39% | n/a |
| R2b NFL spread loose | 72 / +1.0% / +22% | 74 / -0.7% / -33% | 74 / +0.2% / -4% | 64 / -0.5% / +18% | n/a |
| R3 college moneyline | 22 / +1.6% / -29% | 8 / +0.7% / +28% | 12 / +0.5% / -32% | 6 / +4.1% / +12% | — |
| R4 college spread | 40 / +0.4% / +5% | 27 / +0.8% / +21% | 24 / +1.1% / +29% | 34 / +2.9% / +18% | 12 / +5.3% / -20% |
| R3b college moneyline loose | 103 / +1.0% / -10% | 59 / +3.8% / -1% | 67 / +1.8% / -16% | 46 / +0.6% / -0% | 6 / +2.7% / -33% |
| R4b college spread loose | 211 / +0.6% / +1% | 136 / +0.2% / -7% | 126 / +0.9% / +1% | 148 / +0.6% / -3% | 46 / +3.5% / -4% |

### Reading the leads

- **R1b NFL moneyline (weak lead).**
  - The signature of a real stale price is there. Bets where the book had *not touched its price* since before the move: 62 bets, CLV +2.9% ± 1.2%. Bets where the book had already moved partway: 14 bets, -2.6%. Move-conditioned bets beat the no-move baseline by +1.6 ± 1.1 pts (about 1.5 SE, not significant).
  - But the edge lasts less than an hour. The same book filled one snapshot later shows CLV -0.1%.
  - It also fails on size (76 bets) and on the Bonferroni bar (p 0.036 vs 0.00625).
  - Bets per season are 16-35, so another season would not settle it.
  - Pinnacle kept a median 72% of these moves at close, and 24% fully reverted.
- **R4 college spread (lead).**
  - CLV +1.7%, ROI +22%, positive in all three seasons, 2026 so far +5.3% on 12 bets.
  - It misses on bet count (85 < 100) and just misses the Bonferroni bar (p 0.0065 vs 0.00625).
  - It is not better than college spread bets made with no move: +1.7% vs +1.3%, a difference of 0.4 ± 0.8.
- **R3b college moneyline (lead).**
  - The strongest CLV: +2.2% ± 0.6%, p 0.0001, 172 bets, 3/3 seasons. It passes every criterion except ROI (-6% ± 8%, within noise).
  - It is the same college soft-book gap as the static shop rule: the no-move baseline is +1.1%, and the difference is +1.0 ± 0.7 (about 1.5 SE).
  - It is gone by the next college snapshot (-1.2%).
  - It adds nothing the live `cfb_shop` track does not already bet.

## Lag profile: how fast books follow Pinnacle (confirm years)

"Unmatched" means the book's own no-vig price has not moved at least halfway with Pinnacle's move. Moves where the book was already there before Pinnacle moved (book led, 10-30%) are excluded.

**NFL moneyline moves ≥ 3 pts (130 moves).** Share still unmatched at each time after the move:

| Book | at t1 (0-1 h) | +1 h | +2 h |
|---|---|---|---|
| Bally | 35% | 31% | 29% |
| BetRivers | 34% | 30% | 27% |
| betPARX | 31% | 28% | 25% |
| BetMGM | 28% | 21% | 22% |
| FanDuel | 22% | 22% | 24% |
| Caesars | 21% | 22% | 18% |
| Hard Rock | 21% | 14% | 11% |
| DraftKings | 18% | 17% | 17% |
| Fanatics | 7% | 7% | 7% |
| ESPN Bet | 7% | 7% | 7% |

**NFL spread moves ≥ 1 pt (232 moves).** Share still unmatched:

| Book | at t1 (0-1 h) | +1 h | +2 h |
|---|---|---|---|
| BetRivers | 39% | 36% | 34% |
| Bally | 39% | 34% | 34% |
| betPARX | 38% | 34% | 33% |
| Hard Rock | 32% | 26% | 24% |
| BetMGM | 31% | 24% | 25% |
| Caesars | 29% | 27% | 29% |
| FanDuel | 28% | 27% | 29% |
| DraftKings | 23% | 21% | 21% |
| ESPN Bet | 17% | 17% | 17% |
| Fanatics | 10% | 10% | 12% |

**College (558 moneyline moves / 1,600 spread moves).** Unmatched at t1:

| Book | Moneyline | Spread |
|---|---|---|
| Bally | 29% | 33% |
| betPARX | 28% | 35% |
| BetRivers | 27% | 34% |
| BetMGM | 26% | 32% |
| DraftKings | 10% | 19% |
| ESPN Bet | 8% | 21% |
| Fanatics | 5% | 12% |

The Kambi-family books (BetRivers, Bally, betPARX) are the slowest Arizona books to follow Pinnacle, and ESPN Bet and Fanatics the fastest. But about 25-35% of moves stay "unmatched" for 2+ hours at most books. Many of those Pinnacle moves partly revert, so an unmatched book is often simply right. This is why the lag does not turn into significant NFL CLV.

## Kambi cluster: share of co-quoted snapshots with identical quotes

| Pair | NFL 2023 | NFL 2024 | NFL 2025 | College 2024 | College 2025 | College 2026 |
|---|---|---|---|---|---|---|
| Bally / betPARX: moneyline | — | 100% | 100% | 100% | 100% | 99.9% |
| Bally / BetRivers: moneyline | — | 100% | 3% | 100% | 24% | 39% |
| BetRivers / betPARX: moneyline | 85% | 100% | 3% | 100% | 24% | 39% |
| Bally / BetRivers: spread | — | 98% | 3% | 95% | 5% | 8% |
| Typical pair of other books: moneyline | 2-3% | 2% | 2% | 1% | 1% | 1% |

- **Co-lag, NFL confirm years, at t1.** P(Bally unmatched | BetRivers unmatched) = 0.95 on moneylines and 1.00 on spreads (phi 0.96 / 0.95). For Bally and betPARX, phi = 1.0. For comparison, DraftKings / FanDuel phi is 0.29 / 0.42. Most of these pooled moves are from 2024; co-lag was not split by season.
- **Earlier Kambi-feed families.** Barstool, TwinSpires, Unibet and SugarHouse (a BetRivers brand) were 99-100% identical with each other in 2021-23. In the NFL, BetRivers matched them on spreads (99%) but often not on moneylines (31-85%), so the operator sets its own NFL moneyline margin on top of a shared spread feed. In college it matched both (99%+).
- **Practical takeaway.** For line shopping, Bally and betPARX count as one quote, and so did BetRivers through 2024. When one of them is stale, the others almost always are too, so the cluster is one bet, not several.

## Caveats

- NFL hourly data exists only for 2022-25. Explore was a single season.
- College snapshots are not hourly. The "move" can be up to 17 h old at t1, and "1 snapshot later" means 2-17 h.
- Quotes are Odds API polls about 5 minutes before each snapshot. That they could actually be filled, and at what limits, is not tested. Kambi and smaller books limit sharp accounts quickly.
- ROI standard errors are large (8-40%), so CLV is the main evidence.
- The 8 rules include 4 looser variants added after seeing explore bet counts only (see the amendment above).

Not financial advice.
