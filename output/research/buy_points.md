# Buying points at real book prices (alternate spreads, 2023-2025)

Code: `scripts/research/buy_points.py` (stages curve -> dev -> freeze -> holdout -> dash). Numbers: `buy_points.json`.
Frozen rules: `buy_points_frozen.json`. Data: per-event alternate-spread ladders at two snapshots per game (early = Fri
21:40 UTC or kickoff-24h; close = ~75 min pre-kick), each book's main spread + juice from the main-odds file at the
nearest snapshot (<= 90 min; 520/552, 529/570, 530/570 snapshots matched in 2023/24/25). Fair margin = price-implied
mu from the main market (lowvig/betonlineag median, fallback all books) under the corrected key-number distribution
refit on 2012-2019 (`teasers_v2.py`, sigma 13.4). Allowed books present in the alt data: DraftKings, FanDuel, BetMGM,
Caesars (williamhill_us), BetRivers, Fanatics (2025 only). ESPN BET / Hard Rock: no alt data.

## Verdict

* **Buying points is not +EV anywhere we can bet.** Every book charges more than the half point is worth except
  BetRivers (and, marginally, Fanatics) onto 3 and 7 -- and even there it only makes a -4.5% main-line bet a ~-3% bet.
  No systematic buy (onto/off 3 or 7, fav or dog, early or close) had positive fair EV in any season.
* **The dashboard's 10/20/15-cent assumption is too cheap at 4 of 6 books** (BetMGM, Caesars, DraftKings, FanDuel)
  and too expensive at BetRivers/Fanatics for non-key numbers. Measured first-half-point costs below.
* The only "+EV" alt lines the model finds are mostly **sold** points at plus money 2-3.5 pts off the main line. Out of
  sample they did not pay (ROI -6% ± 10%, +0% ± 11%), and the model over-states win rates on exactly those lines
  by ~1.5 pts, i.e. about the size of the "edge".

## 1. Price of one half point off the main line (cents; all books' main lines sit at -110 median)

Measured = median cost of the first half point bought from that book's main line (main price -> alt price at main+0.5),
2023-2025. Break-even = the cost that leaves fair EV unchanged (value of P(margin lands on the number)).

| book | 3 (assumed 20) | 7 (assumed 15) | other (assumed 10) | notes |
|---|---|---|---|---|
| BetMGM | **25** (onto 25 / off 25) | **20** | **15** | 2023 was 20/15/10; got dearer in 2024-25 |
| Caesars | **25** (2024-25: 34-38) | **21** (2025: 26) | **15** (2024-25: 19-28) | 2024+ alt ladder starts ~11c worse than main |
| DraftKings | **22** | **18** | **10-11** (2025: 15) | ladder excludes the main point |
| FanDuel | **23** (off 3 only) | **18** | **12.5** | half-point alts only (no whole numbers) |
| BetRivers | **17** | **14** | **6** | cheapest; same in all three seasons |
| Fanatics (2025) | **20** | **15** | **5** | |
| break-even (fair) | onto 22-23 / off 19 | onto 15.5-16 / off 13.5 | 6.5-7 | P(land on 3) 7.9%, 7 5.8%, other ~2.7% |

Change in fair EV from buying the first half point (2023-25): onto 3 BetRivers +1.8 pts (92% of cases positive),
Fanatics +1.0, DK -0.3, MGM -0.8, Caesars -1.2; off 3 BetRivers +0.7, Fanatics 0.0, others -1.0 to -2.0; 7: BetRivers
+0.8 onto / 0.0 off, Fanatics +0.7 / -0.3, others -0.7 to -2.6; other numbers: BetRivers/Fanatics ~0, others -1.1 to
-3.2 pts.
Inside the alt ladder (alt -> alt steps) books charge ~20c for 3, 15-17c for 7, 8c elsewhere -- 3 and 7 slightly
under-priced relative to the outer ladder, but the alt ladder carries more hold (alt lines average -5.1% to -8.8% fair
EV by book vs -4.4% for main lines), so the cheap step is paid for in the level.

## 2. EV of alternate lines (fair_sharp)

| | 2023 | 2024-25 |
|---|---|---|
| main lines, mean EV | -4.4% to -4.6% by book | -4.3% to -5.1% |
| alt lines within ±3.5, mean EV by book | MGM -5.7, BR -5.1, DK -5.9, FD -5.5, CZR -7.3 | MGM -6.0, BR -5.1, DK -5.8, FAN -4.6, FD -5.5, CZR -8.8 |
| share of alt lines with EV > 0 | 0-2.5% | 0-3% |
| best-book alt line, bought 0.5-3.5 pts | -3.8% to -4.3% | -4.0% to -4.6% |

Exact-margin sanity check (close fair_sharp): |margin| = 3 predicted 38.5/39.8/39.2 vs actual 41/39/46; 7: 25.5/26.4/26.1
vs 23/20/26 (2023/24/25) -- the half-point values above are not under-stated.

## 3. Frozen rules (developed on 2023 only) and the one-time 2024-2025 test

Fixed transitions on 2023 (+2.5->+3, +3->+3.5, -3.5->-3, -3->-2.5, -7.5->-7, +6.5->+7, ±7 off, and their sells; same
book, best allowed book, early or close): model EV -2.8% to -5.5% every one, so none was frozen as an edge.

| rule | 2023 dev | 2024-25 bets | hit (no push) | avg price | model EV | ROI ± SE | CLV ± SE (vs close) | pre-registered test |
|---|---|---|---|---|---|---|---|---|
| B1 any alt line, EV ≥ 1%, close | 56 bets, ROI +10% ± 16 | 138 | 0.382 | +148 | +2.7% | **-6.0% ± 10.2** | +2.7% (= EV, same moment) | fail |
| B2 any alt line, EV ≥ 1%, early | 44 bets, CLV +2.2% ± 0.9 | 114 | 0.416 | +145 | +2.7% | **+0.1% ± 11.2** | +3.2% ± 0.6 | passes the letter (CLV t 5, ROI > 0) -- see below |
| B3 buy onto 3/7, close, no filter | 108 bets, ROI -6% ± 8 | 227 (23 pushes) | 0.525 | -124 | -4.1% | **-5.0% ± 5.7** | -4.0% ± 0.1 | fail (as expected) |

B2 is not evidence of an edge: its CLV is computed with the same distribution that selected the bets and the main
market barely moved, so CLV ≈ the model's own EV; realized wins 47 vs 47.7 expected; and a post-holdout calibration
check (close, all lines 2023-25) shows lines **sold** 2-4 pts off the main line won 1.4-1.8 pts less often than the
model predicts and **bought** lines 1.1-1.9 pts more often (main lines calibrated within 0.2 pts). At +145 that bias
is worth ~3.5-4.5% of EV -- larger than B1/B2's claimed +2.7%. (Same bias means bought lines are ~2-3 pts of EV
better than the tables above say; still negative.)

## 4. Dashboard spread picks: would buying help? (descriptive, 2023-2025)

Reproduced the dashboard pick (blend of walk-forward model margin + consensus, old key weights, best allowed main line,
EV ≥ 3%, price -200..+200; QB/injury gates not reproducible) at every alt snapshot: 474 qualifying snapshots, 302 games.

* +0.5 at the same book (257 with an alt price): actual median -123 vs dashboard estimate -120 (estimate 3.2c too
  cheap on average). Buying raised the dashboard's own EV in 8% of cases (6.1% -> 3.3% on average); under the
  corrected weights 8%; under the fair market 14% (main pick -2.9% -> -5.0%).
* +1.0 (404): actual -132 vs estimate -130; raised dashboard EV in 3% of cases.
* Best bought alt line at any allowed book: improves dashboard EV in 16% of cases, mean change -1.4 pts.
So the dashboard is right not to buy; the estimate just understates the cost a little (and a lot at MGM/Caesars).

## 5. Recommendation for the live system

1. Replace `buy_points_estimate` (bump spread_rules version; never paper-bet buys) with book-specific cents:
   BetMGM 25/20/15, Caesars 30/23/18 (rising), DraftKings 22/18/11, FanDuel 23/18/13 (half-point alts only),
   BetRivers 17/14/6, Fanatics 20/15/5, unmeasured books (ESPN BET, Hard Rock) 25/20/15 -- for 3/7/other.
   Show a buy only when it is a BetRivers/Fanatics buy onto 3 or 7 for a side we already like.
2. Live alternates are cheap enough to pull only where they matter: the current event-odds endpoint costs
   1 credit per market per region (historical is 10x). alternate_spreads with regions us,us2 = 2 credits per game call.
   For games whose spread pick qualifies (~6 games/week in 2023-25), once at the decision run: ~12-25 credits/week.
   Every game at every full run (~16 games x ~9 runs) would be ~300 credits/week -- not worth it given section 1.
3. Do not build a "+EV alt line" track: the model's tails are not accurate enough to price lines 2-4 pts off the main
   number, and that is where all its alt-line "edges" are.

Caveats: two snapshots per game (no intra-day prices); ~7% of snapshots dropped for lack of a main-market snapshot
within 90 min; Fanatics only in 2025; holdout ROI SEs are 6-11%, so only large effects could be detected; prices are
from the Odds API feed, not a bet slip.
