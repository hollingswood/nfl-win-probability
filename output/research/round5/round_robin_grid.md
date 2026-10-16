# Round robin grid: every 2- to 6-leg combination (pre-registered 2026-10-16): no format passes

Rules: `round_robin_grid_prereg.json`. Code: `scripts/research/round5/round_robin_grid.py`. Each week: the top N picks (by EV,
different games), EVERY combination of k legs, equal stake per parlay, 1 unit per week in total; compared with the same N picks as singles.
Pass = beats singles on realized ROI and CLV-implied ROI in BOTH 2021-22 and 2023-25 with drawdown <= 1.5x singles.

Edge legs (college shop vs sharp at your books): 0 of 45 pass. Ordinary legs (closing spreads, -110): 0 of 22 pass.

## Edge legs, realized ROI per week's stake (2021-22 / 2023-25), weeks in sample

| N picks | weeks | singles | by 2 | by 3 | by 4 | by 5 | by 6 | 2-6 all |
|---|---|---|---|---|---|---|---|---|
| 4 | 21 / 40 | -5% / -3% | -2% / -1% | -17% / +5% | -37% / +33% | — | — | -11% / +4% |
| 5 | 18 / 40 | -8% / -1% | -9% / -2% | -12% / -3% | -12% / +3% | +41% / +81% | — | -9% / +2% |
| 6 | 17 / 36 | -2% / +2% | -2% / +9% | -9% / +22% | -30% / +48% | -75% / +109% | -100% / +284% | -21% / +39% |
| 7 | 13 / 30 | -3% / +5% | -4% / +11% | -10% / +25% | -24% / +53% | -54% / +96% | -100% / +162% | -26% / +51% |
| 8 | 9 / 27 | +7% / +4% | +19% / +13% | +33% / +30% | +41% / +62% | +31% / +118% | -19% / +211% | +27% / +79% |
| 9 | 7 / 26 | +18% / +5% | +35% / +14% | +47% / +30% | +49% / +54% | +32% / +78% | -8% / +90% | +32% / +60% |
| 10 | 7 / 15 | +17% / +10% | +37% / +24% | +59% / +43% | +77% / +68% | +84% / +95% | +69% / +107% | +72% / +80% |
| all | 34 / 51 | -10% / -4% | -26% / -15% | -33% / -22% | -36% / -26% | -48% / -32% | -58% / -29% | -42% / -23% |

## Ordinary legs (random closing spreads at -110), 2023-25 realized ROI

| N | singles | by 2 | by 3 | by 4 | by 5 | by 6 | 2-6 all |
|---|---|---|---|---|---|---|---|
| 4 | -14% | -28% | -45% | -70% | — | — | -38% |
| 6 | -12% | -24% | -34% | -45% | -62% | -100% | -38% |
| 8 | -6% | -14% | -24% | -34% | -41% | -52% | -33% |
| 10 | -6% | -15% | -26% | -38% | -50% | -61% | -44% |

## Reading it
- A parlay amplifies whatever its legs did that week. Busy weeks (8+ qualifying picks) happened to go well as singles in both periods (+4% to +18%), so their parlays look great; but there were only 7-9 such weeks in 2021-22 and 15-27 in 2023-25, and the drawdowns ran 1.6-5x singles, so none pass.
- Using every pick every week (N = all), every size loses in both periods (-15% to -58%), even though the legs beat the closing line: the CLV edge per leg (+0.7% to +2.4%) is too small to survive the variance and the losing legs.
- Ordinary legs: every size loses, and more with each leg (-14% singles-equivalent at -110 compounds to -24% to -100%).
- 6-leg round robins cashed 1-5% of tickets; 2-leg 21-29%.
