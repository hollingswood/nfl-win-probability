# CFB frozen rules: one-shot holdout

| rule | bets | W-L | cover/hit | p (vs 52.38%) | result | notes |
|---|---|---|---|---|---|---|
| R1 opener spread (model vs open gap ≥ 5) | 416 | 197-219 | 0.474 | 0.980 | fail | line moved our way 49% / against 39%, mean +0.47 pts; cover vs close 0.472 |
| R2 close residual (LightGBM, top-20% threshold) | 398 | 209-189 | 0.525 | 0.479 | fail | threshold 2.69 pts |
| R3 close totals (prediction vs close gap ≥ 3) | 1379 | 719-660 | 0.521 | 0.571 | fail | overs 73% of bets |

By season:

- R1 2024: 214 bets, cover 0.448
- R1 2025: 206 bets, cover 0.500
- R2 2022: 92 bets, cover 0.506
- R2 2023: 100 bets, cover 0.545
- R2 2024: 118 bets, cover 0.521
- R2 2025: 97 bets, cover 0.526
- R3 2022: 356 bets, hit 0.490
- R3 2023: 386 bets, hit 0.533
- R3 2024: 323 bets, hit 0.562
- R3 2025: 332 bets, hit 0.502
