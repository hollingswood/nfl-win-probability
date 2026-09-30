# Line-movement model: bet early where the line is predicted to go

**Verdict: no edge.** The line's direction can be predicted a little, and that held up out of sample. But the move is too small to pay for the vig at our allowed books. All 3 frozen strategies failed on 2023-2025. The best one (moneyline, LM2) had CLV of +1.7%, t = 1.32, p = 0.094, which misses the 0.0167 bar. Its matched control (the same rule with a predicted move of zero) got +0.7%. The two spread strategies had negative CLV.

Code: `scripts/research/line_move_model.py` (dev / `--freeze` / `--holdout`). Frozen specs: `output/research/line_move_frozen.json`. All numbers: `output/research/line_move_model.json`.

## Setup
- **Snapshots.** Every game snapshot from 9 days out to 3 hours before kickoff. The close itself is about 75 minutes before kickoff and is excluded. Dev (2020-22): 10,322 snapshots from 834 games. Holdout (2023-25): 13,437 snapshots from 854 games.
- **Targets, all in home terms.**
  - `y_sp = mu_close_all - mu_px_all`: the price-implied close minus the price-implied consensus now. Both use the same key-number method, so this is the true move.
  - `y_sp_raw = mu_close_all - m_cons`: the target as originally specified.
  - `y_ml = logit(p_close_all) - logit(p_cons)`: the moneyline move.
- **CLV** is valued at the honest close. For spreads it is the side EV at `mu_close_all` for the exact point and price. For moneylines it is `dec * p_close_side - 1`, using `p_close_all`.
- **Stale-line filter.** Same as edge_lab: the offered point must be within 2.5 of consensus, the price between -200 and +200, and the moneyline implied probability within 0.12 of consensus.
- **Reference: betting every offer blind.** Mean CLV is -4.5% for both spreads and moneylines, in both periods.

## What predicts the move (dev, 2020-22)

| target / model | out-of-fold R² (leave one season out) | corr | forward-only R² |
|---|---|---|---|
| y_sp, ridge with 16 features | -0.020 | 0.03 | -0.035 |
| y_sp, small GBM | -0.050 | 0.01 | -0.051 |
| y_sp, ridge on `x_sharp` only (target clipped at ±3) | +0.002 | 0.06 | -0.008 |
| y_sp_raw, ridge | **+0.124** | 0.35 | +0.107 |
| y_ml, ridge on sharp + favorite features (clipped at ±0.3) | +0.010 | 0.10 | +0.003 |

- **The raw target's R² of 0.12 is an artifact.** It comes from `x_juice = mu_px_all - m_cons` (corr 0.38). The median point ignores juice, so a -3 line priced at -125 "moves" toward 3.3. That is a measurement effect, not a line move.
- **The real move is almost pure noise.** Its SD is 1.5 points, with fat tails from QB news. The only stable signal is `x_sharp`: the sharp books' price-implied margin minus all books'. Its corr was 0.05, 0.15 and 0.14 by season, and the coefficient is 0.58 (the consensus closes about 58% of the gap to the sharp books).
- **Weaker signals:**
  - Model minus market at eligible snapshots: corr about 0.05.
  - Moneyline drifting toward favorites (`z_fav`).
  - Public teams: corr +0.06 overall, but it flipped sign in 2020.
- **Nothing useful:** move so far, key-number position, spread across books.
- **The literal rule loses.** Betting the predicted-move side at the best price whenever the move is at least 0.25-1.0 point gave CLV of -2% to -4%. Predicted moves are much smaller than the vig. What worked better was scoring each offer by its EV at the predicted close, which combines the move and the price.

## Frozen strategies
All three use ridge (alpha 1) on a clipped target and take one bet per game, at the first snapshot where EV at the predicted close reaches the threshold.

| id | features (coefficients) | threshold | dev leave-one-season-out: bets, CLV, t | dev forward-only: CLV, t |
|---|---|---|---|---|
| LM1 spread | `x_sharp` 0.585, intercept -0.032 | EV ≥ 0.02 | 86, +3.2%, 2.32 | +0.9%, 0.52 |
| LM2 moneyline | `z_sharp` 0.259, `z_fav` 0.013, `x_sharp` 0.060, intercept -0.005 | EV ≥ 0.03 | 158, +3.4%, 2.94 | +2.0%, 1.67 |
| LM3 spread | `x_sharp` 0.577, `x_model` 0.038, intercept -0.041 | EV ≥ 0.01 | 168, +1.7%, 2.11 | -0.8%, -0.84 |

The dev evidence was fragile before the holdout ran:
- 2020 drove most of the dev CLV.
- The forward-only check was weak.
- About 350 configurations were tried in dev.

## Holdout 2023-2025 (run once)

| id | bets/season | CLV | t | p (one-sided) | beat close | ROI ± SE | zero-move control CLV | pass |
|---|---|---|---|---|---|---|---|---|
| LM1 | 37.3 | -0.29% | -0.26 | 0.60 | 48.2% | +5.2% ± 9.0% | -0.6% | no |
| LM2 | 77.3 | +1.66% | 1.32 | 0.094 | 53.4% | +12.3% ± 12.1% | +0.7% | no |
| LM3 | 79.0 | -0.81% | -1.15 | 0.88 | 50.2% | -1.4% ± 6.2% | -0.9% | no |

By season:
- LM1: 2023 -3.6%, 2024 +2.7%, 2025 +0.3%.
- LM2: 2023 -0.5%, 2024 +1.9%, 2025 +2.8%.
- LM3: 2023 -3.7%, 2024 +1.1%, 2025 -0.4%.

The models still predict direction on holdout (corr 0.11-0.16, R² 0.002-0.011). The problem is profit: once you pay the price, the predicted move does not beat the close.

## Caveats
- **`x_model` is optimistic.** It uses final-report injury inputs, which is known optimism, and it still failed (LM3).
- **Leave-one-season-out CV trains on later seasons.** The forward-only rows are the stricter check.
- **LM2 is the only mildly promising lead.** It is not an edge. Paper-tracking it forward is the only honest next step, and it should not be re-tuned on 2023-25.
