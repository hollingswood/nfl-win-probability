# Player props at three snapshots (Tue open / Fri early / close): 2023 dev, 2024-25 holdout (2026-10-02)

Code: `scripts/research/props_full.py` (`build` → `fit` → `prep` → `dev` → `holdout`). Numbers: `output/research/props_full.json`. Frozen rules: `output/research/props_full_frozen.json` (k=3, frozen after 2023 dev; 2024-25 run once).

Markets: pass yds, pass attempts, rush yds (RB only for model work), receptions, rec yds (O/U, main lines only: both prices in [−200, +170]), anytime TD (feed has only the Yes price).

Projections: `props.py` pipeline refit for three information sets. Each set uses the game lines (team totals + spread) from the same snapshot.
- **open:** no injury report.
- **early:** Friday Out/Doubtful report.
- **late:** actual actives, used at close.

TD model: LightGBM on red-zone carry/target shares, decayed TD rate and implied team total. It is calibrated out of sample (2023-25 predicted rate within 0.4 pp of actual).

## Data limit that shapes everything
Props are posted by Tuesday 14:10 UTC for only a minority of games. Events with an open snapshot:

| season | O/U markets | anytime TD |
|---|---|---|
| 2023 | 15-29 | 35 |
| 2024 | 43-57 | 80 |
| 2025 | 63-98 | 186 |

The player set at open is skewed to marquee players. Dev on 2023 open has n ≈ 20-170 player-games per market, so open-snapshot rules were developed with almost no power.

## 1. Line accuracy by snapshot
Consensus (modal) line MAE vs actual, 2024-25, on player-games with lines at all three snapshots:

| market | n | open | early | close | open − close (SE) | open line − our open model |
|---|---|---|---|---|---|---|
| pass yds | 292 | 59.09 | 58.59 | 58.59 | +0.50 (0.42) | +0.47 (0.99) |
| pass att | 257 | 6.38 | 6.37 | 6.33 | +0.05 (0.05) | −0.08 (0.11) |
| rush yds | 295 | 23.54 | 23.46 | 23.46 | +0.08 (0.21) | +0.08 (0.66) |
| receptions | 811 | 1.652 | 1.637 | 1.622 | +0.030 (0.013) | +0.017 (0.025) |
| rec yds | 936 | 22.44 | 22.39 | 22.37 | +0.07 (0.10) | −0.41 (0.27) |

- **Opening lines are not soft in accuracy terms.** They are 0.2-1.8% worse than the close, and only receptions is significant. Our open projection does not beat the open line anywhere.
- **Early vs close:** on the larger early/close sets the gap is ≤0.3%.

Anytime TD, AUC of market vs model, 2024-25:

| snapshot | market AUC | model AUC |
|---|---|---|
| open | .758 | .754 |
| early | .762 | .754 |
| close | .766 | .756 |

Adding the model to the market improves out-of-sample log loss (fit 2023, test 2024-25):
- **early:** +0.0013 ± 0.0004
- **close:** +0.0007 ± 0.0002
- **open:** +0.0035 ± 0.0032

So there is small real TD information, but it is tiny next to the Yes-side vig. Allowed-book Yes quotes average about −27% CLV.

## 2. Line moves open → close
- **Size of moves:** the mean |move| of the price-adjusted median is about 5.6 pass yds, 0.75 attempts, 2.6 rush yds, 0.37 receptions and 2.3 rec yds. Most of it happens by Friday.
- **Direction:** moves correlate with our open projection-minus-line gap.

  | market | corr 2024 | corr 2025 |
  |---|---|---|
  | pass yds | .36 | .31 |
  | pass att | .28 | .29 |
  | receptions | .36 | .29 |
  | rec yds | .19 | .10 |
  | rush yds | .00 | .15 |

  The sign of a predicted move is right 60-80% of the time.
- **Out-of-sample R² of the move:** about 0 or negative for yardage markets, +0.05 to +0.09 for receptions and attempts. Adding Friday and late information lifts receptions to 0.28.
- **Why it doesn't pay:** the predictable part is a fraction of a point, far smaller than the ~7% hold. That is why model bets at open lose CLV.

## 3. Frozen rules, 2024-25 (run once)
Pass bar: CLV > 0, one-sided p < 0.0167, and positive in both 2024 and 2025.

| rule | dev 2023 n / CLV | 2024-25 n | bets/wk | CLV ± SE | 2024 / 2025 CLV | ROI ± SE | hit | pass |
|---|---|---|---|---|---|---|---|---|
| td_shop_early (Yes vs calibrated LOO median, 0.10 ≤ EV < 0.6) | 411 / +4.1% | 351 | 9.2 | −6.3 ± 1.6% | −2.1% / −20.6% | +2.7 ± 22.7% | 9.1% (avg dec 17.4) | no |
| rec_shop_early (receptions, EV ≥ 2% vs LOO no-vig median of ≥3 other books, same point) | 87 / +1.3% | 278 | 6.5 | **+2.74 ± 0.40%** (t 6.9) | +1.6% / +3.7% | −6.0 ± 6.5% | 44.2% (avg dec 2.13) | **yes** |
| open_model_thesis (all O/U at open, w=0.2, EV ≥ 12%) | 34 / −0.3% | 293 | 7.3 | −1.3 ± 0.5% | −0.4% / −1.9% | +8.7 ± 6.2% | 53.9% | no |

**rec_shop_early checks**
- **CLV source:** 98% direct (the same point at the close), and 70% of bets beat the close.
- **Under-bias adjustment:** the no-vig close overrates overs. Shading overs down and unders up by the 2023 shortfall (1.4 pp) leaves CLV at +2.3 ± 0.4%. Using the larger 2024-25 shortfall (3.1 pp, in-sample) leaves +1.7 ± 0.5%.
- **ROI:** realised ROI is −6.0 ± 6.5%, about 1.3 SE below CLV. Unders went −11% on 118 bets.
- **Books:** BetMGM supplies 42% of bets (CLV +3.8%, ROI +3.9%).
- **Selection:** this market was picked from 5 markets × thresholds of shop rules, and the other markets ran about −1% CLV in dev.

**td_shop_early caveat:** one-sided TD CLV depends heavily on the calibration map. Refitting the map on 2024-25 (in-sample, descriptive only) turns 2024 CLV into +20%. Treat TD CLV as unreliable, and the rule as failed under its pre-registered definition.

## 4. Systematic biases
- **Under bias exists everywhere.** On 50%-priced lines in 2024-25, overs hit:

  | market | early | close |
  |---|---|---|
  | receptions | 47.5% | 46.6% |
  | rec yds | 48.0% | 47.2% |
  | rush yds | 48.1% | 47.2% |
  | pass att | 46.2% | 46.8% |
  | pass yds | 50.1% | 50.4% |

  The close carries the same bias, so blind unders have about −5% CLV. ROI at the best allowed book at early is +0.4% for rush yds, +1.3% for receptions, −0.3% for rec yds, +2.8% for pass att and −5.3% for pass yds.
- **Players with Tuesday props (marquee players):** over rates are far lower (rush yds 40%, receptions 40-44%, rec yds 44-46%) at every snapshot. Blind under at open, 2024-25:

  | market | ROI ± SE |
  |---|---|
  | rush yds | +13.8 ± 5.1% |
  | receptions | +6.0 ± 2.9% |
  | rec yds | +4.2 ± 3.2% |

  This was not visible in the thin 2023 dev sample. It was found in the holdout, so it is **not a validated result**. The 72%-under open_model_thesis ROI (+8.7%) is this same effect.
- **Game script:** effects are mostly unstable across seasons (2023 rush-yds dog unders faded). Receiving yards on big underdogs (≤ −6) ran under in all three years: over rates of 46%, 46% and 39% at the close.
- **Anytime TD:** strong favourite-longshot pattern. Close implied 6.4% vs actual 4.4%; implied 15% vs actual 9-11%.

## Verdict
- **Opening-line thesis:** not supported. Openers are about as accurate as closes, and our open edges have negative CLV.
- **TD props:** too much vig. Shopping fails.
- **Receptions line shopping at Friday:** the only frozen rule that passes CLV, and robust to under-bias adjustment, but its realised ROI is negative and noisy.
  - **Paper-track only:** receptions O/U, Friday 21:40 UTC snapshot (kickoff−24h otherwise), allowed books. Bet the side/book whose price beats the leave-one-out median no-vig probability of ≥3 other US books quoting the **same point** by EV ≥ 2%. One bet per player, flat stakes, graded on CLV vs the close same-point no-vig median (and ROI).
  - **Expectations:** about 6-7 bets per week, about +2-3% CLV, and limits of tens to low hundreds of dollars.
- **Separate hypothesis (pre-register for 2026, grade on ROI/hit rate, not CLV):** under on rush yds / receptions / rec yds for players whose props are posted by Tuesday open.

## Limits
- **Optimistic inputs:** recorded weather in the projections.
- **2023 open sample:** tiny.
- **Missing books:** espnbet and hardrockbet are absent from the feed.
- **TD voids:** TD bets on players with no offensive snap are dropped. Some of those would be graded losses at books.
- **Prop limits and account restrictions:** not modelled.
