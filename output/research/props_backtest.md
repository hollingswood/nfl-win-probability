# Receiving-yards props vs the market: backtest (2026-10-01)

Code: `scripts/research/props_backtest.py` (`prep` → `dev` → `holdout`). Numbers: `output/research/props_backtest.json`. Frozen rules: `output/research/props_frozen.json`.

Data: Odds API `player_reception_yds`, regions=us, two snapshots per game. Early = Fri 21:40 UTC for Sunday games, kickoff−24h otherwise. Close = kickoff−75 min. Seasons 2023-25, REG+POST.

Model: `props.py` walk-forward GBM in **early** mode (teammate availability from the final injury report only) plus its conditional-empirical distribution.

Discipline: thresholds, filters and the off-point valuation constant were set on **2023 only**. Three rules were then frozen and 2024-25 was run once.

## Bottom line
**No props paper track is warranted.**
- **Information test:** out of sample, the model adds nothing to the closing line.
- **Rules:** all three frozen rules have clearly negative CLV.
- **ROI:** realised ROI is about the same as blindly betting every under at the best allowed book.

## 1. Matching
- **Coverage:** 855 events. 2023 has 274 of 285, and the BUF-PIT wildcard is excluded because it was postponed and the feed's commence date is wrong.
- **Name rules:** lower-case, strip accents, punctuation, parentheses ("Michael Thomas (NO)") and Jr/Sr/II-V. Small alias list (e.g. Hollywood Brown). The fuzzy fallback must match the first name, which prevents Deonte Harris matching Damien Harris. Initial+last is used only for "J. Hill"-style names.
- **Match rate (prop player-games):** 99.5% / 100% / 99.8% resolved for 2023 / 2024 / 2025.
  - 2023: 3,718 played, 35 did not play (void), 2 unmatched.
  - 2024: 3,459 played, 40 void.
  - 2025: 3,725 played, 50 void, 7 unmatched (Travis Hunter, listed as CB in nflverse).
- **Name methods:** 10,861 exact, 26 fuzzy, 15 initial+last.
- **Model coverage of played rows:** 92% / 93% / 88%. Rookies with no prior game have no prediction.
- **Main-line filter:** Kambi books (betrivers, unibet) post 25-33% of their 2023-24 lines as rounded "milestone" points at skewed prices (e.g. 29.5 at +450 when the market is 13.5). These 4.6K of 124K quotes are dropped: both prices must be within [-200, +170]. Without this filter, tail mispricing produced spurious positive CLV in dev.

## 2. Market vs model accuracy (rec yds, MAE in yards; same player-games within each row)
| period / snapshot | n | consensus line | model median (early) | model median (late) | ½ line + ½ model |
|---|---|---|---|---|---|
| 2023 early | 2,466 | **21.96** | 22.47 | 22.37 | 21.96 |
| 2023 close | 2,978 | **20.96** | 21.53 | 21.44 | 21.00 |
| 2024-25 early | 5,510 | **20.76** | 21.43 | 21.36 | 20.90 |
| 2024-25 close | 6,473 | **19.89** | 20.54 | 20.49 | 20.01 |

- **Accuracy gap.** The line beats the model by about 2.5-3% at both snapshots. Averaging the line with the model does not help.
- **Model sits low.** The model median is about 2-3 yds below the line on average.
- **Under bias.** Overs hit only 47-49% at no-vig 50%, so the line sits slightly high. That is the known prop under bias.

**Information test.** Logistic regression of the over outcome on logit(market no-vig P at the consensus line) plus logit(model P), fit on 2023 and scored on 2024-25:
- **Fit on 2023:** the model coefficient is 0.25 ± 0.09 at early and 0.17-0.19 ± 0.08 at close.
- **Scored on 2024-25:** adding the model makes log loss *worse*.
  - Early: −0.0012 [−0.0027, +0.0002].
  - Close: −0.0007 [−0.0016, +0.0003].
- **Verdict:** no evidence of information the market lacks.

## 3. Betting backtest (early snapshot)
**Bet construction**
- **Blend:** shift the model distribution toward the market by (1−w) × c_early.
- **Selection:** one bet per player-game, the best-EV side and allowed book. Flat stakes.

**Grading:** official-style pbp yards.

**CLV method**
- **Direct:** the close no-vig price at the same point, when a close book quotes that point. This covers 38-59% of bets.
- **Anchored:** otherwise, the close consensus P at the modal line, moved along the model distribution damped by k = 0.55. k was estimated on 2023 close quotes: 0.58 [0.21, 0.94] vs outcomes and 0.54 vs books.
- **Why damped:** a pure shift (k = 1) overstates off-point value.
- **Sensitivity:** k = 0.3 and k = 1.0 are both reported. Every variant is negative.

**Frozen rules.** The pass bar is CLV > 0 with one-sided p < 0.0167 (0.05/3).

| rule | dev 2023: n / CLV / ROI | **2024-25 n** | **CLV (t)** | direct-only / k=1 CLV | **ROI ± SE** | hit | bets/wk | pass |
|---|---|---|---|---|---|---|---|---|
| blend w=.25, EV≥3% | 588 / −3.8% / +5.8±3.9% | 1,237 | **−3.3% (−23)** | −4.7% / −1.9% | **+0.6 ± 2.7%** | 53.0% | 28 | no |
| pure model, EV≥12% | 1,012 / −4.4% / +4.6±2.9% | 2,149 | **−4.0% (−39)** | −4.9% / −3.2% | **−0.3 ± 2.1%** | 52.4% | 49 | no |
| line shop w=0, EV≥2% | 146 / −2.0% / −9.6±8.4% | 468 | **−1.8% (−8)** | −4.7% / +1.1% | **+2.9 ± 4.4%** | 54.5% | 11 | no |

**Baselines (descriptive, not frozen), every player at the best allowed price, 2024-25:**
- **Always under:** ROI +0.05 ± 1.3%, hit 52.9%, CLV −5.2%.
- **Always over:** ROI −7.1%.

The rules lean heavily to unders (71-74% of model-rule bets). Their ROI is the under bias, not model skill. The close does move toward rule picks, by 1.2-2.3 pp versus 0.6-1.0 pp for blind picks, but that is far below the ~5-6% vig per bet. Seasons agree: every rule is negative on CLV in both 2024 and 2025.

## 4. Practical
- **Hold:** median 6.9% (about −115/−115). DraftKings is lowest at 5.9% in 2024-25, Caesars (williamhill_us) highest at 7.6%. Fanatics appears only in 2025. espnbet and hardrockbet never appear in the feed.
- **Best number (2024-25 early):** BetMGM is best on 32% of overs and 26% of unders. FanDuel and DraftKings are next.
- **Volume:** 11 / 28 / 49 bets per week for the three rules (NFL weeks with props).
- **Limits (not measured here):** US prop limits are typically tens to low hundreds of dollars, and winning prop accounts are restricted quickly.

## Caveats (they make the result *less* negative than reality)
- **Optimistic inputs:** the early model uses recorded weather and nflverse closing spread/total as features.
- **Unequal coverage:** the early snapshot has fewer lines than the close (about 55K vs 70K quotes).
- **Untested windows:** line-open (Tue/Wed) and post-news windows were not pulled and are untested.
