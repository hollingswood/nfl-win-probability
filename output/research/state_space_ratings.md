# State-space (Kalman filter) team ratings: research result

**Verdict: do not adopt.** No variant meets the adoption rule (validation and holdout log loss both improve by at least 0.001 over production under the same walk-forward protocol). 0 of 25 candidates pass. No variant adds information beyond the Vegas closing line.

Script: `scripts/research/state_space_ratings.py` · raw results: `output/research/state_space_ratings.json`
Run: `cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/state_space_ratings.py` (about 18 min; `--reuse-tuned` skips refitting).

## Method

- **Latent state.** Each team has a net strength θ (in points). All 32 teams share one full covariance matrix.
  - Within a season, θ follows a random walk with variance `q_w` per week elapsed.
  - Between seasons: θ ← ρ(θ − mean) and P ← ρ²P + `q_off`·I.
  - Observation: margin = θ_home − θ_away + HFA·home_field + ε, with ε ~ N(0, 13²). The filter starts in 2002; 2002–2011 is warm-up only.
- **Variants.**
  - `margin`: margin capped at ±21. The cap was picked from {none, 28, 21, 14} on 2012–19.
  - `margin_gauss`: same model, tuned on the Gaussian predictive likelihood of the margin instead of win log loss.
  - `epa`: observation is the GLS composite w·margin + (1−w)·k·(net EPA/play margin), with k = 25.6 (the OLS slope, 2012–19).
  - `epa_tied`: `epa` with the composite's home-field term tied to HFA and floors on the noise terms.
  - `qb` / `epa_qb` / `epa_qb_tied`: QB-neutral ratings. The observation includes β·(starter QB rating edge), using production's leak-tested pre-game QB ratings, and that term is added back for the actual starter at prediction time.
- **Offense/defense split: not fit.** With iid score noise and symmetric O/D priors, the margin prediction is exactly the same as the net-rating model's, because the total only informs off−def.
- **Tuning.** Nelder-Mead on one-step-ahead predictive loss over 2012–2019 only. P(home win) = Φ(μ/√(h'Ph + s_p²)).
- **Pre-game ratings.** For each date, ratings are read before any game on that date updates the filter.

| variant | tune obj (2012-19) | q_w | q_off | ρ | HFA | other |
|---|---|---|---|---|---|---|
| margin (cap 21) | 0.6216 | 0.96 | 7.3 | 0.54 | 1.78 | s_p 7.8 |
| margin_gauss | NLL 3.843 | 0.88 | 10.0 | 0.60 | 1.79 | s_p 9.9 |
| epa | 0.6217 | 0.21 | 0.7 | 0.62 | 2.49 | w .52, hfa_z −4.4 (implausible) |
| epa_tied | 0.6231 | 0.26 | 1.5 | 0.60 | 1.87 | w .48, s_z 6.3 |
| qb | 0.6132 | 0.48 | 6.5 | 0.56 | 1.96 | β 21.2 |
| epa_qb | 0.6125 | 0.00 | 0.00 | 0.59 | 2.41 | degenerate (s_z 0.09) |
| epa_qb_tied | 0.6127 | 0.10 | 0.8 | 0.64 | 2.04 | w → 1.0 (EPA dropped) |

When the EPA observation is constrained to be sensible, it does not help: `epa_tied` is worse than margin-only, and `epa_qb_tied` pushes the EPA weight to zero. The unconstrained EPA fits reach their scores through an implausible negative composite home-field term.

## Results (walk-forward ridge, identical to `scripts/experiment.py`)

| model | val 2015-19 LL | hold 2020-25 LL | hold Brier |
|---|---|---|---|
| **Production (current)** | **0.6192** | **0.6209** | 0.2160 |
| KF alone: margin | 0.6311 | 0.6366 | 0.2229 |
| KF alone: qb (best standalone on holdout) | 0.6219 | 0.6266 | 0.2187 |
| KF alone: epa_qb (best standalone on val) | 0.6200 | 0.6275 | 0.2189 |
| Prod + ss_diff (margin) | 0.6191 | 0.6210 | 0.2161 |
| Prod + ss_diff (epa_qb), best val candidate | 0.6186 | 0.6213 | 0.2162 |
| Prod + team+QB rating (qb) | 0.6190 | 0.6210 | 0.2161 |
| Prod, Elo replaced by ss_diff (margin) | 0.6189 | 0.6212 | 0.2162 |
| Prod, Elo+pt_diff replaced (margin) | 0.6200 | 0.6209 | 0.2161 |
| Prod × KF-variance conversion | 0.6191 | 0.6208 | 0.2160 |
| Vegas closing (no-vig) | 0.6187 | **0.6070** | 0.2100 |
| Vegas recalibrated | 0.6186 | 0.6068 | 0.2098 |
| Production + Vegas blend | 0.6143 | 0.6091 | 0.2107 |
| (Prod + ss_diff epa_qb) + Vegas blend | 0.6141 | 0.6096 | 0.2110 |
| KF epa_qb + Vegas blend | 0.6157 | 0.6118 | 0.2120 |
| Prod + KF + Vegas blend | 0.6142 | 0.6097 | 0.2110 |

- **Changes vs production.** Across all 25 candidates, the largest improvement is −0.0006 on validation and −0.0001 on holdout, and holdout paired standard errors are about 0.0005. Some variants are worse, by up to +0.0029 on validation. Adding the rating to production helps validation slightly (up to −0.0006) and then gives it back on holdout (+0.0001 to +0.0011). That is the pattern of a feature that duplicates Elo and point differential (correlation with elo_diff is 0.90–0.97).
- **Blend vs Vegas.** Blends are fit on 2015–19 and scored on 2020–25. Every blend is worse than raw closing on holdout: +0.0020 for production and +0.0026 to +0.0048 with the KF rating added. **No information beyond the closing line.**

## Uncertainty (question e)

- **The uncertainty is nearly constant.** The tuned filter regresses hard between seasons (ρ ≈ 0.55) and lets strength drift quickly within a season, so the posterior standard deviation of the rating difference is about the same in week 1 and week 10 (margin variant: 5.0 vs 5.2 points).
- **Using the filter's own variance does not help.** Converting with it is slightly worse than a fixed σ (hold 0.6375 vs 0.6366).
- **Other ways of using the uncertainty.** Adding the variance as a ridge feature changes log loss by +0.0002/+0.0002 (val/hold). Adding an early-season (weeks 1–4) × rating interaction changes it by +0.0002/+0.0002. Heteroscedastic Φ conversion of production changes it by −0.0001/−0.0001.
- **Early season.** Production log loss in weeks 1–4 on holdout is 0.633; the KF rating is worse there (0.642).

## Leakage test

- **Setup.** Mirrors `test_no_future_leakage`: erase all scores and play-by-play on or after the cutoff, recompute everything including QB ratings, and compare ratings, μ, and variance for every game dated on or before the cutoff. Cutoffs are 2019-11-10 and 2023-10-01, and all 7 variants were tested.
- **Result: passed.** The maximum absolute difference is 0.0 across 4,686 and 5,707 games.
- **The check has power.** Ratings more than 7 days after the cutoff move by up to 9 points.
