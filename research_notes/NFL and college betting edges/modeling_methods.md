# State-of-the-art modeling for NFL and FBS game outcomes vs betting markets

Research run 2026-10-04. Note on coverage: the search tool returned links without snippets, so findings below come only from pages actually fetched (about 10). Many key questions are only partly answered; see Gaps under each question. Claims not backed by a fetched page are marked as inferences or gaps, never as findings.

## 1. Methods with documented success vs closing lines (NFL and CFB)

### Takeaway
No fetched source documents a model that beats NFL or CFB **closing** lines over a large, out-of-sample, verified record. The best-supported public approach is to **blend the model with the market** (nfelo: about 65% market / 35% model, better Brier than either alone). Published "profitable" results come from tiny samples (one season, 15 to 127 games) or from opening lines and alt lines.

### Cited Findings
- nfelo (public NFL Elo-style model) combines its own spread with the Vegas spread as a weighted average, choosing weights to minimize prediction error. The best flat weight found was **65% market / 35% model**. — [nfelo: Using market regression to improve prediction accuracy](https://www.nfeloapp.com/analysis/using-market-regression-to-improve-prediction-accuracy-in-the-nfl/)
- nfelo's refinement is **error-weighted regression**. For each team it compares the model's recent squared error (an exponential average over recent games) with the market's error on that team. Teams where the model has recently beaten the market get less shrinkage toward Vegas; teams where it has trailed get more. The article says this beats both raw nfelo and the Vegas spread on Brier score, but gives no sample size, years or ATS record. — [nfelo](https://www.nfeloapp.com/analysis/using-market-regression-to-improve-prediction-accuracy-in-the-nfl/)
- Kuper (Berkeley undergraduate thesis): adding ESPN Pick'em fan consensus to closing moneylines was statistically significant for the 2011 NFL season (probit p=0.025, LPM p=0.019; 256 games). Fans picked winners 66.8% of the time vs 66.0% for the market. The claimed returns (+34% on 33 games; +72.7% on 15 out-of-sample games) rest on tiny samples and should be read as noise-level evidence. — [Kuper, UC Berkeley](https://econ.berkeley.edu/sites/default/files/Kuper.pdf)
- CFBD's public LightGBM/NGBoost CFB model used 714 features: recruiting (4 years), returning usage/production, talent, season stats, havoc, QB hurries, and also **pregame spreads and win probabilities**. Test RMSE was 15.72 (2019+) and 16.77 in 2020 production, vs 16.0 cited for Ed Feng's model. Against DraftKings **opening** lines it went **50% ATS on 127 games** in 2020. A claimed 68.5% on 35 alt-spread picks is too small to mean anything. — [CFBD blog: predicting spreads with GBDT](https://blog.collegefootballdata.com/predicting-spreads-gbdt/)
- A 2024 systematic review of ML in sports betting reports football results mostly as accuracy. For example, Patel (2023) XGBoost had 58.5% validation accuracy and 53.65% on 2021, and the review calls that "profitable at 10-11 odds". The review stresses that optimizing calibration rather than accuracy gives higher betting returns, citing Walsh & Joshi (2024; soccer/basketball context). — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)
- Bayesian work that puts bookmaker odds into the prior exists in soccer: Egidi, Pauli & Torelli, "Combining historical data and bookmakers' odds in modelling football scores" (2018). — [arXiv 1802.08848](https://arxiv.org/pdf/1802.08848)

### Inferences
- The pattern across sources matches our own experience: a pure stats model is a weak signal next to the market. The only documented improvement is **dynamic, team-level shrinkage toward the market**, where model weight depends on the model's recent relative error. We have not listed this as tried; our blends use fixed weights. It is cheap to test in `scripts/vs_vegas.py`:
  - per-team, or per-game-type, model weight from an exponentially weighted squared-error ratio of model vs market;
  - fit only on 2015-19, scored on 2020-25.
- The CFBD model used pregame spreads as **inputs**. Its "edge" is therefore a residual-on-market model, which we already tried for CFB (LightGBM residual). The 50% ATS on openers suggests no gain.
- Calibration-first model selection (log loss or Brier, then a reliability check, then betting only where calibrated probability beats no-vig price) is consistent with our CLAUDE.md rules.

### Gaps
- Not fetched: verified long-run records for Massey-Peabody, FTN/Football Outsiders DVOA picks, ThePredictionTracker ATS tables, Kevin Cole or Unabated ratings, PFF Greenline. The Prediction Tracker page (thepredictiontracker.com/ncaaresults.php) could not be fetched because of a tool restriction. A manual check is recommended.
- No fetched source on player-level WAR/plus-minus team ratings, drive-level simulation, or tracking-data (NGS/PFF) features showing value vs closing lines.
- No 2023-2026 SSAC or JQAS papers were retrieved with results. Searches returned only titles.

## 2. CFB-specific methodology (SP+, returning production, transfers, conference/market pricing)

### Takeaway
SP+ is built from opponent-adjusted play-by-play components:
- success rate (50% / 70% / 100% of needed yards on downs 1 / 2 / 3-4);
- IsoPPP explosiveness;
- finishing drives (points per trip inside the 40);
- field position;
- havoc and line yards.

Preseason SP+ now leans on **returning production with transfers credited at their previous team's production**. Connelly says transfer data has largely replaced recruiting weight. Peer-reviewed studies find interconference CFB mispricing for certain conferences, and an under bias after a rules change that the market fixed within the season.

### Cited Findings
- **SP+ components:**
  - Success-rate thresholds are 50% of yards to go on 1st down, 70% on 2nd, 100% on 3rd/4th.
  - IsoPPP is equivalent points per play on **successful plays only**.
  - Finishing drives is points per trip inside the 40.
  - Field position is measured as defense's starting field position for the offense, and vice versa.
  - Havoc is the share of plays with a TFL, forced fumble or pass defensed.
  - Line yards is an opponent-adjusted split of rushing credit between runner and blockers.

  — [Football Study Hall advanced-stats glossary (Connelly, 2015)](https://www.footballstudyhall.com/2015/2/9/8001137/college-football-advanced-stats-glossary)
- **Returning production weights (Connelly):**
  - Offense: OL **39.6%**, WR/TE receiving yards **35.0%**, QB passing yards **22.3%**, RB rushing yards **3.1%**.
  - Defense: snaps **65.9%**, tackles **19.2%**, TFL **14.9%**.

  — [ESPN via ABC7: returning production for 2026](https://abc7chicago.com/post/what-returning-production-looks-like-2026-college-football-season/18754857/)
- **Transfer handling:** an incoming transfer's production from his previous team goes into both the numerator and denominator for the new team. Players transferring up from lower divisions get **half credit**. Connelly says "recruiting rankings' weight ... has diminished significantly" in projections because of the portal. — [ESPN via ABC7](https://abc7chicago.com/post/what-returning-production-looks-like-2026-college-football-season/18754857/)
- In 2025, the top-10 returning-production teams improved on average by **1.0 wins and 6.4 SP+ spots**. Teams at or below 36% returning production mostly regressed. — [ESPN via ABC7](https://abc7chicago.com/post/what-returning-production-looks-like-2026-college-football-season/18754857/)
- Moore & Francisco (2019, Atlantic Economic Journal) studied CFB interconference spreads from Sept 2003 to Jan 2016. They rejected market efficiency for these matchups and found that betting certain conferences to cover was profitable over the period, while across all games profitability was limited. This bears on whether bettors over-rate major conferences. — [IDEAS/RePEc](https://ideas.repec.org/a/kap/atlecj/v47y2019i2d10.1007_s11293-019-09616-7.html)

### Inferences
- Untried CFB ideas that follow from this:
  1. **Position-weighted returning production with transfer credit**, using the OL-heavy offensive weights above, rather than generic returning PPA. Lower-division transfers at half credit.
  2. **Explosiveness on successful plays only** (IsoPPP-style) as a separate input from success rate. Our CFB set uses "explosiveness", but whether it is the conditional IsoPPP form should be checked.
  3. **Field-position and finishing-drives components** as separate opponent-adjusted factors.
  4. An **interconference/conference-tier ATS feature or filter**, tested walk-forward only. The 2003-2016 result may have decayed (see totals finding below).
- Recruiting composites likely lose weight in the portal era (2021+). Any preseason prior fit on pre-2021 data may over-weight recruiting. Refit or decay the weights for 2021+ seasons.

### Gaps
- Exact SP+ preseason formula weights (prior-season SP+ vs recruiting vs returning production) were not found in a fetched source.
- FPI, Sagarin and Massey methodology details were not retrieved.
- No fetched source covers small-conference pricing difficulty, FCS opponents, bowl opt-outs or motivation effects.

## 3. Totals models and known totals biases

### Takeaway
Documented CFB totals biases were temporary. An under bias followed the 2006 clock-rule change (47.2% overs) and faded by November. A 2003-2015 study finds no profitable naive over or under. That suggests mispricing comes from **structural changes** (rules, clock, overtime format) that the market has not absorbed yet, rather than from persistent biases.

### Cited Findings
- Paulson (2008):
  - 2003-2005 CFB totals were efficient: 50.26% overs on 1,924 games.
  - After the 2006 NCAA timing-rule change, only **47.23% of 714 games went over** (p≈0.07).
  - By November, overs recovered to 49.2% as bettors adapted.

  — [Paulson, Business Quest 2008](https://www.westga.edu/~bquest/2008/football08.pdf)
- Francisco & Moore (2018, J. Economics & Finance): using NCAA totals 2003-2015, they found the earlier over-bias (Paul & Weinbach 2005) "has largely corrected itself". Neither naive over nor naive under is profitable. — [IDEAS/RePEc](https://ideas.repec.org/a/spr/jecfin/v42y2018i4d10.1007_s12197-018-9437-y.html)

### Inferences
- An actionable "rules change" angle: early in seasons after CFB clock or overtime changes, test whether totals lag. Examples include the 2023 CFB rule letting the clock run after first downs (except the last 2 minutes of each half) and the 2019/2021 overtime format changes. This should be tested on our historical totals, not assumed.
- No fetched evidence was found on NFL primetime unders, pace/tempo totals models, or 4th-down aggressiveness effects on totals.

### Gaps
- No sourced data on NFL primetime under rates, pace-of-play (seconds per snap, neutral-situation pace) models, or how overtime affects CFB totals pricing.

## 4. Validation: avoiding overfitting, CLV vs ATS, how many bets

### Takeaway
The fetched sources back calibration-focused selection and show how easily small samples produce fake edges (15- to 127-game "profitable" results). No fetched source gave a rigorous CLV sample-size rule.

### Cited Findings
- Calibration rather than accuracy should drive model selection for betting returns (review citing Walsh & Joshi 2024). — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)
- The examples of sample-size problems in published "edges":
  - Kuper: 15-game out-of-sample "+72.7%".
  - CFBD: 35-pick alt-spread "68.5%" next to 50% on all 127 openers.

  — [Kuper](https://econ.berkeley.edu/sites/default/files/Kuper.pdf); [CFBD](https://blog.collegefootballdata.com/predicting-spreads-gbdt/)
- CFBD used a time-based split: train on games before 2019, test on 2019 onward. — [CFBD](https://blog.collegefootballdata.com/predicting-spreads-gbdt/)

### Inferences
These are standard statistics, not sourced here.
- At -110, beating 52.4% ATS with 95% one-sided confidence when the true rate is 55% takes roughly 1,000+ bets. That is why CLV (lower variance per bet) is the faster metric. This repo already uses CLV and pre-registered rules, which is consistent with that.

### Gaps
- Pinnacle and Unabated articles on CLV significance were not fetched. Search returned only titles. A primary source for the CLV-vs-results sample-size math is still needed.

## 5. Recent (2023-2026) papers and public models with verified records

### Takeaway
Retrieval of recent academic work was thin. The only 2024+ source fetched (an ML-betting systematic review) reports football results mostly as accuracy, not ROI vs closing lines. No public model with a verified long-run positive ATS record vs closing lines was confirmed.

### Cited Findings
- The 2024 systematic review of ML in sports betting (arXiv 2410.21484) covers SVM, neural network, XGBoost and HMM work on American football. Most of it is play-call prediction or win accuracy, with no audited betting record vs closing lines. — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)
- Other relevant titles appeared in search but were not fetched:
  - "The Performance of Betting Lines for Predicting the Outcome of NFL Games" ([arXiv 1211.4000](https://arxiv.org/pdf/1211.4000));
  - "Anchoring, affect, and efficiency of sports gaming markets around playoff positioning" ([UGA open journals, "fsr"](https://openjournals.libs.uga.edu/fsr/article/view/3242); journal name and year not confirmed).

### Inferences
- Untested ideas suggested by sources or titles:
  - **Anchoring around playoff positioning.** Late-season games where one team's seeding is settled; fits the backlog's clinch-aware feature.
  - **Crowd/consensus signals** (Kuper-style fan picks; public betting percentages) as a contrarian or confirmatory input, on a separate labeled track.

### Gaps
- Not verified: Massey-Peabody records, DVOA picks records, Prediction Tracker long-run leaders, Kaggle NFL competition outcomes (the Big Data Bowl is tracking-data analytics, not betting), PFF Greenline, SSAC 2023-2026 football betting papers. These need direct fetches of thepredictiontracker.com, masseypeabody.com, ftnfantasy.com and sloansportsconference.com.
