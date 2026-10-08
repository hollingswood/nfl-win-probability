# Quant modelling and pricing angles for NFL/college markets that most bettors miss

Scope note: there were 19 web calls. Two fetches failed. The Springer "Holdover Bias in the College Football Betting Market" page returned HTTP 429 (rate-limited by the proxy), and the Emerald SEF "divisional rivals" paper returned a bot-check page. The accessecon 2022 PDF redirect loop also could not be read. For those academic items, only the title or existence is cited, never their numbers. The repo has already run many studies on these topics (output/research/*.md), and they are cited where they close or narrow an angle, so the report writer does not propose re-tests.

Evidence-quality tags used below:
- **[A]** primary data or official source
- **[B]** reputable secondary or practitioner source with numbers
- **[C]** illustrative or educational, no measured evidence
- **[R]** this repo's own pre-registered study

## Q1. Cross-market consistency: props to team scoring, team totals vs game total vs spread, halves vs full game, alt ladders vs main

### Takeaway
Most of the "obvious" consistency trades have already been tested in this repo, and they fail at retail holds. That covers half/team totals vs the full game, moneyline vs spread, opening derivatives and alt ladders. The untested piece is the *props-to-game* direction: player-prop lines aggregated into implied team points, used as a lead/lag signal for the team total and game total. The repo's hourly 2026 snapshots now make that lead/lag measurable.

### Cited Findings
- **Half and team-total derivatives vs full-game fair [R].** Distributions were fit on 2012-22 closing lines. The sharp derivative book was BetOnline (LowVig posts none). Allowed books posting derivatives: DK, FD, MGM, Caesars, BetRivers. Team-total coverage thinned after 2023 (2025: only BetOnline, BetRivers, FanDuel and part of Caesars). — [output/research/derivatives.md](/home/claude/nfl-win-probability/output/research/derivatives.md)
- **Derivatives and alt ladders at the Tuesday open [R].** "The Tuesday open does not make derivatives or alt spreads beatable… the opening fair is too noisy and the move too small to clear the 4-5% hold… Opening alt ladders are no softer than later ones… No rule passes." The only live derivative idea left is cross-book shopping 75 minutes before kickoff (derivatives_v2 R1). — [output/research/openers_derivs.md](/home/claude/nfl-win-probability/output/research/openers_derivs.md)
- **Moneyline vs spread [R].** A total-aware key-number margin model gives no out-of-sample gain on 2020-25. Exact-margin log loss changed by -0.0007 ± 0.0006, and cover outcomes around 3/7/10/14 by +0.0000 ± 0.0001. — [output/research/ml_spread_consistency.md](/home/claude/nfl-win-probability/output/research/ml_spread_consistency.md)
- **Props [R].** "Openers are about as accurate as closes… TD props: too much vig. Shopping fails." The only CLV pass is receptions same-point shopping on Friday (about +2-3% CLV, 6-7 bets a week). The feed carries only the *Yes* price for anytime TD. Props are posted by Tuesday for only a minority of games: anytime TD on 35 events in 2023 and 80 in 2024. — [output/research/props_full.md](/home/claude/nfl-win-probability/output/research/props_full.md)
- **Other prop markets [R].** None pass. Pass-TD unders return -7.0% ± 2.7; solo-tackle unders +5.0% ± 2.6 (p = 0.03, below the p < 0.006 bar). — [output/research/props_more.md](/home/claude/nfl-win-probability/output/research/props_more.md)
- **Middles [R].** True middles appear in about 12% (spreads) and 16% (totals) of game-week snapshots, but are worth about -5% to -6% per package. +EV packages are worth about +0.1 unit per season per market. — [output/research/middles_segments.md](/home/claude/nfl-win-probability/output/research/middles_segments.md)

### Inferences
- **Untested angle P1: props-implied team points as a leading indicator.**
  - **Mechanism.** Prop markets at some books are set and moved by separate traders or feeds from the game lines. When QB-out or WR-out news hits, prop boards are often pulled and re-posted (or the reverse), so for minutes to hours the two markets can disagree.
  - **Construction (per team, per snapshot):**
    - E[offensive TDs] ≈ Σ_players P_noVig(anytime TD).
    - The feed has Yes prices only, so calibrate a single overround-shrink factor on 2023 against realised TD counts. The repo's TD LightGBM is calibrated within 0.4 pp, which gives a fit to anchor to.
    - Implied points ≈ 6.95·E[TD] + 3·E[FG]. Get E[FG] from the kicker-points prop if present, otherwise from a fitted FG/TD ratio.
    - Second proxy: regress closing team totals on QB pass-yds median + Σ RB rush-yds medians + Σ receiving-yds medians, fit on 2023.
  - **Test.** Dataset: Odds API props 2023-25 (three snapshots) + 2026 hourly snapshots. Compute the gap G = props-implied team total − game-line-implied team total ((T ± S)/2).
    - Pre-register: |G| ≥ 1.5 points → bet the team total (or the game total) in the direction of the props at an allowed book.
    - Grade on CLV against the close of the same market.
    - Also run a lead/lag Granger test on the 2026 hourly snapshots: does ΔG(t) predict Δ(team total)(t+1 h) or the reverse?
    - Pass bar, as in house rules: one-sided p < 0.05/k, positive in each season.
  - **Expected size.** Small and rare. The most likely use is as a *timing* filter, much as totals.md found model disagreement predicts line moves.
- **Same-point consistency inside the prop board.** Example: a QB pass-yds line versus the sum of his receivers' receiving-yds lines. Passing yards ≈ Σ receiving yards (sacks do not count against passing yards in NFL stats). This is a free internal identity: Σ_receivers median rec yds should ≈ QB median pass yds × (share of team targets with posted props).
  - Test on the 2023-25 props: measure the identity residual. Check whether the market with the larger move toward consistency by the close was the stale one. Bet the stale one at Friday early.
  - **Caveat.** Only the receivers with posted props are included, so the share has to be estimated from nflverse target shares (shift(1) to avoid leakage).
- Do not re-run half/team-total vs full-game or alt-ladder tests. The repo has closed them.

### Gaps
- No public, quantified study of props-vs-game-line inconsistency was found. A search for Unabated material on this returned only spam and SEO pages. The mechanism above is practitioner lore and is not sourced.
- The Odds API anytime-TD feed lacks the No price, so a proper de-vig is impossible. Any implied-TD sum relies on an assumed margin.

## Q2. Correlation pricing: same-game parlays and known correlation structures

### Takeaway
SGP pricing uses marginal probabilities plus a copula or empirical correlation, with holds of roughly 15-25%, which is several times single-market holds. Without SGP price data (the Odds API does not carry it), the only usable output here is a correlation table estimated from nflverse/CFBD. That table can price correlated *singles*, for example betting a team total and a QB prop together, or support SGP spot checks done by hand.

### Cited Findings
- **How SGPs are priced [C].** Steps: estimate each leg's marginal, estimate the joint via a Gaussian copula or empirical frequencies, add margin, round, then adjust for action. Typical correlations in sports-betting outcomes run about -0.4 to +0.6. Example correlations (team win vs QB 275+ yds ρ = 0.35; win vs over ρ = 0.28; QB 275+ vs over ρ = 0.42) are **illustrative, not measured**. — [Wizard of Odds, SGP mathematics](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- **Worked example and hold [C].** The article claims a 500-game empirical example (favorite by 3-7, total 45-51): favorite wins 58%, QB over 275 yds 51%, over 53%. All three hit together 20.4%, against 15.7% under independence (1.30×). It quotes SGP hold as "routinely… 15-25% or higher" versus about 4-5% on singles. It gives **no quantified evidence of systematic mispricing**, and says books "are more likely to misprice novel combinations" without measuring it. — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- **Negative-correlation example [C].** At ρ = -0.30, the joint is 19.2% against 24.8% under independence. — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)

### Inferences
- **Correlation matrix (no price data needed).** From nflverse pbp/weekly stats 2012-25, estimate pairwise and rank correlations, conditional on spread/total buckets, among:
  - game total over/under
  - favorite cover
  - each team's points
  - QB pass yds and attempts
  - top-2 WR rec yds
  - RB1 rush yds
  - RB1 attempts

  Do the same for college from CFBD box scores: favorite cover vs over, by spread bucket (≤7, 7-14, 14-21, 21+).
  - Hypothesis to measure, **not sourced**: in big college spreads, favorite cover and over are positively correlated (the favorite does the scoring); in NFL small spreads, the dog cover vs under link is weak.
  - Output: copula parameters per bucket.
- **Use 1: price "two singles vs a parlay" correctly.** When two correlated singles are both +EV, Kelly sizing should shrink the combined stake by the correlation. The repo runs the moneyline, spread, totals and props tracks independently, and they can stack on one game, for example a night_west cover plus a props under on the same team.
  - Test: from the paper-bet ledgers, compute realised pairwise correlation of results for bets on the same game. Report the effective number of independent bets. This matters for p-values in pass/fail grading (game-clustered SE are already used in some studies; check that every track grades with clustering).
- **Use 2: the only plausible SGP edge is books pricing a pair as independent when the true ρ is large.** Examples: QB pass yds over + his WR1 rec yds over, or game over + both QBs over. That edge cannot be tested without SGP quotes. Logging a few SGP quotes by hand from allowed books against the copula fair would be a cheap feasibility check. Expect the 15-25% hold to swamp most of it.

### Gaps
- No peer-reviewed or measured study of SGP mispricing at US books was found.
- The Odds API has no SGP prices, so no backtest is possible.

## Q3. Distribution modelling: drive/possession simulation, pace, garbage time, overtime

### Takeaway
Drive-level Markov or simulation models are an established academic approach, but the repo's evidence says the key-number margin model already captures what matters for main lines. Making it total-aware added nothing out of sample. Simulation's value is in **regime changes with little history** (the 2025 regular-season OT rule, the 2024-26 kickoff rules, college clock rules), where historical margin frequencies are stale and a mechanistic model can re-derive push and key-number rates before the market learns them.

### Cited Findings
- **Published simulation work [B].** Existing work includes NFLSimulatoR (simulation-based decision-making in the NFL from play-by-play) and "A Markov model of football: using stochastic processes to model a football drive". These are titles only; neither was read in full. — [arXiv 2102.01846](https://arxiv.org/pdf/2102.01846); [Markov model of football](https://lida.sport-iat.de/twm/Record/4025215?lng=en)
- **Pace and efficiency don't beat the totals close [R].** "The market prices totals well, and pace and efficiency add nothing beyond the close." Model disagreement predicts line moves, but is "not a standalone edge". — [output/research/totals.md](/home/claude/nfl-win-probability/output/research/totals.md)
- **Key numbers vs total are mostly era [R].** w3 = 3.33 at total 38 vs 2.96 at total 50 in 1999-2019, but most of that is era (low totals = 1999-2011). There is no 2020-25 gain. — [output/research/ml_spread_consistency.md](/home/claude/nfl-win-probability/output/research/ml_spread_consistency.md)
- **The market already discounts garbage time [R].** It already discounts garbage-time points and return/defensive TDs; "if anything it discounts garbage time too much". — [output/research/market_ratings.md](/home/claude/nfl-win-probability/output/research/market_ratings.md)
- **2025 regular-season OT rule [A].** Approved April 1, 2025: both teams get a possession in regular-season OT, matching the playoff format. After both possessions it is sudden death. OT stays 10 minutes, and ties remain possible. — [NBC New York / AP, Apr 1 2025](https://www.nbcnewyork.com/news/sports/nfl/nfl-owners-rule-changes-overtime-rules-kickoff-touchback-tush-push-2025/6207820/)

### Inferences
- **OT regime test (D1).** Under the old rule, a walk-off TD on the first possession ended the game, so OT margins concentrated at 3 and 6. Under the 2025 rule, more OT games should see both teams score, which means more OT points and a different margin mix.
  - From nflverse pbp, take every OT game from 2017-24 (old rule) and 2025-26 (new rule). Tabulate:
    - final margins in OT games
    - OT points
    - tie rate
    - share of OT games ending on the first possession
  - Feed the shift into margins.py/margin_total.py as a regime adjustment. This affects push probabilities at ±3 and ±6/7 and the over probability on totals near game-total lines.
  - **Sample is tiny:** OT games are a small share of games, so expect only about one to two dozen per season. Use a drive simulator calibrated on regulation drives to price the OT rule instead of the empirical margin mix.
  - The **betting relevance is small** (OT probability × the change in push rate), so treat this as pricing hygiene, not an edge.
- **Drive simulator as a regime tool (D2).**
  - Model the number of drives per team from pace (seconds per play, run rate, neutral-situation pass rate) and per-drive outcome probabilities (TD/FG/punt/TO) from team EPA (shift(1)).
  - Make start field position depend on kickoff regime: 2023 TB 73% → 2024 64.3% → 2025 20.7%, average start own 28.8 → 30.1 → 30.7 (see Q4).
  - Simulate 10k games per matchup.
  - Validate the exact-margin log loss vs margin_total.py on 2020-25. Expect no gain on main lines.
  - Then use it **only** to re-weight key-number frequencies when a rule change hits, and test whether early-season totals and pushes after the change were mispriced.

### Gaps
- Neither simulation paper was read, so their reported accuracy against market lines is unknown.
- No source quantified how the 2025 OT rule changed OT points or margin distributions. That has to be computed from nflverse.

## Q4. Rule changes that shift numbers and whether markets adjusted

### Takeaway
The NFL dynamic kickoff (2024, made permanent in 2025 with the touchback moved to the 35) raised scoring efficiency about 12% per drive from 2023 to 2025, and the average posted total rose with it (43.1 → 44.8). The market adjusted in level, but nobody has published whether it lagged in the first weeks. That lag is testable with nflverse schedule lines. College has had the 2023 clock rule (about 4.5 fewer plays per game, scoring lowest since 2009) and the 2024 two-minute warning and helmet communications. The 2026 NFL changes are small (onside kick declarable at any time, a kickoff loophole closed, receiving-team alignment).

### Cited Findings
- **NFL kickoff and scoring by season [B]:**

  | | 2023 | 2024 | 2025 |
  |---|---|---|---|
  | Touchback rate | 73% | 64.3% | 20.7% |
  | Average start | own 28.8 | own 30.1 | own 30.7 |
  | Points per drive | 1.88 | 2.07 | 2.11 |
  | Scoring % per possession | 35.5% | 38% | 38.8% |
  | Average O/U | 43.1 | 44.6 | 44.8 |

  The article reports **no over/under records**. — [VSiN, Adam Burke, Aug 30 2026](https://vsin.com/nfl/how-have-the-new-nfl-kickoff-rules-impacted-scoring-and-field-position/)
- **2024 vs 2023 [B].** +622 total points, +85 TDs, +26 made FGs on +55 attempts. 2025 vs 2024: +55 points, +8 TDs, -6 made FGs. Yards per kick return: 27.6 (2024), 25.9 (2025). — [VSiN](https://vsin.com/nfl/how-have-the-new-nfl-kickoff-rules-impacted-scoring-and-field-position/)
- **2025 rule package [A].** The dynamic kickoff was made permanent and the touchback moved from the 30 to the 35. The league projected the return rate to rise from 32.8% (2024) to 60-70%. Regular-season OT was changed to give both teams a possession. — [NBC New York / AP, Apr 1 2025](https://www.nbcnewyork.com/news/sports/nfl/nfl-owners-rule-changes-overtime-rules-kickoff-touchback-tush-push-2025/6207820/)
- **2026 NFL changes [A].** Approved March 31, 2026:
  - Onside kick may be declared at any time, not only when trailing (surprise onside and stacking still banned).
  - Kicking out of bounds from the 50 no longer gains anything (closes a Dallas 2025 loophole that pinned KC at its own 25).
  - Receiving-team setup-zone alignment changed so most players stay deeper.
  - Two officiating items.
  - No touchback or OT change is mentioned. — [CBS Sports](https://www.cbssports.com/nfl/news/nfl-rule-changes-2026)
- **College 2023 clock rule [B].** The first-down clock rule cut about 4.5 plays per game (projection was 7, per NCAA rules editor Steve Shaw). 2023 FBS scoring was 27.78 points per team, the lowest since 2009, and a third straight annual decline. Total offense was 385.7 yds (lowest since 2010) and TDs 3.47 per game (lowest since 2008). Average game time fell about 5 minutes to 3:23. Other causes cited: defensive-line talent and schemes. — [CBS Sports, Dennis Dodd, Feb 16 2024](https://www.cbssports.com/college-football/news/college-football-offenses-continue-declining-in-key-categories-resulting-in-lowest-scoring-season-since-2009)
- **College 2024 [A].** The NCAA approved a two-minute warning and helmet communications for college football starting in the 2024 season (headline; article body not read). — [CBS Sports](https://www.cbssports.com/college-football/news/ncaa-approves-2-minute-warning-in-college-football-games-helmet-communications-beginning-in-2024-season)

### Inferences
- **R-lag test, NFL.** Dataset: nflverse schedules (total_line, spread_line, results) 2018-2026, plus Odds API close 2020-25.
  - For each season, compute mean(actual total − closing total) and the over rate by week block (Weeks 1-4, 5-9, 10-18).
  - Hypothesis: in 2024 Weeks 1-4 (the first dynamic-kickoff season) and 2025 Weeks 1-4 (touchback to the 35), actual minus close > 0 and the over rate > 52.4%, fading by midseason.
  - The average O/U rose 1.5 points in 2024 while points per drive rose about 10%. A rough check is drives × Δ(pts per drive) ≈ 22 drives × 0.19 ≈ +4 points, against only +1.5 in the average posted total. That suggests the market under-adjusted at first, *if* drives per game were stable. **This is an inference to verify, not a finding.**
  - If confirmed, pre-register a generic rule-change rule for future seasons: when the league changes kickoff/OT/clock rules, bet overs or unders on the first 4 weeks only if a simulator-derived Δtotal exceeds 1.5 points vs the opener. Grade on CLV.
  - For 2026, the onside and loophole changes are likely too small. The simulator can say so.
- **R-lag test, college.** Dataset: CFBD lines 2018-2025.
  - Run the same week-block test on 2023 (the clock rule, expected unders early) and 2024 (the two-minute warning adds a stoppage before half and full time, a plausible small over effect concentrated in Q2/Q4).
  - Use CFBD line scores to test Q2+Q4 scoring changes for 2023 vs 2024 at equal closing totals.
  - The repo's college factor screen found all 10 situational factors fail, but rule-regime lag was not among them.
- **12-team CFP (from 2024).** It changes late-season incentives: more teams stay "alive" into November. Test idea with CFBD rankings and lines 2018-25: ATS in November games where one team is still in CFP contention and the other is eliminated, before and after 2024.
  - **The 2024-25 sample is small** (two seasons), so treat it as a 2026 pre-registration candidate, not a backtest claim.

### Gaps
- No source reported over/under records or week-by-week line lag after the 2024/2025 kickoff changes. The NFL Ops piece "Dynamic kickoff making leaguewide impact on offense through Week 4" appeared in search but was not fetched.
- No numeric evaluation was found for the 2024 college two-minute-warning or helmet-communication effects on scoring.
- No source confirmed the drives-per-game trend for 2023-25. It is needed for the under-adjustment arithmetic above.

## Q5. Bayesian/hierarchical player models: QB value, injury replacement, backup QBs, OL

### Takeaway
The repo has already built the quant-standard tools: a hierarchical/empirical-Bayes QB model and a RAPM player-value model. Neither beats the market or the production model. The market prices absences by snap share, and any edge is in *timing* (knowing the final status before the report), not in better valuation. This angle is essentially closed. The remaining work is the news-timing pipeline that already exists (news_audit, qb_availability).

### Cited Findings
- **Bayes QB [R].** The posterior is 0.94-0.95 correlated with the production EWMA, with no walk-forward log-loss gain overall or on QB-change / <300-dropback games. The Bayes-minus-production revision does predict the early-week-to-close line move on QB-change games (dev t = 3.7, holdout t = 2.3, p = 0.011), but it misses the Bonferroni bar, and the bettable version has -0.3% CLV in the holdout. — [output/research/bayes_qb.md](/home/claude/nfl-win-probability/output/research/bayes_qb.md)
- **RAPM [R].** Ratings from pbp participation 2016-25 "carry real team-strength signal", but per-player absence value does not improve the model, does not predict the ATS residual, and does not explain line moves. "The market moves on how much a missing player plays (snap share)… The early-week edge on injury news is real only if you already know the final status." — [output/research/player_ratings.md](/home/claude/nfl-win-probability/output/research/player_ratings.md)

### Inferences
- Further valuation modelling of backup QBs or OL is unlikely to pay. If any modelling remains, it is a **sit-probability** model (practice participation Wed/Thu/Fri → P(out)), priced against the early-week line move. That is what qb_availability.py and news_audit already approach.
- College is the exception worth one test. CFBD has no snap counts, and college injury reporting was weak before conference availability reports. Repo note: "QB news mostly priced by Sunday" (cfb round2). So any college QB edge would have to come before Sunday, which is impractical. Low priority.

### Gaps
- No external source with measured backup-QB or OL pricing errors was searched (the web budget was spent on rule changes and correlation).

## Q6. Market-implied ratings, sharp consensus as priors, derivatives at soft books, and academic inefficiencies 2015-2025

### Takeaway
What works in this repo's own evidence is market-structural: soft-book prices vs a sharp no-vig reference (college shop rules, receptions same-point shopping, cross-book derivative shopping near the close). Model-driven and situational angles (Walters factors, market over-reaction, holdover/early-season) are mostly priced or too noisy. The academic literature on football market biases (holdover bias, bowl vs regular season, censoring bias, divisional rivals, preseason bias) exists. Its post-2015 relevance is doubtful given the repo's own replication that Walters-style factors decayed from a slope of +0.96 (2003-22) to about 0 (2020-22).

### Cited Findings
- **Market week-to-week updating is close to efficient [R].** Lines move 0.05-0.06 points per point of result surprise, against an ideal weight of 0.03-0.05. All three frozen over-reaction rules failed the 2023-25 holdout. — [output/research/market_ratings.md](/home/claude/nfl-win-probability/output/research/market_ratings.md)
- **Walters factors [R].** The residual slope was +0.96 ± 0.23 in 2003-22, +1.25 in 2003-12, +0.73 in 2013-22, −0.19 ± 0.54 in 2020-22 and +0.34 ± 0.48 out of sample (2023-25). The one replicating lead is night-game time zones: the western team beats the close by +2.20 ± 1.05 points (n = 113, 56.5% cover), and that is being paper-tracked. — [output/research/walters_factors.md](/home/claude/nfl-win-probability/output/research/walters_factors.md)
- **Academic literature [titles only; papers not read]:**
  - "Holdover Bias in the College Football Betting Market" (Springer, 2019 per the DOI; journal not verified) — [Springer](https://link.springer.com/article/10.1007/s11293-019-09611-y)
  - "Inefficient pricing from holdover bias in NFL point spread markets" — [ResearchGate](https://www.researchgate.net/publication/263080729_Inefficient_pricing_from_holdover_bias_in_NFL_point_spread_markets)
  - Cox, Schwartz, Van Ness & Van Ness (2021), "The Predictive Power of College Football Spreads: Regular Season Versus Bowl Games", Journal of Sports Economics — [SAGE](https://journals.sagepub.com/doi/abs/10.1177/1527002520975837)
  - "Next game reaction to mispriced betting lines in college football", Applied Economics Letters 28(12) — [T&F](https://www.tandfonline.com/doi/abs/10.1080/13504851.2020.1795063)
  - "Market Efficiency and Censoring Bias in College Football Gambling" — [ResearchGate](https://www.researchgate.net/publication/367224660_Market_Efficiency_and_Censoring_Bias_in_College_Football_Gambling)
  - "NFL betting market efficiency, divisional rivals, and profitable strategies", Studies in Economics and Finance — [DOI](https://www.doi.org/10.1108/SEF-11-2018-0354)
  - "Preseason bias in the NFL and…" — [UWF](https://ircommons.uwf.edu/esploro/outputs/journalArticle/Preseason-bias-in-the-NFL-and/99380090299306600)

### Inferences
- **Holdover / preseason bias test (H1)** (check first whether screen7/s6_preseason.json already covers it).
  - Mechanism: early in the season, bettors and books anchor on last season's results.
  - Test on CFBD 2015-25 and nflverse 2015-25. Regress Weeks 1-4 ATS residual (margin − closing spread) on the prior-season win% gap, controlling for the preseason market rating (season_sim's market ratings) and returning production (CFBD).
  - Pre-register: bet against the team whose prior-season win% exceeds its preseason-rating rank by ≥ 2 deciles in Weeks 1-3. Pass bar: > 52.4% at p < 0.05 on 2020-25 *and* positive CLV vs the Tuesday opener.
  - Given the decay seen in Walters factors, the prior is low.
- **"Next-game reaction to mispriced lines" (college).** This overlaps the repo's market_ratings over-reaction study, which was done for NFL only. A college replication is cheap: CFBD lines and results 2015-25, with next-game ATS regressed on last-game surprise (margin − close). It is worth one run because college markets are thinner.
- **FCS-vs-FBS and Week 0-2 college lines.** No source found. The test is cheap with CFBD:
  - closing-line error variance and ATS bias in FBS-vs-FCS games and Weeks 0-2, vs the rest
  - the cfb shop rules' CLV in those segments, where soft-book vs Pinnacle gaps may be widest

  This is the most promising extension of what already works (college price rules).
- **Sharp-consensus pricing of derivatives at soft books.** It is already implemented for NFL (derivatives_v2 R1, ml v2-v4) and college (cfb_shop). The new extension to test is **college derivatives** (1H spreads/totals, team totals) priced from Pinnacle full-game lines with the college key-number distribution (cfb_dist.json). Check cfb/deriv_screen.md first; a derivative screen exists, so confirm whether it covered 1H college at Arizona books.

### Gaps
- None of the academic papers listed could be read (rate limiting and bot checks), so their effect sizes, sample years and post-2015 robustness are unknown. They should not be cited for numbers.
- No sourced evidence was found on totals bias (over/under), home-underdog or big-college-favorite inefficiencies post-2015. The repo's own studies (totals.md, walters_factors.md, cfb factor_screen) are the best available evidence and mostly say "priced".
