# Model-Based Edges in College Football (FBS/FCS) Betting Markets

Context for the report writer: the project's own CFB results were combined-model MAE 12.92 vs closing line 12.49 (2019-21), and frozen rules on the 2022-25 holdout gave 47.4% (opener disagreement >=5 pts), 52.5% (close-residual GBM top quintile) and 52.1% (totals). The findings below are for comparing against that.

Research limitations: ESPN Insider pages (SP+ and FPI ATS reviews) are mostly paywalled or JS-rendered, X/Twitter is blocked by robots.txt, and The Prediction Tracker's results page could not be fetched. Several key records are therefore only partly verified. Each such case is flagged.

---

## 1. Tracked ATS/CLV records of public CFB models

### Takeaway
The best-documented public CFB model, SP+, has run at roughly 51-53.5% ATS per season against both midweek and closing lines (2018-2021). That is about break-even to slightly profitable at -110 (52.4%), not a large edge. Most other "winning" CFB model records are self-reported, short (1-2 seasons), or cover only selected picks. No public CFB model I found has a verified, multi-year, all-games record clearly above 54-55% against the close.

### Cited Findings
**SP+ (Bill Connelly, ESPN), verified from the free part of an ESPN article:**
- 2018: "52.8% against the midweek spread, 51.2% against the Caesars closing line." — [ESPN, "Numbers game: How SP+ and other ratings can give you a betting edge" (2021)](https://www.espn.com/college-football/insider/story/_/id/32250032/numbers-game-how-sp+-other-ratings-give-betting-edge)
- 2019: "53.4% midweek and 53.2% against the closing line." — [ESPN Numbers game](https://www.espn.com/college-football/insider/story/_/id/32250032/numbers-game-how-sp+-other-ratings-give-betting-edge)
- 2020: "only 51.2% midweek but 52.8% against the closing line." — [ESPN Numbers game](https://www.espn.com/college-football/insider/story/_/id/32250032/numbers-game-how-sp+-other-ratings-give-betting-edge)
- Early 2021: "52.8% against the midweek spread and 53.4% against the close." — [ESPN Numbers game](https://www.espn.com/college-football/insider/story/_/id/32250032/numbers-game-how-sp+-other-ratings-give-betting-edge)
- When SP+ and the spread disagreed by less than 1 point, SP+ went 11-16-1 (41%), which Connelly called noise that "should even out over time." His advice is to keep a consistent model and "wait to pounce" on large discrepancies. — [ESPN Numbers game](https://www.espn.com/college-football/insider/story/_/id/32250032/numbers-game-how-sp+-other-ratings-give-betting-edge)
- SP+ vs FPI, per Connelly (Sept 2021 tweet, seen as a search-result title only; not fetched): "SP+ is generally better against the spread, FPI is generally better in terms of absolute error." — [X/@ESPN_BillC](https://x.com/ESPN_BillC/status/1440001228017111044)
- 2024 early season, per Connelly (tweet title, Sept 2024; not fetched): SP+ was "54% against the early spread so far this year" after 4 weeks. — [X/@ESPN_BillC](https://x.com/ESPN_BillC/status/1837857641151778937)
- 2017 (S&P+), one week, per Connelly (tweet title; not fetched): "Really annoyed by S&P+ going 56% ATS, btw. It was at like 64% before bombing the late night games." This is a weekly figure, not a season record, and it shows how much single-week results swing. — [X/@ESPN_BillC](https://x.com/ESPN_BillC/status/927236697900900356)

**CFBD community model (gradient-boosted trees, published on the CFBD blog):**
- Uses LightGBM/NGBoost with 714 features (home/away, four-year recruiting, returning production, pregame spread, rankings, talent, aggregated stats such as havoc and QB hurries). Trained on 1997-2018 (17,662 games), tested 2019 onward, and run live for 131 games from 26 Sept 2020.
- Results: spread RMSE 15.72 in development vs 16.77 in production; ATS 64/127 (50%) on all games. Its "high-confidence" picks went 24/35 (68.5%), a tiny and selected sample. — [CFBD blog: Using machine learning to predict game outcomes and spreads](https://blog.collegefootballdata.com/predicting-spreads-gbdt/)

**Orb Analytics (independent Substack modeler; self-reported):**
- 2023 spreads: 60-42-7 (58.2%), +12.5u. 2024 spreads (weeks 2-17): 45-36 (55.6%), +5.0u. Two-year combined: 108.5-81.5 (57.1%). These are selected picks, not all games, and no CLV is reported.
- In 2024 the record swung from 60.1% in weeks 2-9, to 52.3% after a slide in weeks 10-13, then recovered by weeks 14-17. 50 of 81 picks were favorites, and underdog picks hit 51.6%.
- The model is an ensemble of 3 algorithms. — [Orb Analytics 2024 regular season recap](https://orbanalytics.substack.com/p/2024-regular-season-recap)

**Academic ensemble of public systems (Coleman 2025, Journal of Sports Analytics):**
- Combined 29 rating systems (from The Prediction Tracker) across 5,925 games, 2016-24, into a 5-system metamodel. It "achieves strong results vis-à-vis the opening, midweek, and closing betting lines." The edge was statistically significant against **opening** lines in the validation and test samples. — [Coleman, "A predictive metamodel for college football" (abstract record)](https://lida.sport-iat.de/dfb/Record/4094978?lng=en); [DOI](https://doi.org/10.1177/22150218251365223)

**Market vs ratings on bowl games (The Power Rank / Ed Feng):**
- On bowl games, the closing line picked 61.5% of straight-up winners (208-130). The preseason Coaches Poll picked 59.9% and the preseason AP Poll 58.8%. Over 339 bowl games (2005-2014), margin-of-victory plus schedule-adjusted ratings came close to the market but did not beat it. No ATS records were published there. — [The Power Rank: Essential guide to predictive CFB rankings](https://thepowerrank.com/guide-cfb-rankings/)

**The Prediction Tracker:**
- It publishes opening, midweek and updated lines next to about 30 computer systems (Sagarin, FPI, Sonny Moore, Congrove, Billingsley, Kambour, etc.), plus a "Season Totals" results page. I could not retrieve the results tables. — [Prediction Tracker NCAA predictions](https://www.thepredictiontracker.com/predncaa.html)

### Inferences
- SP+ is the benchmark: about 51-53.5% ATS per season against both midweek and closing lines, on all games. The project's 52.5% (close-residual GBM top quintile) is in the same range as the most respected public CFB model. That suggests it is near the practical ceiling for public-data rating models, not that it has failed outright.
- Connelly's SP+ midweek and closing ATS rates are similar, while Coleman's metamodel was significant only against **openers**. Together these suggest that whatever edge public ratings have is mostly captured by line movement before kickoff.
- The project's 47.4% when disagreeing with the opener by 5+ points runs against the hope that large disagreements are where the value is. In CFB, large model-vs-market gaps often reflect information the model lacks (QB status, opt-outs, portal turnover), so they may signal adverse selection rather than edge.
- The project's MAE gap (12.92 vs close 12.49) matches Connelly's comment that FPI beats SP+ on absolute error while SP+ wins ATS. Lower MAE and better ATS rate are different goals, so a model can trail the close on MAE and still be useful as one input among several.

### Gaps
- Season-by-season SP+ ATS for 2021 (full), 2022, 2023, 2024 and 2025: Connelly publishes these in ESPN Insider columns that I could not fetch.
- FPI, Massey-Peabody CFB, Sagarin, FEI (Fremeau), Parker Fleming, Kelley Ford and Action Network/PFF model records: I found no verifiable multi-year ATS or CLV figures. Parker Fleming's "Bless Your Chart" Substack post #111 contained only team ATS records, not a model record ([link](https://blessyourchart.substack.com/p/111-ats-summary)).
- The Prediction Tracker's per-system ATS% and MAE tables (including the line's own MAE) could not be retrieved, so there is no verified multi-system comparison.
- Full Coleman (2025) metamodel numbers (exact ATS % against open, midweek and close) are paywalled. A PDF on ResearchGate returned HTTP 429.

---

## 2. Which inputs matter and may be underpriced

### Takeaway
Peer-reviewed evidence of mispricing in CFB lines covers: travel and time-zone effects (especially late season, and underdogs with a 1-hour time-zone disadvantage), favorites and "hot hand" overpricing in older data, and slow adjustment to structural rule changes for totals. Each of these anomalies tends to decay once published. Practitioners stress fast-moving news (QB changes, bowl opt-outs, portal departures), which public ratings miss and the market prices fast.

### Cited Findings
- **Travel and time zones:** In Coleman (2017, Journal of Sports Economics), "the betting market is found to be an inaccurate and inefficient processor of travel effects," particularly in late-season games. The clearest inefficiency was underdogs facing a 1-hour time-zone disadvantage. Variables tested were time-zone change, distance and direction of travel, temperature, elevation and aridity. — [Coleman, "Team Travel Effects and the College Football Betting Market"](https://scholars.unf.edu/en/publications/team-travel-effects-and-the-college-football-betting-market/)
- **Favorites and hot hand:** Sinkey & Logan studied more than 11,000 games, 1985-2003. They found "favorites are systematically overpriced"; betting $1000 on underdogs in prominent games returned $1,116.99 in expectation before the bookmaker's cut. Books inflate lines for teams on cover streaks: "beating the betting line adds over 2 points to the betting line" the following week. They found no evidence that inefficiency was concentrated early or late in the season. — [Sinkey & Logan, "Betting Markets and Market Efficiency: Evidence from College Football" (AEA 2010)](https://www.aeaweb.org/conference/2010/retrieve.php?pdfid=406)
- **Coaching (reputation) bias that decayed:** Across the 15 highest-paid coaches (1,937 games over 10 years), betting on them won 53.95%. After five years the market became efficient, falling to 51.6%. Urban Meyer was the only coach significantly profitable over the full decade (61.79%). No totals strategy beat the market. — [Farinella & Moffett, Business Quest 2016](https://www.westga.edu/~bquest/2016/football2016.pdf)
- **Totals after structural changes:** In 2006 the NCAA timing rule change cut about 14 plays per game. Overs hit only 47.23% that season (vs 50.26% in 2003-05), and average scoring fell from 52.33 to 47.15 points. By November the over rate had recovered to 49.2% as the market learned. — [Paulson, Business Quest 2008](https://www.westga.edu/~bquest/2008/football08.pdf)
- **Bowl opt-outs and steam:** In bowl games since 2005, following line moves of 3+ points went 53-36 ATS (59.5%) against the opener but only 47-40-2 (54%) against the close. Since 2017 the record is 25-13. Action Network attributes big bowl moves to opt-outs, COVID absences and portal news reaching insiders before it is public (e.g., Virginia Tech-Maryland and Braxton Burmeister). — [Action Network, bowl line moves](https://www.actionnetwork.com/ncaaf/bowl-game-betting-tips-has-following-big-line-moves-worked)
- **QB news (practitioner claim, no data):** "NFL markets would've priced that QB injury more aggressively. College lines often don't." The same source says totals of 59.5-63.5 matter in CFB because of tempo teams. — [Business of College Sports](https://businessofcollegesports.com/sports-betting/how-college-football-betting-differs-from-the-nfl-and-which-numbers-matter/) (low-evidence general article)
- **Early-season inputs (PFF):** PFF's 2021 early-season piece says running-game proficiency and pass-rush pressure are associated with early-season covers. It notes talent turnover, extreme spreads in mismatches and thin data as the challenges, but publishes no ATS numbers. — [PFF 2021](https://www.pff.com/news/college-football-betting-2021-examining-early-season-cfb-betting-lines)
- **Preseason priors / wisdom of crowds:** Preseason human polls keep "surprising predictive power" through bowl season (58.8-59.9% straight-up on bowls vs 61.5% for the closing line). — [The Power Rank](https://thepowerrank.com/guide-cfb-rankings/)

### Inferences
- The published anomalies are mostly old (pre-2010 data) and tend to disappear once documented (the coaching bias did; the 2006 totals bias corrected within the season). Building them in as frozen rules matches the project's holdout failure. CFB edges seem to be temporary and need re-estimating each season.
- The bowl data shows that information moving the market is worth about 5.5 percentage points between open and close (59.5% vs 54%). The edge sits in being early on news, not in the rating model itself. For a ratings-only model, this argues for betting openers only where the model's disagreement is **not** explainable by pending news, or for building news-flag features (QB status, opt-outs, portal).
- Travel and time-zone effects (Coleman 2017) are among the few peer-reviewed, reasonably recent findings worth testing as features. That especially applies to late-season games with an underdog crossing one time zone.

### Gaps
- No quantitative public studies found on: how well the market prices transfer-portal roster turnover; coordinator changes; garbage-time filtering as a betting edge; altitude by venue; lookahead or letdown spots; or FCS-opponent games. Practitioner claims exist, but I found no verifiable ATS data.
- I did not retrieve how SP+ and FPI weight returning production, recruiting and recent performance in their preseason priors (ESPN pages could not be fetched).
- No CFB-specific study of weather in totals pricing was found within this budget.

---

## 3. Market structure: when and where CFB lines are softest

### Takeaway
Practitioners agree that CFB lines are softer and stay soft longer than NFL lines, with lower limits. The evidence that edges show up against openers but shrink against closers (Coleman 2025 metamodel; bowl steam 59.5% vs opener but 54% vs close) points to openers and early-week numbers as where public-ratings value exists. I found no rigorous public data comparing small conferences, midweek MACtion or limits by book.

### Cited Findings
- CFB markets "still breathe": NFL lines are mostly efficient by midweek, while "soft lines exist longer in the college market." Lower limits in college mean it stays softer through the week. Spreads of 30 points are routine. — [Business of College Sports](https://businessofcollegesports.com/sports-betting/how-college-football-betting-differs-from-the-nfl-and-which-numbers-matter/) (no named sources or numbers)
- Coleman's 29-system metamodel was statistically significant against **opening** lines in validation and test (2016-24), with "strong results" against midweek and closing lines but significance reported only for openers. — [Coleman 2025 abstract](https://lida.sport-iat.de/dfb/Record/4094978?lng=en)
- Bowl line moves of 3+ points: 59.5% ATS against the opener vs 54% against the close (2005-2021). — [Action Network](https://www.actionnetwork.com/ncaaf/bowl-game-betting-tips-has-following-big-line-moves-worked)
- SP+ results against midweek and closing lines are close together (e.g., 2019: 53.4% vs 53.2%; 2020: 51.2% vs 52.8%). — [ESPN Numbers game](https://www.espn.com/college-football/insider/story/_/id/32250032/numbers-game-how-sp+-other-ratings-give-betting-edge)
- Orb Analytics' 2024 picks were strongest in weeks 2-9 (60.1%) and weakest in weeks 10-13. — [Orb Analytics](https://orbanalytics.substack.com/p/2024-regular-season-recap)
- Connelly reported 54% ATS in weeks 1-4 of 2024 (tweet title; single season). — [X/@ESPN_BillC](https://x.com/ESPN_BillC/status/1837857641151778937)

### Inferences
- Taken together, the weak evidence points to openers in weeks 2-9 as the most likely place for a public-ratings edge. Two caveats from the project's own holdout: the opener edge is in **selection**, not raw size of disagreement, and the closing line is the right benchmark for checking whether a bet was good.
- The 2006 totals episode and Sinkey-Logan suggest totals and less-watched games were historically softer. Modern evidence is anecdotal.

### Gaps
- No verifiable data found on: limits by book for CFB, Sunday opener limits, which books lag, line-movement speed by conference, MACtion and midweek game efficiency, or G5 vs P4 closing-line accuracy. ESPN's bookmaker-insights article ([link](https://africa.espn.com/chalk/story/_/id/32348377/nfl-college-football-line-moves-early-action-bookmaker-insights)) and the 2026 ESPN storylines piece ([link](https://www.espn.com/espn/betting/story/_/id/49734086/college-football-betting-storylines-2026-season-ohio-state-texas-notre-dame-miami)) could not be rendered.
- No academic study found that directly tests whether weeks 1-4 lines are less accurate than later lines in recent seasons. Sinkey & Logan (1985-2003) found no week-of-season concentration.

---

## 4. CFB-specific methodology

### Takeaway
Methods with documented value: opponent-adjusted, margin-aware ratings (raw records or schedule strength alone are near coin-flips); team-strength adjustment when estimating home field; ensembling several independent rating systems; and preseason priors from recruiting and returning production that fade as the season goes on. Home field in FBS is now around 2.4 points, not the folk 3-3.5, and has been falling by about 1 point per decade.

### Cited Findings
- **Home field:** 2023 home advantage was FBS 2.39 points (95% credible interval 1.89-2.88), FCS 2.49 (1.99-2.96), D-II 2.40 and D-III 2.37. FBS home field is declining at about -0.097 points per year over 2004-2023 ("roughly a point drop per decade"). Earlier FBS estimates were 3.65 (2008) and 3.85 (2010). — [arXiv 2401.16392, "A comprehensive survey of the home advantage in American football"](https://arxiv.org/pdf/2401.16392)
- Unadjusted home-field estimates are inflated because "top programs pay worse opponents to visit and get blown out." Model-based vs raw estimates differed by more than 3 points in some states. Travel distance explained little of home-field variation (R² = 0.016). Replay review (adopted from 2006) and better travel are proposed as reasons for the decline. — [arXiv 2401.16392](https://arxiv.org/pdf/2401.16392)
- **Margin of victory:** Ratings using margin plus schedule adjustment approach the market on bowl games, while "win percentage is hardly better than flipping a coin." EPA/play-by-play systems (FPI, S&P+, Massey-Peabody) are described as the modern improvement. — [The Power Rank guide](https://thepowerrank.com/guide-cfb-rankings/)
- **Ensembling:** Coleman (2025) found a 5-system ensemble drawn from 29 public systems beat opening lines significantly over 2016-24. That supports blending several independent ratings over relying on one. — [Coleman 2025](https://lida.sport-iat.de/dfb/Record/4094978?lng=en)
- **Priors and feature-heavy machine learning:** The CFBD gradient-boosted model used four-year recruiting, returning production and talent composite. Its production RMSE (16.77) was worse than in development (15.72), and it went 50% ATS on all games. Adding many features did not produce an edge. — [CFBD blog](https://blog.collegefootballdata.com/predicting-spreads-gbdt/)
- **Totals pace:** Rule changes that alter the number of plays (2006 clock rules) moved scoring by about 5 points per game, and the market took most of a season to adjust. — [Paulson 2008](https://www.westga.edu/~bquest/2008/football08.pdf)

### Inferences
- Using a flat 3+ point home field in CFB would bias predictions toward home teams by about 0.5-1 point in recent seasons. The project should check the HFA its models use, and estimate it from strength-adjusted data with a downward time trend.
- The CFBD 714-feature model and the project's GBM both show that adding features to a gradient-boosting model does not beat the close. The documented successes are ensembles of independent rating systems compared against openers, plus information timing (news, opt-outs).
- Sparse schedules (130+ FBS teams with few links between them) argue for strong priors (talent, returning production) that fade over weeks 1-6, and for keeping FCS games but down-weighting them. No source quantified the best decay rate.

### Gaps
- No public source found that quantifies venue-specific home field (e.g., altitude at Air Force, Wyoming, Colorado or Utah, or night games at specific stadiums) against the betting line.
- I could not retrieve how Connelly, FPI or FEI handle FCS opponents, garbage-time cutoffs, or how preseason weights decay by week.
- No public documentation found of Massey-Peabody CFB methodology or results.
