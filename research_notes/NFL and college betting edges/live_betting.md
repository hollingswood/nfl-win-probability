# Live (in-game) betting edges in NFL and college football

Researched 2026-10-04. Scope: NFL and NCAA FBS, US legal market 2023-2026. Caveat up front: rigorous, peer-reviewed evidence on in-play efficiency comes almost entirely from **soccer and basketball on Betfair**. I found **no peer-reviewed study using US-book or Betfair live odds for NFL or college football**. Football-specific claims below are practitioner opinion or inference unless marked otherwise.

## 1. Academic and practitioner evidence on in-play market efficiency (overreaction/underreaction, favorite-longshot bias, momentum)

### Takeaway
The best evidence finds **systematic but small, short-lived** in-play mispricings. Markets overweight early, weak signals (an early score) and underweight late, strong ones. After a surprise event they misprice for seconds to minutes. All of it comes from exchange data (Betfair) in soccer, basketball and multi-sport samples, not from US sportsbooks on American football. "Fade the overreaction to an early score" has academic support in principle. Whether it survives US live-book vig and limits is untested.

### Cited Findings
- **Augenblick, Lazarus & Thaler, "Overinference from Weak Signals and Underinference from Strong Signals"** (Quarterly Journal of Economics, per the Berkeley Haas write-up; working paper first circulated Sept 2021, version June 2024). Data: experiments with 500 basketball fans, plus **over 5 million Betfair transactions across ~260,000 games** in basketball, soccer, football, ice hockey and one more sport, plus CBOE options quotes 1996-2018. Finding: early baskets got **~60% more weight than warranted** and fourth-quarter baskets were **underweighted by ~33%** (experimental numbers). In betting data, price "movement is generally higher than uncertainty reduction early on… and lower toward the end of the event." — [Berkeley Haas magazine, Spring 2025](https://newsroom.haas.berkeley.edu/magazine/spring-2025/the-over-under-conundrum/); [arXiv 2109.09871](https://arxiv.org/abs/2109.09871v4); [arXiv HTML v5](https://arxiv.org/html/2109.09871v5)
  - Caveat: the arXiv text I retrieved emphasizes an NBA subsample. It does not clearly state that "football" in the Haas summary means American football rather than soccer, and I found no numbers specific to American football. I found no profit-after-commission test in the excerpts.
- **Angelini, De Angelis & Singleton, "Informational efficiency and behaviour within in-play prediction markets"** (International Journal of Forecasting; Reading discussion paper 2019/20). Data: Betfair odds every 10 seconds for 1,004 EPL matches, 2009-2014. Findings:
  - Semi-strong inefficiency after the **first goal**, strongest ~20 s after the goal and persisting at least 5 min.
  - Mispricing is larger after **surprise** news (an underdog scoring late), and the market underestimated the underdog's chances.
  - A **reverse favorite-longshot bias** on the exchange (β = -0.26).
  - Reported gross ROI of 35-70% betting 20 s after the first goal, falling to 12-25% at 5 min. These are gross figures on small samples.
  - [Reading working paper PDF](https://www.reading.ac.uk/web/files/economics/emdp201920.pdf); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207021000996)
- **Croxson & Reade, "Information and efficiency: goal arrival in soccer betting"** (Economic Journal): the classic test of how exchange prices absorb a goal. Its headline finding is usually summarized as prices updating swiftly and fully at goal arrival, i.e. semi-strong efficiency. I did not fetch the full text, so treat this summary as unverified. — [Birmingham repository](https://research.birmingham.ac.uk/en/publications/information-and-efficiency-goal-arrival-in-soccer-betting/); [Reading CentAUR](https://centaur.reading.ac.uk/34884)
- Related soccer evidence on overreaction and surprise: "The Role of Surprise: Understanding Overreaction and Underreaction to Unanticipated Events using In-Play Soccer Betting Market" (Choi & Hui) and "Profiting from overreaction in soccer betting odds" (JQAS 2020). — [ResearchGate (Choi & Hui)](https://www.researchgate.net/publication/256013000_The_Role_of_Surprise_Understanding_Overreaction_and_Underreaction_to_Unanticipated_Events_using_In-Play_Soccer_Betting_Market); [De Gruyter JQAS](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2019-0009/html?lang=en)
- **Kalshi in-game calibration** (Moshrefi, Princeton, July 2026): ~23 million Kalshi moneyline trades in NBA, MLB and NHL, Mar-May 2026. Prices are well calibrated mid-game but **diverge sharply in the final ~10 minutes before settlement**, which the author attributes to insurance demand from traders holding losing positions. Cross-game parlays are overpriced by ~3% per extra leg. No NFL or CFB data, and no profit figures. — [arXiv 2607.14430](https://arxiv.org/pdf/2607.14430)
- Practitioner view (Kambi, 2018): Kambi's head of in-play said an edge by a skilled player won't last long because "the line or price would move quite quickly if we felt someone had an edge on us." — [Kambi / Sports Handle interview, Aug 15 2018](https://kambistage.expre.co.uk/?p=4940)

### Inferences
- The academic pattern (early overreaction, late underreaction) points to two live-football angles:
  - **Fade early-score moves**, e.g. take the team that conceded a first-quarter TD, if the live line moves more than a calibrated WP model says it should.
  - **Back the leader late**, if late-game prices under-update.
- The Kalshi near-expiry miscalibration suggests the last minutes of decided games are a separate pocket. That is where "insurance" buyers overpay for comebacks.
- The exchange evidence shows a *reverse* favorite-longshot bias. US books usually show the normal FLB pre-game. Do not assume the in-play direction; measure it.
- Football scores are discrete and frequent. So any "early overreaction" should be checked against a model that knows each score's true WP impact given the spread (nflfastR `vegas_wp`), not against intuition.

### Gaps
- No NFL or CFB in-play efficiency study using actual live prices (US books, Betfair NFL markets, Kalshi or Polymarket football) was found. This is the main hole and the project's own data could partly fill it.
- No study found on in-play reaction to **turnovers or QB injuries** in American football specifically.
- "Momentum" myths in NFL betting: no football-specific peer-reviewed source was retrieved this session.

## 2. How books price live football, and their known weaknesses

### Takeaway
US live football prices come from official-data-fed models: Genius Sports is the NFL's exclusive official data distributor, and Sportradar, Kambi and in-house models compete. Traders intervene in fast situations. Books protect themselves with suspensions, bet delays and much higher live margins. The sourced material on specific weaknesses is thin and mostly vendor or opinion content.

### Cited Findings
- **Genius Sports is the NFL's exclusive official data and "Watch & Bet" distribution partner.** The deal was extended July 2023, and again in a multi-year deal reported June 12, 2025. This is the official low-latency play-by-play source books use for live NFL markets. — [Business Wire, Jul 6 2023](https://www.businesswire.com/news/home/20230706824603/en/National-Football-League-Extends-Strategic-Partnership-with-Genius-Sports-as-Exclusive-Official-NFL-Data-and-Watch-Bet-Distribution-Partner); [Covers, Jun 12 2025](https://www.covers.com/industry/nfl-genius-sports-announce-extension-with-multi-year-deal-june-12-2025); [Legal Sports Report](https://www.legalsportsreport.com/123307/genius-extends-nfl-betting-data-deal-at-least-five-years/)
- Kambi (2018) described a hybrid of heavy modeling and automation plus **~300 traders**. Human judgment matters most in fast situations such as **hurry-up offense**. Competitors without expert staff "will likely suspend their markets." A common competitor failure is not updating core lines to the current game state. Noy said in-play betting already dominates "most regulated markets." This is old (2018) and comes from a vendor. — [Kambi interview](https://kambistage.expre.co.uk/?p=4940)
- Genius Sports' public "Trader's View" content covers soccer only: Monte Carlo simulation pricing, automation tools and BetVision. It describes nothing NFL-specific. — [Genius Sports Trader's View](https://www.geniussports.com/content-hub/traders-view-in-play-football/)
- **Live margins are much higher than pre-game.** An industry and affiliate blog (low reliability, unsourced figures) claims NFL pre-match overround of **2-4% vs 6-9% in-play**, and 7-12% vs 4-6% across sports generally. Its stated reasons are micro-markets, recency bias, no odds-comparison tools, time pressure, operator latency advantage and auto-suspension. — [Track360, 2026](https://www.track360.io/blog/in-play-betting-margins-operator-economics-2026)

### Inferences
- With roughly 6-9% overround on live NFL markets (if accurate), a model needs about a **3-4.5 percentage-point probability edge** per side just to break even. Pre-game needs about 1-2. The Angelini-type post-event mispricings in soccer were large in gross ROI but lasted seconds. A retail bettor behind a bet delay (Section 4) would mostly miss them.
- Plausible weak spots (inference, not sourced): lower-tier FBS games with less trader attention; quick in-game QB injuries before the feed and traders react; weather that changes mid-game; and **CFB overtime rules**. Since 2021, CFB OT goes to alternating 2-point plays from the 3rd period. Whether automated models handle this correctly was not verified.
- Hourly snapshots cannot measure these effects (see Section 7).

### Gaps
- No primary source found on Sportradar's or Genius's NFL/CFB live model design, suspension rules or official-data latency in seconds.
- No sourced evidence on CFB live-market quality vs NFL, or on which conferences are covered with official data.

## 3. Halftime lines

### Takeaway
I found **no rigorous study** of NFL or CFB halftime-line mispricing. The search returned only odds pages and generic strategy guides.

### Cited Findings
- Searches surfaced only odds listings (BetQL first- and second-half lines, BetUS) and a generic strategy guide. — [BetQL NCAAF 2H lines](https://betql.co/ncaaf/odds/second-half-lines); [OddsIndex halftime guide](https://oddsindex.com/guides/halftime-betting-strategy)

### Inferences
- Halftime is structurally the most bettable in-play moment for a non-latency bettor. The market is static for ~15-20 minutes, and a model can price it carefully from the pre-game spread, score, possession to start the 2nd half (known from the opening coin toss) and 1st-half efficiency. The project can backtest this. nflverse PBP has the halftime state, and `vegas_wp` at the start of Q3 is a model-implied price.
- Halftime lines are also where the Augenblick et al. "early overreaction" should show up: a 1st-half score swing over-moving the 2H line or the live full-game moneyline.

### Gaps
- No historical halftime-line dataset was located in this session. The Odds API historical feed may not carry 2H markets, and that was not verified.

## 4. Latency, courtsiding and bet delays (US)

### Takeaway
Courtsiding is in a **legal gray area in the US**: there is no federal ban, and state bills have stalled. In practice, licensed books defeat it with official low-latency data, bet-acceptance delays, suspensions and account limits. A TV-watching bettor is structurally behind the book's feed.

### Cited Findings
- No federal US ban exists. A New York bill to give stadium officials enforcement power died in the Assembly. Michigan discussed anti-courtsiding provisions. One legal expert cites First Amendment news-gathering protection. As of the article (updated Sept 16 2026), there were **no reports of courtsiding at US-licensed books**, though "nobody is really looking all that hard." Australia made the first courtsiding arrest in 2014. — [BettingUSA, courtsiding](https://www.bettingusa.com/sports/courtsiding/)
- An affiliate blog lists "operator latency advantage over retail bettors" and "auto-suspension policies protecting against stale prices" as structural features of in-play markets. — [Track360](https://www.track360.io/blog/in-play-betting-margins-operator-economics-2026)
- UK Gambling Commission guidance on in-running betting covers operator controls such as delays and suspensions. — [UKGC in-play guidance](https://www.gamblingcommission.gov.uk/licensees-and-businesses/print/in-play-or-in-running-betting)

### Inferences
- For this project, latency arbitrage is neither practical nor desirable: it raises legal and terms-of-service risk, gets limited fast, and our data is hourly. Any live edge must come from **better pricing of a state that persists** (halftime, timeouts, between possessions), not from speed.

### Gaps
- No sourced figure for US sportsbook live-bet acceptance delays (commonly said to be several seconds) or for how far TV lags the Genius feed.

## 5. Live win-probability models (nflfastR, ESPN, cfbfastR, postgame win expectancy) vs markets

### Takeaway
nflverse ships open WP models, including a spread-aware `vegas_wp`, through the `fastrmodels` package. I found no published head-to-head of these models against live market prices. That comparison is the core experiment the project would need to run.

### Cited Findings
- `fastrmodels` is the CRAN/r-universe package holding the trained models used by nflfastR, including win probability. — [fastrmodels on r-universe](https://nflverse.r-universe.dev/api/packages/fastrmodels); [CRAN reference manual](https://cran.case.edu/web/packages/fastrmodels/refman/fastrmodels.html)
- A practitioner Substack walks through building a WP model. Not fetched. — [nfosignal, "Building win probability"](https://nfosignal.substack.com/p/building-win-probability-a-practical)
- A survey of sports-analytics "big ideas" covers win-probability modeling. Not fetched in detail. — [arXiv 2301.04001](https://arxiv.org/pdf/2301.04001)

### Inferences
- `vegas_wp` anchors on the closing spread and decays its weight with game time. That makes it a close cousin of what a book's automated model does. A live edge would need features the generic WP model lacks: team-specific offense/defense strength beyond the spread, QB-in-game status, weather, kicker quality and tempo. Disagreements between `vegas_wp` and live prices are the place to look, but they must be scored with log loss and Brier score on held-out seasons (CLAUDE.md rules 2 and 4).
- ESPN WP, cfbfastR WP and Bill Connelly's postgame win expectancy were not compared with markets in any source I found.

### Gaps
- No accuracy or calibration figures for nflfastR WP vs live odds. No source found on cfbfastR's WP calibration. Bill Connelly's postgame win expectancy is a retrospective "deserved to win" metric, not a live price; its predictive value for next-game lines was not researched here.

## 6. Documented specific strategies

### Takeaway
Named strategies (live unders after fast starts, live dogs after an early deficit, CFB tempo totals) circulate widely in practitioner content, but I found **no backtested evidence with live prices**. They are best treated as hypotheses consistent with the "early overreaction" literature.

### Cited Findings
- Practitioner live-betting guides (DeucesCracked NFL in-play, a Substack on "gridiron live betting", RotoWire's Super Bowl LVIII live-total strategy, TheSpread on live timing and liquidity) exist but are opinion pieces. None were fetched or verified. — [DeucesCracked NFL in-play](https://www.deucescracked.com/blog/live-betting-strategy-in-play-nfl-markets); [RotoWire SB LVIII live totals](https://www.rotowire.com/football/article/super-bowl-58-live-betting-strategies-and-best-bets-for-the-overunder-79606); [TheSpread](https://www.thespread.com/?p=537075); [Fiddle's Picks Substack](https://fiddlespicks.substack.com/p/gridiron-gold-live-betting-strategies)
- Academic support for the *direction* of "fade the early move": [Augenblick, Lazarus & Thaler](https://arxiv.org/abs/2109.09871v4). Academic support for mispricing after a *surprise* score: [Angelini et al.](https://www.reading.ac.uk/web/files/economics/emdp201920.pdf)

### Inferences
- Testable hypotheses with project data:
  1. **Early-TD overreaction.** After a first-quarter TD by the pre-game underdog, the live favorite's moneyline is too long relative to `vegas_wp`.
  2. **Live unders after a fast start.** A live total that extrapolates early scoring pace too far. Points-per-drive regresses, and the effect may be stronger in CFB with tempo offenses.
  3. **Late-game underreaction.** Leaders with two or more scores in Q4 are priced too cheaply. This mirrors the Kalshi near-expiry and Augenblick late-underreaction results.
- Each needs a large in-game price sample with exact timestamps matched to PBP `time_of_day`.

### Gaps
- No backtest of any named live NFL or CFB strategy with real live prices was found. Kicking-game and weather in-game strategies: nothing sourced.

## 7. Data sources for historical live odds, and costs

### Takeaway
Live odds history at play-level resolution is scarce and expensive. This project's **hourly Odds API snapshots are far too coarse** to study live overreaction. Prices change every play, and mispricings in the literature decay within seconds to minutes. The cheapest dense source is likely **Kalshi or Polymarket trade history**, which is public via API, for NFL and CFB games from 2025 on, or **Betfair historical data** for NFL.

### Cited Findings
- Betfair publishes historical exchange data with a tooling workbook on GitHub. Pricing tiers for that data were not retrieved. — [Betfair historic-data-workbook](https://github.com/betfair/historic-data-workbook); [Betfair plans listing (apis.io)](https://apis.io/plans/betfair/betfair-plans-pricing/)
- Commercial historical-odds vendors exist, including SportsDataIO's "Vault" historical odds and SportsGameOdds' Betfair feed; their in-play coverage and cost were not verified. — [SportsDataIO historical odds](https://sportsdata.io/historical-odds); [SportsGameOdds Betfair API](https://sportsgameodds.com/betfair-odds-api/)
- Kalshi lists NFL markets, including in-game trading, in 2026, and cross-venue price gaps were reported during a live NFL game (Chiefs-Raiders, Oct 3 2026). Kalshi microstructure was studied in [GWU "Makers or Takers" (2026)](https://www2.gwu.edu/~forcpgm/2026-001.pdf). — [SI Kalshi NFL review](https://www.si.com/prediction-markets/reviews/kalshi-nfl); [Goal.com prediction-market update](https://www.goal.com/en-us/betting/news/prediction-market-update-10-03-chiefs-raiders-exposes-huge-trading-gap-across-kalshi-polymarket-and-novig/A%3Abltcfebd6f0da642c5e)
- One open-source CFB project grades picks against Kalshi and Polymarket prices: `cassandra` PR #73, "betting.py --book kalshi|polymarket". — [GitHub NathanDeMaria/cassandra PR #73](https://github.com/NathanDeMaria/cassandra/pull/73)
- Local project context, from repo code: `src/nflpred/odds_history.py` downloads historical snapshots only at pre-game moments (Tue/Fri windows and 75 min before kickoff). Player props come from the per-event historical endpoint, with data from May 2023 and **10 credits per market per region**. The Odds API historical page could not be fetched this session, so its snapshot interval and in-play coverage are unverified.

### Inferences
- What the project already has: hourly `history/odds_*.json.gz` snapshots that sometimes land mid-game, and nflverse PBP with `time_of_day` and `vegas_wp`. With these it can run only a **coarse pilot**. Match each in-game snapshot to the PBP state at that wall-clock time. Compare the de-vigged live moneyline with `vegas_wp`, then check which is better calibrated by log loss and Brier score. A few hundred game-states from 2022-25 would show whether there is a systematic gap, e.g. live prices over-moving after early scores. The sample is too sparse and too unevenly timed to design a bettable strategy.
- To go further, the project needs dense in-game prices: Kalshi or Polymarket trade history (free, but prediction-market fees and liquidity differ from books), Betfair NFL historical data (paid, thin US-game liquidity), or a paid vendor.
- An exploitable edge must clear roughly 6-9% live overround (if the Track360 figure is right), bet delays, and fast limits on live winners. Halftime and between-quarter states are the only realistic targets for a non-latency bettor.

### Gaps
- Odds API historical: whether snapshots include live prices for started games, the snapshot interval, and the credit cost. The page was not retrievable this session; check the docs directly.
- Betfair historical data pricing for NFL, and OddsJam, Bet365 or SportsDataIO in-play archive coverage and cost: not verified.
- Kalshi NFL and CFB in-game calibration studies: none found; the only Kalshi in-game study covers NBA, MLB and NHL.
