# Model Inputs and Data Sources Cited as an Edge Over the NFL Betting Market

Scope note: research run 2026-10-03. Tool budget was limited, so several vendor pricing pages (SIS, TruMedia, Sportradar, NGS licensing, Unabated) and two key papers (Gregory-Smith 2021; the ESPN win-rate study) could not be fetched in full. Those items are listed under Gaps rather than filled from memory. Search results on injuries and betting splits were mostly sportsbook and affiliate marketing pages. They are flagged as such and are not treated as evidence.

## 1. Player availability valuation (non-QB injuries, cluster injuries, practice-report designations)

### Takeaway
I found no primary, peer-reviewed evidence in this pass that NFL closing lines misprice non-QB injuries or injury-report designations. The material that surfaced was sportsbook and affiliate content giving rule-of-thumb point values. The most promising academic source is a 2021 *Economic Inquiry* paper that uses injuries to measure positional value, but it could not be retrieved (403). The free nflverse injury data, which includes practice participation, already goes back to 2009, so the extra signal would have to come from *how* the data is used (positional point values, cluster interactions) rather than from a new source.

### Cited Findings
- nflverse `load_injuries` gives weekly official injury report data from **2009 onward**, "collected from an API for weekly injury report data." That history covers backtesting of report designations and practice participation. — [nflreadr reference manual (CRAN PDF)](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)
- nflverse `load_depth_charts` gives week-level depth charts from **2001 onward**. These can identify the starter and the backup who replaces him, which a "replacement-level drop-off" valuation needs. — [nflreadr reference manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)
- nflverse `load_contracts` (OverTheCap) gives active and non-active contracts. Cap hit or APY is sometimes used as a cheap prior for player value when no grades are available. — [nflreadr reference manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)
- Gregory-Smith (2021), "Wages and Labor Productivity: Evidence from Injuries in the National Football League," *Economic Inquiry*, uses injuries to estimate player productivity by position. The full text returned 403, so its estimates and any point-spread findings are **not verified here**. — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/ecin.12960)
- An analogous NBA study ("Player absence and betting lines in the NBA," *Finance Research Letters*, vol. 13, 2015) examines whether lines adjust correctly for absent players. It shows the method exists in the literature, but it is NBA-only and its findings were not retrieved. — [RePEc/IDEAS](https://ideas.repec.org:443/a/eee/finlet/v13y2015icp130-136.html)
- Keefer & Kniesner, IZA DP No. 16289, is an NFL injury/labor paper that came up in search. Its relevance to point spreads was not confirmed. — [IZA](https://docs.iza.org/dp16289.pdf)
- Non-evidentiary (marketing or affiliate) pages make claims about how injury news moves NFL lines. They give no data or methodology and should not be cited as evidence: [Packernet (2026)](https://www.packernet.com/blog/2026/03/10/how-nfl-injury-reports-impact-betting-lines-and-market-movement/), [NFL Betting Strategies](https://nflbettingstrategies.com/articles/nfl-injury-impact-betting-lines/), [Crown Wagers](https://www.crownwagers.com/injuries-and-the-nfl-point-spread/), [OddsShopper](https://www.oddsshopper.com/articles/betting-101/betting-nfl-injury-news), [BetNow](https://www.betnow.eu/nfl/injury-report-impact-on-odds/), [MyBookie](https://www.mybookie.ag/sports-betting-guide/determining-wagering-impact-of-injured-players/).

### Inferences
- The project already uses nflverse injuries, so any gain from the injury inputs would most likely come from (a) **position-specific replacement drop-offs**, measured as starter vs backup grade or snap-weighted EPA contribution, (b) **interaction terms for clustered absences** (for example 2+ OL starters out, or 2+ CBs out), and (c) **timing**: modeling the Wednesday, Thursday and Friday practice trajectory before the market settles. None of these is shown to beat the close in the sources found here.
- Closing lines absorb final inactive news roughly 90 minutes before kickoff. A practice-report model can therefore only add value against *earlier* lines (open or midweek), not the close. If the project's pre-registered tests are against the close, injury features will likely look priced in by construction.

### Gaps
- Point values by position from a verified quantitative source (OL, EDGE, CB, WR1). These were not obtained, and the Gregory-Smith paper could not be accessed.
- Studies of whether "Questionable" or "Limited" practice status predicts playing or performance, and whether markets over- or under-react to it. None found in this pass.
- Peer-reviewed evidence on whether markets correctly price cluster injuries. None found.

## 2. Offensive line vs pass rush matchups (PBWR/PRWR, pressure, time to throw, PFF grades)

### Takeaway
ESPN's Pass Block Win Rate (PBWR) and Pass Rush Win Rate (PRWR), built from NGS tracking, are the most-cited public "trench" metrics. ESPN published an analysis titled "Pass blocking matters more than pass rushing, and we can prove it," but its contents could not be retrieved, so its specific stability and predictive numbers are unverified. I found no evidence in this pass that trench metrics beat the spread beyond EPA.

### Cited Findings
- ESPN Analytics explainer on how PBWR and PRWR are built from player-tracking data. — [ESPN explainer](https://africa.espn.com/nfl/story/_/id/24892208/creating-better-nfl-pass-blocking-pass-rushing-stats-analytics-explainer-faq-how-work)
- ESPN Analytics article "Pass blocking matters more than pass rushing, and we can prove it." The title states the thesis, but the fetch returned empty and the numbers were **not verified**. — [ESPN](https://www.espn.co.uk/nfl/story/_/id/26888038/pass-blocking-matters-more-pass-rushing-prove-it)
- An arXiv paper (2305.10262) came up for the pass-rush/pass-block query. Its content and findings were not retrieved. — [arXiv](https://arxiv.org/pdf/2305.10262v1)
- PFF now sells programmatic access to grades and Premium Stats, which include pass-block and pass-rush grades, through its API (see Section 6). — [PFF Developer API](https://developer.pff.com/)

### Inferences
- PBWR and PRWR are published as team and player tables on ESPN and have no documented public API. Historical backfill would require scraping, and the history is likely short (ESPN launched the metric around 2018).
- Free proxies for pressure exist in nflverse: NGS passing summaries (time to throw) and the PFR advanced stats loader (2018+, which includes pressures and blitzes per the nflverse documentation). These may capture much of the same signal, so the incremental value of paid trench data is uncertain.

### Gaps
- Year-over-year stability coefficients for PBWR/PRWR, and evidence of out-of-sample or ATS value beyond EPA. Not obtained.
- Any study showing PFF OL/DL grades predict spreads or results beyond the market. None found.

## 3. Coaching and scheme changes, QB changes, early-season priors and early-season market inefficiency

### Takeaway
This is the best-documented area of the scan. Peer-reviewed work finds that NFL lines anchor on preseason information. Fodor, Patterson & Shank (2025) report that betting in line with preseason Super Bowl odds was **significantly profitable in weeks 2–8** across 2003–2023, and that the effect is statistically significant all season. Earlier work found a week-2 bias driven by overweighting prior-season results. This supports using an independent, roster-change-aware preseason prior that is updated in-season at a different rate from the market.

### Cited Findings
- Fodor, Patterson & Shank (2025), "Anchoring bias in the NFL gambling market," *Economics Letters* vol. 250. Sample: 2003–2023, N=5,088 games. Bettors keep favoring teams with strong preseason Super Bowl odds well into the season. Sportsbooks "continue to incorporate pre-season odds into their closing lines throughout the entire season." Betting aligned with preseason odds was significantly profitable in weeks 2–8. The effect was strongest in week 1 and declined but stayed significant (p<0.01). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0165176525001259)
- Davis, Fodor, McElfresh & Krieger (2015), "Exploiting Week 2 Bias in the NFL Betting Markets," *Journal of Prediction Markets* 9(1):53–67. Week 2 lines show inefficiencies from how bettors weigh week-1 results against prior-season information, which the authors say creates "the opportunity for profitable betting strategies." — [RePEc/IDEAS](https://ideas.repec.org/a/buc/jpredm/v9y2015i1p53-67.html)
- "Inefficient pricing from holdover bias in NFL point spread markets" is a related paper on prior-season information carrying into new-season lines. Details were not retrieved. — [ResearchGate](https://www.researchgate.net/publication/263080729_Inefficient_pricing_from_holdover_bias_in_NFL_point_spread_markets)

### Inferences
- The Fodor et al. result suggests the closing line *itself* carries a preseason-odds component that persists. Preseason Super Bowl or win-total odds are cheap and historically archived, so they could be tested directly as a feature, or as a residual against the model's own in-season rating, in weeks 2–8. It is notable that the profitable direction reported is *with* preseason odds, which implies the market under-weights them later. The exact direction should be confirmed from the full text.
- Coordinator and play-caller changes are not in nflverse, so they would have to be hand-curated. I found no study quantifying their market effect.

### Gaps
- Quantified market reaction to mid-season QB changes or play-caller changes. Not found.
- Whether the Fodor et al. effect survives after 2023 or against sharp-book closes (vs a consensus close). Not stated in the abstract.

## 4. Special teams, garbage-time filtering, opponent adjustment, explosive-play and turnover regression

### Takeaway
No primary evidence was gathered in this pass on whether these adjustments beat the market. These are standard parts of public models such as DVOA and EPA-based ratings, so they are likely already reflected in the line. The free FTN charting data in nflverse adds play-level context (for example play-action, RPO, blitz counts, screens, drops) that can feed these adjustments at no cost.

### Cited Findings
- FTN charting data comes free through nflverse from **2022 onward**, under CC-BY-SA 4.0 with attribution to FTN Data via nflverse. Plays are "charted within 48 hours following each game," so the data is usable in-season. — [nflreadr reference manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)

### Inferences
- Win-probability-filtered EPA, opponent adjustment and turnover/fumble-luck regression can all be built from the play-by-play the project already has. They are low-cost to test, but the prior that the market already prices them is strong.

### Gaps
- Any study showing ATS value from kicker quality, garbage-time filtering or turnover-luck regression after 2015. None retrieved.

## 5. Market-derived features (betting splits, line movement, sharp vs consensus disagreement, limits)

### Takeaway
I found no peer-reviewed or methodologically transparent evidence that public ticket and money percentages predict NFL ATS results or closing-line moves. The pages that surfaced are affiliate or sportsbook guides. Splits are cheaply available (Action Network PRO costs about $10–$120 depending on term), but their provenance is a limited set of books, and their historical archives are not clearly available for backtesting.

### Cited Findings
- Action Network in-app subscription prices as of 2026-04-24: PRO Weekly $9.99, PRO Monthly $19.99 or $24.99, PRO 3-Month $59.99, PRO Annual $119.99, EDGE Annual $99.99, EDGE $29.99. — [App Pricing Lab](https://apppricinglab.com/iap/apple/1083677479)
- Action Network markets an "NFL PRO Report" built on "sharp action" and betting systems. This is vendor marketing, not evidence. — [Action Network](https://www.actionnetwork.com/nfl/nfl-pro-report-sharp-action-betting-systems-model-projections-expert-picks)
- A third-party Action Network scraper (Apify) advertises odds, splits and props extraction, the only programmatic route surfaced. Terms of service may restrict it. — [Apify](https://apify.com/parseforge/action-network-scraper)
- Free public-splits pages exist, all affiliate guides with no evidence of predictive value: [OddsAssist](https://oddsassist.com/sports-betting/nfl/nfl-public-consensus-betting-percentages/), [SportsBettingDime](https://www.sportsbettingdime.com/nfl/public-betting-trends/), [WiseGuyTeam](https://wiseguyteam.com/nfl-betting-splits), [Cleatz](https://cleatz.com/public-betting/nfl/), [BettingUSA](https://www.bettingusa.com/sports/nfl/public-betting/), [TheSpread](https://www.thespread.com/betting-guides/nfl-public-betting-guide-read-splits-percentages/).
- Weak-form efficiency literature on sports betting markets (Robbins, ECU) addresses whether past price information predicts outcomes. Its specific NFL findings were not retrieved. — [ECU PDF](https://myweb.ecu.edu/robbinst/PDFs/Weak%20Form%20Efficiency%20in%20Sports%20Betting%20Markets.pdf)

### Inferences
- Split data is a sample from individual books (for example, data reported by partner books), not a market-wide figure. Any backtest needs archived, timestamped splits, which none of the sources above document.
- If the project's evaluation target is the closing line, then "line movement history" is mostly information the close already contains. Its plausible use is predicting the close from the open, which matters only if bets can be placed early.

### Gaps
- Unabated pricing and features (odds screen, "Unabated Line"). The search returned no Unabated pages.
- Sports Insights / Bet Labs current pricing and archive depth. Not found.
- Any study of ticket% vs money% predicting ATS or CLV. None found.

## 6. Data sources beyond nflverse: contents, cost, access, history, overlap with nflverse

### Takeaway
PFF is the only premium source with clear, public 2026 consumer pricing and an API. API access requires the new **PFF Pro** tier at $199.99/yr, which was listed as "coming soon" as of September 2026. FTN charting (2022+) and FTN participation (2023+) are already free in nflverse, but **participation from 2023 onward is published only after the postseason**, so it cannot be used in-season. SIS, TruMedia, Sportradar and licensed NGS have no public pricing in what I retrieved and appear to be enterprise or B2B contracts.

### Cited Findings
**PFF**
- PFF pricing (announced 2026-09-01): PFF+ Monthly $9.99/mo; PFF+ Annual $99.99/yr, which includes Premium Stats; **PFF Pro $199.99/yr ("Coming Soon")**, which includes "Upgraded Premium Stats Pro" and "Programmatic data access for modeling (API/scrape)." API and scraping are exclusive to Pro. — [PFF: A new chapter of PFF pricing](https://www.pff.com/news/a-new-chapter-of-pff-pricing)
- The PFF Developer API serves "PFF player grades and Premium Stats Pro data." The "docs are public, but every data endpoint needs a PFF Pro subscription." Access is through the Restish CLI with PFF Pro login. Usage is governed by PFF Terms of Service. Rate limits and historical depth are not stated on the page. — [PFF Developer API](https://developer.pff.com/)
- Related PFF pages: [Subscribe](https://www.pff.com/subscribe), [Terms of Use](https://www.pff.com/terms), [Support: does API access come with a subscription?](https://profootballfocussupport.zendesk.com/hc/en-us/articles/32094827302163-Does-API-access-come-with-a-subscription), [Premium Stats overview](https://www.pff.com/news/pro-pff-premium-stats-highlighting-all-of-pffs-advanced-metrics-and-grades).

**nflverse loaders (free; what is already covered)**
- `load_ftn_charting`: 2022+, FTN manual charting subset, CC-BY-SA 4.0, charted within 48 hours of each game. — [nflreadr manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)
- `load_participation`: 2016+. "Participation data prior to 2023 is from NFL NGS. Participation data from 2023 onwards is courtesy of FTN and is provided **after all post-season games are completed**." This makes it a backtest-only feature from 2023 onward, unless it is replaced with a paid real-time source. — [nflreadr manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)
- `load_nextgen_stats`: 2016+, player-level weekly NGS summaries, only for players above minimum attempt thresholds. — [nflreadr manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)
- `load_pfr_advstats`: 2018+, PFR advanced stats. `load_snap_counts`: 2012+ (PFR). `load_espn_qbr`: 2006+ (season or week). `load_injuries`: 2009+. `load_depth_charts`: 2001+. `load_contracts`: OverTheCap. — [nflreadr manual](https://cran.nics.utk.edu/cran/web/packages/nflreadr/nflreadr.pdf)

**Betting-market data**
- Action Network PRO: $9.99/week to $119.99/year (2026-04). App and web UI only. No official API was surfaced, and third-party scrapers exist. — [App Pricing Lab](https://apppricinglab.com/iap/apple/1083677479); [Apify scraper](https://apify.com/parseforge/action-network-scraper)

**Official league data distribution**
- A news item reports accusations that Genius Sports (the NFL's official data distributor) price-gouges for NFL data. This indicates official real-time data is expensive and B2B-only. It is a secondary source, and details were not verified. — [GGB News](https://ggbnews.com/article/genius-sports-accused-of-price-gouging-for-nfl-data)

**Sports Info Solutions**
- Search returned only CB Insights company profiles and no SIS DataHub pricing. — [CB Insights: SIS](https://www.cbinsights.com/company/sports-info-solutions); [PFF vs SIS](https://www.cbinsights.com/compare/pro-football-focus-vs-sports-info-solutions)

### Inferences
- **Cheapest new signal:** FTN charting (free and in-season since 2022) is probably under-used. Its fields, such as play-action, RPO, screen, blitz and "is_qb_out_of_pocket"-type flags (exact field list not retrieved here), can be added to team-efficiency splits at no cost. However, it gives only about 4 seasons of history (2022–2025) for validation.
- **Cheapest paid upgrade:** PFF Pro at $199.99/yr is the lowest-cost way to get player grades programmatically. That makes positional injury valuation (starter vs replacement grade) and OL/DL matchup features feasible. The tier was still "coming soon" as of September 2026, and historical depth through the API is undocumented.
- Personnel and participation features from 2023 onward cannot be used in-season with free data. A model trained on them for 2023–2025 would leak information that is unavailable at prediction time.

### Gaps
- SIS DataHub pricing, contents and history depth. No public pricing found.
- TruMedia pricing (enterprise, not found). Sportradar and Stats Perform NFL API pricing (not found). Licensed NGS tracking data, beyond the public Big Data Bowl samples (not researched in this pass).
- Unabated subscription pricing and features. Search did not return Unabated pages.
- Sports Insights / Bet Labs current pricing and whether historical splits archives are sold.
- PFF API historical depth (how many seasons of grades) and rate limits.
- Documented evidence that any of these paid sources adds predictive value *against closing lines*. None was found. All value claims found were vendor marketing.
