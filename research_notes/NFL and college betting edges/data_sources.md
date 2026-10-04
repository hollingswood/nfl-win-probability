# Data Sources and Information Edges for an Automated NFL + FBS Betting System (2025-2026)

Research date: 2026-10-04. Prices and policies are as reported by the cited pages on their stated dates; several provider comparisons come from competitor marketing blogs (flagged below) and should be confirmed on the vendor's own pricing page before purchase.

## Odds data providers, betting splits, and whether sharp-money/splits signals have value

### Takeaway
Paid odds feeds split into three tiers: cheap developer APIs (The Odds API, OddsPapi, SportsGameOdds at $0-$249/mo), sales-gated mid-market feeds (OddsJam, OpticOdds, Unabated Enterprise from $3,000/mo, SportsDataIO), and enterprise feeds (Sportradar at $30K+/mo). Betting-split "systems" are widely marketed (VSiN/DraftKings/Circa splits), but the vendors publishing them give no backtested records. The peer-reviewed evidence (Sports Insights data, offshore books, 2004-2010) shows books do not balance action, yet contrarian or streak-fading strategies were not significantly profitable in the NBA or the NFL hot-hand tests.

### Cited Findings
**Odds APIs (pricing, coverage, history)**
- OddsPapi: free tier 250 requests/month, claims "370 bookmakers (verified live, May 2026)" including Pinnacle, Singbet, SBOBet, Betfair Exchange and crypto books; historical odds via `/v4/historical-odds` on the free tier; WebSocket only on Pro. (Source is OddsPapi's own comparison blog, dated May 26, 2026 and updated Sep 25, 2026, so treat its competitor claims as biased.) — [OddsPapi blog](https://oddspapi.io/blog/?p=2926)
- The same OddsPapi blog describes The Odds API as having a 500-credit/month free tier and "~40 bookmakers", says it has no Pinnacle, and says historical data is a paid add-on. — [OddsPapi blog](https://oddspapi.io/blog/?p=2926). This conflicts with the user's existing setup, which already pulls Pinnacle through The Odds API (EU region). The Pinnacle claim looks wrong or region-dependent.
- SportsGameOdds: free tier has 9 books with a 10-minute delay; Rookie is $99/mo (77 books) and Pro is $249/mo; ~85 books including Pinnacle; WebSocket is enterprise-only. — [OddsPapi blog](https://oddspapi.io/blog/?p=2926)
- OddsJam: API pricing is "contact us"; consumer subscriptions are $99-$499/mo; 100+ books including Pinnacle; historical data only on the enterprise tier. — [OddsPapi blog](https://oddspapi.io/blog/?p=2926)
- OpticOdds: sales-gated, ~200 operators, "sub-800ms streaming latency", historical data enterprise-only. — [OddsPapi blog](https://oddspapi.io/blog/?p=2926)
- Sportradar: "Contracts in the $30K+/mo range", with official league partnerships. — [OddsPapi blog](https://oddspapi.io/blog/?p=2926)
- Unabated API: the Developer/Deeplink tier is free or low-cost with unpublished terms; Enterprise starts at $3,000/month and includes the full "Unabated Line" (a vig-free consensus built from sharp market-making books, blended per sport). It covers NFL and college football, US regulated books, offshore books, DFS (PrizePicks, Underdog), Kalshi/Polymarket and sweepstakes books, with WebSocket support. The review was last updated Jan 2024, so pricing may be stale. — [sportsapis.dev](https://sportsapis.dev/apis/unabated)
- SportsDataIO: NFL and NCAA Football; "pre-match, in-play, historical and closing lines" with opening and closing lines and all intermediate price changes, line-movement timestamps, injuries and lineups; free trial; pricing not published. — [SportsDataIO](https://sportsdata.io/betting-data)
- Third-party scrapers on Apify exist for Action Network (odds, splits, props) and VegasInsider (odds, betting splits). Using them carries ToS risk (see the legal section). — [Apify Action Network scraper](https://apify.com/parseforge/action-network-scraper); [Apify VegasInsider scraper](https://apify.com/parseforge/vegasinsider-scraper)

**Prediction markets (Kalshi / Polymarket)**
- OddsPapi aggregates Kalshi and Polymarket NFL prices with full snapshot history on its free tier. One Polymarket fixture returned 1,821 snapshots totalling 23.68 MB, against 0.18 MB for a sportsbook fixture. — [OddsPapi Kalshi/Polymarket blog, Aug 24, 2026](https://oddspapi.io/blog/?p=3660)
- Depth is thin. A Polymarket Week 1 NFL moneyline quoted a 0.98% margin with only $7.22 at top of book and $454.62 across three levels. Kalshi had $107.59 at top of book and $8,990.74 across the ladder. In August 2026 Kalshi listed only Week 1 games, and neither venue offered player props. — [OddsPapi blog](https://oddspapi.io/blog/?p=3660)
- Unabated's feed includes Kalshi and Polymarket prices. — [sportsapis.dev](https://sportsapis.dev/apis/unabated)

**Betting splits (tickets vs handle) and evidence of value**
- VSiN (Josh Appelbaum, Aug 30, 2026) recommends two rules: fade teams with 65%+ of DraftKings spread *bets* (back the side with 35% or less), and back the side at Circa when % of handle exceeds % of bets by 10+ points (example: 57% bets / 78% dollars). The article gives no historical records, win rates, ROI or sample period. — [VSiN](https://vsin.com/nfl/how-to-be-a-winning-football-bettor-using-draftkings-and-circa-nfl-betting-splits/)
- Paul & Weinbach (2008, Int. J. Sport Finance) used Sports Insights betting percentages from four offshore books over 3,625 NBA games (2004-05 to 2006-07). Books did not balance action: favorites and overs drew a disproportionate share of bets. Fading 70%+ public favorites won ~52.5%, below the 52.38% break-even at -110, and no contrarian strategy produced statistically significant profit. The paper says this differs from its earlier NFL findings. — [Paul & Weinbach 2008 PDF](https://kylewoodward.com/blog-data/pdfs/references/paul+weinbach-international-journal-of-sport-finance-2008A.pdf)
- Paul, Weinbach & Humphreys (2011) used Sports Insights NFL betting-volume data (1,278 games, 2005-06 to 2009-10). Bets were not balanced. Teams on 2-game win streaks drew about 3% more bets, but fading streaks won only 49.4%-51.2%, which is not significantly different from 50%. Spreads already priced the streak information. — [UAlberta WP 2011-16](https://sites.ualberta.ca/~econwps/2011/wp2011-16.pdf)

### Inferences
- The user already holds The Odds API, including Pinnacle and historical lines. Of the cheap options, only OddsPapi (Asian sharps such as Singbet/SBOBet, and Betfair) or Unabated (a ready-made sharp consensus line) would add something new. Both should be trialed against the existing Pinnacle-based fair line before paying.
- Splits data is cheap to get but has no published out-of-sample validation. Academic tests find little or no contrarian edge after vig. Splits are better used as a logged, pre-registered paper-track feature, which fits this repo's discipline, than as a bet trigger.
- Kalshi/Polymarket quotes are useful as another price signal, but top-of-book depth is too small to bet size, and Arizona has legal exposure (see the legal section).

### Gaps
- No vendor-confirmed current prices for OddsJam API, OpticOdds, SportsDataIO, Don Best, BetQL, OddsShark or SBR odds. All are sales-gated or I could not retrieve them.
- Action Network PRO pricing and features could not be fetched (robots disallowed). Sports Insights/Bet Labs current pricing and status were not found.
- I found no peer-reviewed study from 2015 onward that tests "reverse line movement" or ticket-vs-handle splits from regulated US books (DraftKings/Circa) out of sample. The NFL-specific contrarian finding that Paul & Weinbach say differs from the NBA was not retrieved directly.
- The Odds API's current paid tier prices were not re-verified (the user already subscribes).

## Injury and lineup information timing (NFL and CFB availability reports)

### Takeaway
NFL practice reports come out Wed-Fri by 4:00 p.m. ET, with game statuses on Friday for Sunday games. All Power Four conferences now publish reports for conference games: SEC since 2024, Big 12 and ACC since 2025-26, Big Ten expanded to 4 reports per week from Sept 19, 2026, and the CFP since the 2025 season. A Wed-Sat cadence ending 90 minutes to 2 hours before kickoff creates fixed, schedulable news moments to watch. Industry sources say injury news is priced within seconds, so the realistic edge is reacting to official report drops and pre-game statuses, not beating Twitter.

### Cited Findings
**NFL**
- NFL policy: practice reports are due by 4:00 p.m. ET on Wed/Thu/Fri for Sunday games, Thu/Fri/Sat for Monday games, and Mon/Tue/Wed for Thursday games. Game status reports are due by 4:00 p.m. ET Friday for Sunday games, Saturday for Monday games, and Wednesday for Thursday games. Violations can bring fines, suspensions or forfeited draft picks. (2015 policy document.) — [NFL Football Operations 2015 policy](https://operations.nfl.com/media/1818/2015-injury-report-policy.pdf)
- The "Probable" designation was removed starting in 2016, leaving Out / Doubtful / Questionable. — [ABC7 / AP](https://abc7news.com/post/nfl-removes-probable-designation-from-team-injury-reports/1478760/); [NFL 2016 injury report policy](https://operations.nfl.com/updates/football-ops/2016-nfl-injury-report-policy)
- Action Network says the gap between injury news and market adjustment is "measured in seconds, not minutes" because books run algorithms that scan social media and news feeds. It gives an NFL backup-QB swing of "7 points or more" and a CFB QB loss as worth "7-10 points". These are assertions without data. — [Action Network](https://www.actionnetwork.com/education/how-injuries-affect-betting-lines-a-guide-to-market-movement)

**College (FBS) availability reports**
- SEC (2024 onward): initial report Wednesday by 8 p.m. ET; Thu/Fri updates between end of practice and 8 p.m. ET; final report by 90 minutes before kickoff. Wed-Fri designations are Available (100%), Probable (75%), Questionable (50%), Doubtful (25%) and Out. Gameday designations are Available, Game Time Decision and Out. Reports are posted at secsports.com/reports. Fines are $25K / $50K / $100K for 1st / 2nd / 3rd+ offenses, plus head-coach fines. Non-conference games are not required. — [SI (Sep 11, 2024)](https://www.si.com/college/florida/explanation-of-the-sec-new-availability-report-rule-01j7gw225xf2)
- Big Ten: started in 2023 with one report at least 2 hours before kickoff (questionable/out). — [CBS Sports](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26). From the 2026 season it publishes **4 reports per week** for conference games. For Saturday games these are Wed/Thu/Fri at 8 p.m. ET and Saturday 2 hours before kickoff. Midweek designations are Probable, Questionable, Doubtful, Out and Out-for-first-half. Gameday designations are Game-time decision, Out and Out-for-first-half. Reports are posted on BigTen.org, the first ones came on Sept 19, 2026, and visiting teams list only travel-squad or two-deep players. — [theScore](https://fr.thescore.com/ncaaf/news/3587551/sms:)
- Big 12 (2025-26 onward): daily reports starting three days before each conference game, with a final update 90 minutes before kickoff. Designations are Available, Probable, Questionable, Doubtful and Out. — [CBS Sports](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26)
- ACC (announced July 2025): football reports two days before, one day before, on gameday, and two hours before kickoff, for conference games, posted on theACC.com. No fine policy existed when it was announced. The commissioner cited "pressure from entities or individuals who are involved in sports wagering that attempt to obtain inside information." — [SI](https://www.si.com/college/georgiatech/football/acc-commissioner-jim-phillips-announces-player-availability-reports-will-be-required-in-acc-conference-games-01k0s6d6q12p)
- The CFP requires availability reports from the 2025 season; exact timing was not specified when announced. — [CBS Sports](https://new.cbssports.com/college-football/news/college-football-playoff-will-require-teams-to-provide-player-availability-reports-beginning-with-2025-season/)
- Source conflict: the CBS CFP article lists only four SEC categories (out/questionable/probable/available), while SI lists five including Doubtful. SI quotes the actual policy more fully. — [CBS](https://new.cbssports.com/college-football/news/college-football-playoff-will-require-teams-to-provide-player-availability-reports-beginning-with-2025-season/) vs [SI](https://www.si.com/college/florida/explanation-of-the-sec-new-availability-report-rule-01j7gw225xf2)

### Inferences
- The fixed drop times are a tractable automation target: 8 p.m. ET Wed/Thu/Fri for SEC and Big Ten, plus T-90 min (SEC/Big 12) and T-2 h (Big Ten/ACC). The hourly `watch` job could poll conference report pages right after each drop and timestamp them, the way `news.log_first_seen` does for NFL. The Big Ten's richer 2026 schedule, with probabilistic labels (Probable 75%, etc. in the SEC), maps directly onto the existing `qb_availability` sit-probability blend.
- Non-conference and Group of Five games have no mandated reports. Information asymmetry, and therefore any speed edge, is most plausible there, but so are low limits.
- The repo's earlier finding that "college QB changes are mostly priced by Sunday/Monday" fits Action Network's "seconds" claim. Official reports mostly confirm news already priced. Any residual edge is in gameday "Game Time Decision" and "Out for first half" resolutions near kickoff.

### Gaps
- I found no rigorous public study measuring CFB line reaction to conference report drops (minutes after the 8 p.m. ET release).
- Group of Five conference (AAC, MWC, Sun Belt, MAC, C-USA) report policies were not found.
- Underdog/Rotowire/FantasyPros news-feed latency, pricing and API terms were not researched in depth. No data compared beat-reporter X posts with official reports.

## CFB-specific data sources

### Takeaway
CollegeFootballData remains the free or Patreon-funded backbone, and its tier details live on a separate page. PFF college grades are now cheap at the consumer level ($99.99/yr PFF+), and a "PFF Pro" tier ($199.99/yr, "coming soon" as of Sept 2026) explicitly advertises programmatic/API access. Transfer portal and depth-chart data (On3, 247, Ourlads) have no public APIs; access is by scraping or forum data.

### Cited Findings
- CFBD's usage page points to collegefootballdata.com/api-tiers for current limits and Patreon pricing and does not repeat them. — [CFBD usage and access](https://api.collegefootballdata.com/usage-and-access)
- PFF pricing (Sep 1, 2026): PFF+ is $9.99/mo or $99.99/yr, with Premium Stats on the annual plan, NFL and college tools, and a Guru AI chatbot. PFF Pro is $199.99/yr ("Coming Soon") and adds "Premium Stats Pro" and "Programmatic data access (API/scrape capability)". — [PFF](https://www.pff.com/news/a-new-chapter-of-pff-pricing)
- On3 transfer-portal data is discussed on On3 user forums, and third-party scrapers target On3/247/Rivals athlete pages. I found no official On3 or 247 data API. — [On3 forum thread](https://www.on3.com/boards/threads/transfer-portal-data.6592449/); [Apify scraper listing](https://godberrystudios.com/apify-radar/actor/jungle_synthesizer/college-athletes-on3-247-rivals-scraper)
- FTN sells white-label betting models for CFB, but its charting data is NFL/NBA. — [FTN Data](https://ftnfantasy.com/data)

### Inferences
- Once released, PFF Pro at $199.99/yr would be the cheapest licensed route to college player grades for OL/QB-quality features. Check its ToS for automated use before building on it.
- Without APIs for portal and depth charts, the AI news reader plus conference availability reports is probably a better use of effort than scraping On3/247 or Ourlads, which has ToS risk.

### Gaps
- Current CFBD tier prices and limits (the api-tiers page could not be fetched).
- Ourlads depth-chart terms and update cadence; SIS college data pricing; GameOnPaper, cfbfastR and Saturday Tradition data specifics; team travel or airport data sources. I did not reach these within the tool budget.
- I found no documented evidence that portal or NIL data has improved CFB spread prediction beyond returning production and talent composites, which the user already has via CFBD.

## NFL data sources (PFF, FTN/DVOA, SIS, NGS, referees, depth charts)

### Takeaway
FTN charting is licensable at $3,000/yr private or $5,000/yr commercial, with an API, data from 2019 and overnight delivery about 24 hours after games, plus DVOA history back to 1978. PFF consumer pricing is low, with an API-capable Pro tier announced. The repo's own README shows most of these signals (NGS, referee tendencies, OL clusters, etc.) were already tested without gain.

### Cited Findings
- FTN Data: "750+ NFL data points" of charting; participation and route data; coverage and pressure metrics; charting from 2019 and expanded participation from 2021; DVOA back to 1978; API docs at charting.ftntools.com/api/docs; pricing "begins at $5,000 annually for commercial use and $3,000 for private use"; games charted about 24 hours after completion. — [FTN Data](https://ftnfantasy.com/data)
- PFF Pro ($199.99/yr, coming soon) lists programmatic data access. — [PFF](https://www.pff.com/news/a-new-chapter-of-pff-pricing)

### Inferences
- With NGS, referee, OL-continuity proxies, travel and special teams already tested without gain (CLAUDE.md backlog), FTN's paid charting (coverage and pressure) is the main untested NFL dataset. It would mostly duplicate EPA signals, though. Use the $3,000 private license only if a free-sample backtest shows a gain on the 2015-19 validation set.

### Gaps
- Sports Info Solutions pricing, current NFL tracking-data access (public NGS only), and Ourlads ToS were not verified.
- FTN's license terms for the free charting subset distributed through nflverse were not found on the FTN page.

## Weather: forecast vs observed, wind and totals

### Takeaway
The best public CFB study (~7,300 games since 2005) finds unders hit 54.7% above 10 mph and about 58-60% above 15-17 mph. That works out to about 1.5 points of total per 10 mph not priced. It used observed station data and warns that forecasts often miss. Newer NFL claims of a 10-15 mph "moderate wind" edge publish no numbers.

### Cited Findings
- Football Study Hall (2018): ~7,323 CFB games from 2005 on, using Weather Underground nearest-station wind. Unders hit 50.2% under 10 mph, 54.7% over 10 mph, ~58% over 15 mph and ~60% over 17 mph (n≈250), against a 51.3% all-games baseline. The study cites "1.5 points of value per 10mph of wind" relative to Vegas totals and asks "How much do you trust weather predictions?" — [Football Study Hall](https://www.footballstudyhall.com/2018/6/25/17500384/football-betting-windy-conditions-effect)
- StartupHub.ai (Aug 27, 2026) claims ROI on blind unders in 10-15 mph sustained wind, stronger in warm weather, but gives no sample, hit rate or forecast-vs-observed detail. — [StartupHub.ai](https://www.startuphub.ai/news/nfl-betting-does-moderate-wind-create-unders-market-inefficiencies)

### Inferences
- The FSH study used observed wind, which matches the repo's finding that the "recorded-wind version leaks". A tradeable rule has to use the forecast available at bet time, which is what `totals_wind_rules.json` already does. The FSH effect size is an upper bound for a forecast-based strategy.
- CFB has more open-air, high-wind venues and softer totals than the NFL, so extending the Tuesday forecast-wind under track to FBS is a reasonable pre-registered next step.

### Gaps
- No head-to-head accuracy comparison of Open-Meteo, NWS and Tomorrow.io for stadium wind at T-72h vs T-3h was found. No stadium wind-exposure (bowl orientation or shielding) dataset was found.

## Social / alternative data (X API, Reddit, contests, consensus)

### Takeaway
X's API became pay-per-use in February 2026: $0.005 per post read with a 2M posts/month cap, or Pro at $5,000/mo for higher volume and streaming. Monitoring a curated list of ~100 beat reporters is now affordable, but real-time filtered streaming needs Pro. I found no recent quantitative evidence that fading the public in SuperContest or Circa contests is profitable.

### Cited Findings
- X API pay-per-use: post read $0.005 per post returned, user lookup $0.010, post creation $0.015 ($0.20 with a URL from April 2026), owned reads $0.001. Pay-per-use reads are capped at 2M posts/month. Basic ($200/mo) closed to new signups in Feb 2026 and auto-migrated June 1, 2026. Pro is $5,000/mo, and filtered stream and archive search "push you into Pro". — [OpenTweet explainer](https://opentweet.io/how-to/x-api-pay-per-use-explained)
- X ended its free tier and moved to pay-per-use as the default on Feb 6, 2026; existing free users got a one-time $10 voucher. — [Roboin](https://roboin.io/article/en/2026/02/08/x-transitions-api-to-pay-per-use-model-ending-free-plan/)
- SuperContest consensus: in one 2018 snapshot the top-5 consensus picks went 3-2 in a 3,120-entry contest. This is anecdotal, with no multi-season record. — [National Football Post (Oct 2, 2018)](https://www.nationalfootballpost.com/columns/betting/supercontest-contestants-maintain-pace-as-top-consensus-pick-deliver/)

### Inferences
- Polling ~100 beat-reporter timelines every few minutes during practice-report windows costs about $0.005 × posts returned. Since only new posts bill, it is likely tens of dollars per month. This is an inference; verify actual billing behavior. True sub-second streaming at $5,000/mo cannot be justified given the "seconds" market reaction.

### Gaps
- Reddit API commercial pricing, Discord sharp communities, and Google Trends value were not researched in this pass.
- Multi-season SuperContest or Circa Million consensus-fade records were not found.

## LLM-based news extraction edges

### Takeaway
I found no documented, verifiable case of an LLM news parser beating NFL/CFB betting markets. Industry sources say books already scan social and news feeds automatically within seconds. Academic LLM-sentiment trading work exists for equities, not sports.

### Cited Findings
- Sportsbooks "employ automated algorithms that scan social media and news feeds for injury keywords", and adjustment happens in seconds. — [Action Network](https://www.actionnetwork.com/education/how-injuries-affect-betting-lines-a-guide-to-market-movement)
- LLM sentiment trading research exists for stock returns (e.g., LSE "Sentiment trading with large language models"; arXiv 2304.07619 on ChatGPT forecasting stock moves from headlines), not betting markets. — [LSE Research Online](https://researchonline.lse.ac.uk/id/eprint/122592); [arXiv 2304.07619](https://arxiv.org/html/2304.07619v6)

### Inferences
- An RSS + Haiku pipeline is unlikely to beat books' keyword scanners on headline speed. Its likely value is interpretation: resolving ambiguous practice notes, depth-chart signals and "expected to start" language before the official report codifies them. The repo's `news_audit` lead-time measurement is the right test.

### Gaps
- No public backtest or paper quantifying LLM-news lead time against sports line moves was found.

## Legal / ToS constraints (scraping, prediction markets in Arizona)

### Takeaway
DraftKings' Terms of Use (updated May 5, 2026) explicitly ban automated scraping. Arizona criminally charged Kalshi in March 2026 and the dispute was still in litigation in September 2026, so an Arizona-based bettor should treat prediction-market execution as legally unsettled.

### Cited Findings
- DraftKings ToU prohibits "Using automated means (including but not limited to harvesting bots, robots, parser, spiders or screen scrapers) to obtain, collect or access any information on the Website or of any User for any purpose"; the terms were last updated May 5, 2026. — [ConductAtlas](https://conductatlas.com/platform/draftkings/draftkings-terms-of-use/provision/CA-P-039378/automated-scraping-of-website-information-prohibited/)
- Kalshi sued Arizona officials on March 12, 2026. Arizona filed 20 misdemeanor criminal charges on March 17, 2026 for an unlicensed gambling business. A federal judge denied Kalshi's TRO on March 18, 2026. As of September 2026 Arizona and Kalshi were suing each other, with 20+ lawsuits nationally. — [Front Office Sports (Sep 15, 2026)](https://frontofficesports.com/article/arizona-sues-kalshi-after-kalshi-sued-arizona/); [Morgan Lewis](https://morganlewis.com/pubs/2026/03/arizona-files-first-criminal-charges-against-a-prediction-market)

### Inferences
- Getting sportsbook prices through licensed aggregators (The Odds API, Unabated, OddsPapi) instead of scraping book sites avoids the DraftKings-style ToS breach and the account-closure risk that comes with it. The same logic applies to the Apify scrapers of Action Network and VegasInsider.

### Gaps
- Explicit consequences in the DraftKings terms (voided bets, closure) were not shown on the provision page. FanDuel, BetMGM and Caesars scraping clauses and Arizona Department of Gaming rules on automated betting were not checked.
