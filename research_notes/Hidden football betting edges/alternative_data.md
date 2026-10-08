# Alternative / non-mainstream data sources for NFL + college football betting

Scope note: research run 2026-10-07, about 20 tool calls. Sources were fetched where possible; search-snippet-only items are marked. Evidence grades: **A** = peer-reviewed or large out-of-sample test vs closing line; **B** = practitioner tracking with stated sample but no CLV/out-of-sample discipline; **C** = anecdote or marketing; **none** = no evidence found. "Test" proposals are the researcher's inferences, written to fit the repo's pre-registration conventions (rules JSON with frozen version, separate ledger, CLV as the primary judge).

## Public betting splits (ticket % vs handle %) — fade the public / follow the sharps

### Takeaway
DraftKings is the one major U.S. book whose bet-count and handle splits are published free, and VSiN republishes and backtests them every week. VSiN's own NFL "splits systems" show the strongest public-side groups (for example >65% of handle) covering around 48%. That is a weak contrarian lean, not enough to beat -110. Academic tests of contrarian betting with older percentage data (Sports Insights) could not reject efficiency. Splits are cheap to log, but on the current evidence they are a CLV-test candidate, not a known edge.

### Cited Findings
- VSiN's weekly "NFL Betting Splits Systems" articles use **DraftKings betting splits**, giving both handle (money) % and bets (tickets) % for spreads, moneylines and totals. DK is the only book named as the source — [VSiN Week 18 splits update](https://vsin.com/nfl/nfl-week-18-betting-splits-systems-update/)
- VSiN system records, quoted from the 2024-25 Week 18 update (backward-looking, sample periods described loosely) — [VSiN](https://vsin.com/nfl/nfl-week-18-betting-splits-systems-update/):
  - Handle >65% on one side (spread): **130-139 (48.3%)**, "past two-plus seasons"
  - Handle majority in divisional games: **91-117 (43.8%)**, 2022-24. Bets majority in divisional games: **96-114 (45.7%)**
  - Majority backing road favorites: **108-123 (46.7%)** since 2022. Majority backing road dogs: **71-65 (52.2%)**
  - Majority in non-Sunday-afternoon games: handle **78-95 (45.1%)**, bets **76-94 (44.7%)**
  - Handle 64%+ on the Over: **121-153 (44.2%)**. Same, with the total moving down during the week: **50-74 (40.3%)**
  - Handle 56%+ on the Under: **70-52 (57.4%)**. Majority handle on totals of 44 or less: **97-66 (59.5%)** since Nov 2023
  - ML: handle >75% on favorites of -4.5 or more: 23-13, **-43.1% ROI** (2024). Majority bets on small ML favorites in Sunday-afternoon games: 33-10, +44% ROI (2024)
- Many VSiN systems are subgroups mined from 2-3 seasons of data (divisional, Sunday-afternoon, total ≤44 and so on), with no stated pre-registration and no CLV — [VSiN](https://vsin.com/nfl/nfl-week-18-betting-splits-systems-update/). Evidence grade **B/C**: a classic multiple-comparisons setting.
- Academic evidence: Paul & Weinbach (2008, *International Journal of Sport Finance* 3(3):137-145) used **Sports Insights** bet-count percentages from four offshore books (BetUS, CaribSports, SportBet, Sportsbook.com) across **3,625 NBA games (2004-05 to 2006-07)**. Books do not balance action: favorites and overs draw a disproportionate share of bets. But "none of the win percentages based on these simple strategies could reject the null of no profitability." Betting the dog when 70%+ of bets were on the favorite won **52.474%**, and betting the under when 70%+ were on the over won **48.469%**. The authors contrast this with their earlier NFL work, where similar imbalances did yield exploitable results (that NFL paper used Sportsbook.com dollar data) — [Paul & Weinbach 2008 PDF](https://kylewoodward.com/blog-data/pdfs/references/paul+weinbach-international-journal-of-sport-finance-2008A.pdf). Evidence grade **A** for "imbalance exists", **A** for "simple contrarian rules are not reliably profitable (NBA)".
- Paul & Weinbach also published "An Analysis of the Last Hour of Betting in the NFL" (pp. 307-316). The title was seen in search only and the content was not fetched — [FiT Publishing](https://fitpublishing.com/node/1024)
- Popular-press claims such as "fading the public has worked in 2017, unpopular teams covering at a high rate" are single-season anecdotes — [theScore 2017](https://www.thescore.com/news/1396316); [Yahoo](https://au.sports.yahoo.com/nfl-betting-fading-the-public-has-worked-unpopular-teams-covering-at-a-high-rate-140205279.html) (redirect, not fetched). Evidence grade **C**.
- A 2006 Las Vegas Sun column advised against "following the money" — [Las Vegas Sun](https://m.lasvegassun.com/news/2006/nov/10/jeff-haney-advises-against-following-the-money-on-/) (snippet only)

### Inferences
- **Free sources an automated system could log:** the DraftKings splits feed, reachable through VSiN's DK splits page and articles (check robots.txt and terms before scraping the page itself; the repo already reads VSiN RSS in `picks_log`). Circa has published sharp-vs-public splits in the past, but its availability and terms were not verified in this run. Action Network splits are off-limits because the repo already treats Action Network's robots.txt as blocking.
- DK splits are **one retail book's customers**. That makes them a good "public money" proxy, but "handle% >> ticket%" at DK reflects a few large DK bettors, who are not necessarily sharp, since sharps get limited at DK. Treat the handle-vs-ticket gap as noisy.
- The VSiN numbers suggest the effect, if real, sits mostly in **totals** (over-heavy public loses, the under is favored when handle leans under) and in **heavy-handle favorites on the moneyline**. Both fit the known favorite/over bias found in academic work.
- **Concrete test (pre-register as `splits_rules.json` v1):** log DK splits hourly next to the existing `history/odds_*.json` snapshots for 2026 weeks 6-18. Rule A: side with DK handle ≥ 70% on a spread → bet the other side at the best allowed-book price 1-3 h before kickoff. Rule B: total with DK handle ≥ 64% on the Over → Under. Judge on **CLV vs Pinnacle Shin fair** (the repo standard), pass at mean CLV > 0 with p < 0.05 over ≥ 150 bets. As a secondary check, add a regression of closing-minus-snapshot line movement on the handle-ticket gap, to see whether DK splits *lead* Pinnacle moves (they probably lag).
- The repo has no historical splits, so any backtest would rely on VSiN's published records. Those can't be audited, so live logging is the only clean test.

### Gaps
- No free historical archive of DK/Circa splits (with timestamps) was found. Sports Insights/Action Labs historical data is paid, and its current pricing was not verified.
- No peer-reviewed NFL study using *handle vs ticket* splits (as opposed to bet counts) against **closing** lines was found in this run.
- Circa's current splits page, and whether it is machine-readable, was not verified.

## Practice reports, beat reporters, transaction wires, and college availability reports

### Takeaway
The biggest *structural* change in 2025-26 is in college football. All Power Four conferences, plus the Mountain West and the CFP, now require player availability reports, and in 2026 the Big Ten moved to **four reports a week (Wed/Thu/Fri at 8 p.m. ET plus gameday)**. This is a new, timestamped, official, scrape-able source that `cfbpred` does not yet ingest. Its value against the line is untested, but it allows a clean "news-vs-line timing" study like the NFL `news_audit`.

### Cited Findings
- **Big Ten:** first Power Four league to adopt reports (2023). The original rule required submission **two hours before kickoff**, with categories "questionable" or "out" — [CBS Sports (CFP article)](https://new.cbssports.com/college-football/news/college-football-playoff-will-require-teams-to-provide-player-availability-reports-beginning-with-2025-season/); [CBS Sports Big 12 article](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26)
- **Big Ten 2026 expansion:** for Saturday games, reports are due **Wednesday, Thursday and Friday by 8 p.m. ET**, plus a gameday report **two hours before kickoff**. Midweek categories are probable / questionable / doubtful / out / out for first half. Gameday categories are game-time decision / out / out for first half. Unlisted players count as available. Visiting teams must also list players who are not on the travel squad but have been on the two-deep or play regularly. Reports are published on **BigTen.org**, cover **conference games only**, and started Sept 19, 2026 (USC-Rutgers, Purdue-UCLA) — [Santa Fe New Mexican (AP)](https://www.santafenewmexican.com/sports/big-ten-will-release-4-player-availability-reports-per-week-for-games-matching-conference-teams/article_b2453480-1e2a-5983-a8c1-9e651f7726cb.html); also [Altoona Mirror, Aug 2026](https://www.altoonamirror.com/sports/psu/2026/08/cfb-bigten-upping-availability-reports/) (snippet only)
- **SEC (2024):** categories are out / questionable / probable / available. Initial statuses are filed Wednesdays and updated daily, with the final report due **90 minutes before kickoff**. Escalating fines apply for late or inaccurate reports — [CBS Sports (CFP article)](https://new.cbssports.com/college-football/news/college-football-playoff-will-require-teams-to-provide-player-availability-reports-beginning-with-2025-season/). A second CBS article summarizes the SEC as "reports required three days before games" — [CBS Big 12 article](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26). These are consistent if the reports start on Wednesday for a Saturday game.
- **ACC:** announced July 2025 that reports are required for football, basketball and baseball conference games from 2025 — [CBS Big 12 article](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26); [SI Georgia Tech](https://www.si.com/college/georgiatech/football/acc-commissioner-jim-phillips-announces-player-availability-reports-will-be-required-in-acc-conference-games-01k0s6d6q12p) (snippet). The ACC's exact timing was not confirmed.
- **Big 12:** announced Aug 13, 2025, effective 2025-26 for all conference games. Daily reports start **three days before** each game, with a final update **90 minutes before kickoff**. Categories are available / probable / questionable / doubtful / out — [CBS Sports](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26)
- **Mountain West** adopted football availability reports for 2025 — [SI Boise State](https://www.si.com/college/boise-state/football/mountain-west-follows-lead-power-four-adds-player-availability-reports-2025-football-season) (snippet only)
- **CFP** requires teams to provide availability reports from the 2025 season. Timing and format were not specified in the article — [CBS Sports](https://new.cbssports.com/college-football/news/college-football-playoff-will-require-teams-to-provide-player-availability-reports-beginning-with-2025-season/)
- Officials say the stated purpose is integrity: to protect athletes and staff from pressure to leak inside information as legal betting spreads — [CBS Sports](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26)

### Inferences
- **College availability reports are the strongest new-data lead in this catalogue.** They are official, timestamped and free. Before them, college injury information came only from coach-speak and beat reporters, so the market's absorption of the information is plausibly slower than in the NFL, especially for G5 games and lower-profile P4 games.
- **Concrete test (`cfb_availability_rules.json` v1):** scrape BigTen.org, SEC, ACC, Big 12 and MWC report pages every hour (Wed-Sat). Record first-seen time and status per player, and join to CFBD rosters, usage/PPA and returning production to get a "starter weight". For each first appearance of a QB or a starter with ≥ 50% of last season's snaps listed as doubtful/out, compare the Pinnacle (or sharpest available) spread at first-seen time with the close. Hypothesis: mean move toward the injured team's opponent ≥ 0.5 pt after first-seen. If the move is mostly *after* our timestamp, bet at first-seen on allowed books. Judge with CLV (≥ 100 events, p < 0.05). Use the same audit pattern as `news_audit.py`: precision of report vs actual participation, from CFBD/PBP player appearances.
- Non-conference games aren't covered, which gives a natural **control group** for measuring how much the reports changed market efficiency (compare pre-2025 and post-2025 closing-line error on conference vs non-conference games).
- **NFL practice participation (DNP/limited/full, Wed/Thu/Fri):** the repo already parses Sleeper/ESPN injury fields and notes that practice_participation hasn't yet been seen live. The obvious test is "Wednesday DNP for a starting QB/LT/CB1 → line move by Friday". No study measuring this against the line was found in this run.
- **NFL transaction wires** (IR designations and practice-squad elevations, due by 4 p.m. ET the day before the game) are on NFL.com transactions, and nflverse has a players/rosters feed. These are mostly *confirmations* of news already priced. Low priority.

### Gaps
- Exact ACC report timing, and whether conference sites expose machine-readable (non-PDF) reports, was not verified.
- No study was found measuring college line moves around availability-report release times. The test above would be original work.
- No evidence was found (pro or con) on beat-reporter practice observations, local TV/radio, or NFL practice-participation timing vs line moves. The repo's own `news_audit` lead-time measurement is the only data.

## Tracking / charting data (PFF, SIS, FTN, Big Data Bowl, TruMedia, college tracking)

### Takeaway
The only charting source that is free and machine-accessible is **FTN charting through nflverse (2022+)**: play action, motion, RPO, blitzers, pass rushers, catchable ball, drops, interception-worthy throws, QB-fault sacks. It is *not* among the already-tested sources (NGS was). PFF and SIS are paid, with no public evidence of value against the closing line. PFF's own betting content is promotional.

### Cited Findings
- nflverse distributes **FTN charting** with 28 fields, including `is_motion`, `is_play_action`, `is_screen_pass`, `is_rpo`, `is_trick_play`, `is_qb_out_of_pocket`, `is_interception_worthy`, `is_throw_away`, `read_thrown`, `is_catchable_ball`, `is_contested_ball`, `is_created_reception`, `is_drop`, `n_blitzers`, `n_pass_rushers`, `is_qb_fault_sack` and `n_offense_backfield`. It is keyed to `nflverse_play_id` and available **"from 2022 onwards"**, pulled by nflverse jobs from the FTN Data API. The license and update cadence are not stated on the dictionary page — [nflreadr FTN charting dictionary](https://cloud.r-project.org/web/packages/nflreadr/vignettes/dictionary_ftn_charting.html)
- PFF publishes data studies (e.g. "Coverage vs. Pass Rush") and betting articles ("Why betting early is critical to beating NFL markets", weekly "data-backed bets"). These are paywalled or marketing material with no audited record against the close — [PFF coverage vs pass rush](https://www.pff.com/news/pro-pff-data-study-coverage-vs-pass-rush); [PFF betting early](https://www.pff.com/news/bet-why-betting-early-critical-beating-nfl-markets) (snippets only). Evidence grade **C**.

### Inferences
- **Concrete test (FTN):** build "process" stats from FTN charting that EPA doesn't capture directly. Examples: interception-worthy-throw rate (luck-adjusted INT rate), drop rate (offense luck), QB-fault vs OL-fault sack split, and blitz/pressure rates allowed. Add them to `scripts/extra_features.py` with `shift(1)` rolling and the leakage test. Validation has to run on **2022-25 only**, which breaks the repo's 2015-19 / 2020-25 split. Use weeks 1-9 of 2022-23 for tuning and 2024-25 for holdout, and say clearly that the sample is small (about 1,100 games). Most promising: **"INT luck"**, actual INTs minus expected INTs from interception-worthy throws, as a regression-to-mean signal. The market may overreact to turnovers, which the turnover-regression literature suggests. Kill rule: if the change in holdout log loss is under 0.001, drop it.
- The **Big Data Bowl** tracking data (Kaggle, one partial season per year) is too small and too lagged for weekly prediction. It is useful only for building priors (e.g. coverage-type effects). Not verified in this run.
- **PFF grades** (paid, about $40/yr for consumer tiers; price not verified) and **SIS DataHub** (paid) have no public out-of-sample evidence against closing lines. Low priority unless a trial allows a 2020-25 backtest.
- College tracking (Telemetry, SportSource Analytics, PFF college) is B2B and paid. No public evidence was found.

### Gaps
- PFF/SIS/TruMedia pricing and API licensing were not verified.
- No public study was found showing that pass-rush win rate or PFF grades predict ATS outcomes against closing lines.

## Schedule / fatigue / logistics data (charter flights, crowd noise, altitude, college academics)

### Takeaway
Nothing found in this run documents value against the line for charter flight tracking, crowd noise, hotel disruption, or college exam/eligibility data. Travel, body clock, rest and altitude-adjacent factors are already tested without gain in the NFL model. This category is low priority except for **college late-season eligibility and bowl opt-outs**, which overlap with availability reports and portal news.

### Cited Findings
- No sources with evidence were retrieved in this run. (Searches were budgeted toward splits, availability reports and markets.)

### Inferences
- Charter flight tracking (ADS-B: ADS-B Exchange or FlightAware; team charters often fly under charter carrier callsigns) could detect **late or disrupted travel**. These events are rare (a few per season), so they can't be tested statistically. At most, log them as an alert.
- **Bowl season opt-outs**, from portal entries, NFL draft declarations and coach announcements, are a known, widely discussed bowl-market factor. The CFP availability-report requirement now formalizes some of this for playoff games.

### Gaps
- No evidence on crowd noise, hotel disruption, or college exam schedules vs the line.

## Market data others ignore (Kalshi/Polymarket, offshore openers, futures/win totals, props-implied totals, lookahead lines)

### Takeaway
Prediction markets are now large, with about 5% of regulated handle reported, but their NFL pricing was found to be *worse* (laggier and costlier) than DraftKings/FanDuel in 2025. That makes them more of a **target** (stale prices to pick off, or resting-bid fills, which the repo's `kalshi_maker` track already tests) than an information **source**. By 2026, straight-bet costs on Kalshi had converged with the big books. No study comparing Kalshi closing prices with Pinnacle was found.

### Cited Findings
- Bettormetrics' 2025 NFL regular-season cost study (covered Jan 14, 2026) found that Kalshi's fee of 7% × contracts × P × (1−P), rounded up, makes a $0.51 contract cost about $0.5275-0.53. DraftKings -110 is about $0.524. Kalshi is costlier than the big books when the fair price is $0.40-0.60. Crypto.com charges a flat $0.02. Sporttrade charges 2% commission but has 2-4% spreads. Kalshi NFL pricing was described as "consistently worse" (lagging) than DK and FD. Prediction markets reportedly hold about 5% of regulated sportsbook handle — [Casino.org](https://www.casino.org/news/prediction-markets-have-sports-pricing-problems/)
- By 2026, Citizens (Jordan Bender) measured implied vig on straight bets at **Kalshi 4.28% vs FanDuel 4.41% vs DraftKings 4.47% (Week 4, 2026)** and Kalshi 4.38% / FD 4.37% / DK 4.52% in Week 3. On parlays/combos, Kalshi was 26.5% vs a 23.2% DK/FD average. Kalshi holds an 84% 30-day share of the combo market (Polymarket US 9%, DKeX 3%). The article says Kalshi was "more expensive" than both books during 2025 — [RotoWire](https://www.rotowire.com/article/kalshi-beats-draftkings-fanduel-on-nfl-week-1-pricing-134248)
- Reported analyst notes flagged a pricing gap between Kalshi and sportsbooks in NFL Week 1 2025, when Kalshi traded about $441M (headline only) — [Yogonet](https://www.yogonet.com/international/news/2025/09/09/115258-kalshi-trading-hits-441m-in-first-nfl-week-but-analysts-flag-pricing-gap-with-sportsbooks); [CDC Gaming: Kalshi pricing worse than sportsbooks, says Citizens](https://cdcgaming.com/brief/kalshi-pricing-remains-worse-than-sportsbooks-says-citizens/) (snippets only)
- Repo context (from internal CLAUDE.md, not external): Circa and Bookmaker never appear in The Odds API feed, so offshore and Circa openers are not available through the current vendor.

### Inferences
- **Kalshi/Polymarket as an information source:** their public order books and trade tapes are free (and Polymarket's on-chain trades are fully public). A testable question is whether **large Kalshi trade prints lead Pinnacle moves**. **Test:** for each NFL game, find Kalshi trades over $10k notional (or 99th-percentile size), and measure Pinnacle's no-vig move over the next 30 min vs a matched control window. If the move is > 0 with p < 0.05 over ≥ 200 events, Kalshi flow leads the market. Prior: unlikely, given the documented lag, but cheap to run because `kalshi_maker` already pulls Kalshi public trades.
- **Lookahead lines** (next week's games posted the Sunday or Monday before): The Odds API lists events once books post them. Compare lookahead lines with the re-opened lines after the current week's results to measure overreaction. Test: on Sunday night, after re-open, if the line moved ≥ 2 pts from lookahead, bet back toward the lookahead number. Judge by CLV to the close. No external evidence was found; this is an original hypothesis. Check it against the repo's "Sunday-night openers" result, which was tested without gain.
- **Win-total and futures movement as a team-strength signal:** futures prices are already used in `futures_value`. A movement-based rating, for example implied season wins, could be built from Odds API outrights. But season win totals mostly come off the board in-season, so coverage is thin.
- **Props-implied team totals:** sum the player prop medians (passing, rushing and receiving yards, anytime-TD probabilities) into team-level expected points and compare with the game total and spread. Disagreements flag either a soft prop market or a soft main market. The repo already stores props responses (`history/props_*.json.gz`), so this is a free test: compute TD-prop-implied team points vs the main-market implied team total, and test whether the residual predicts the closing move of the team total.

### Gaps
- No academic or practitioner study was found of Kalshi/Polymarket NFL closing prices vs Pinnacle (efficiency or information leadership).
- No free feed of BetCRIS, Bookmaker or Circa openers was confirmed.
- No evidence was retrieved on lookahead-line overreaction.

## Social / attention data (Google Trends, Wikipedia, Reddit/Bluesky, ticket resale, TV)

### Takeaway
The best-documented attention source, Google Trends, does carry sentiment that shows up in Vegas closing spreads. In a Duke study, search interest the day after the previous game even predicted realized margins after controlling for the close. Even so, **no strategy beat 52.38% out of sample (2016)**. Attention data looks like a public-money proxy (it *moves* lines) rather than an edge. Expect the same from Wikipedia pageviews, which are free through the Wikimedia API.

### Cited Findings
- Gidumal & Muench (2017, Duke economics honors, *Duke Journal of Economics*): Google Trends for each NFL team's name, 2010-16, 32 teams × 119 weeks (966 and 774 observations), spreads from Caesars, 2016 held out. The "Game Day +1" search differential was significant for **opening** spreads (coefficient 0.1949, p < 0.01, wrong expected sign). The weekly search differential was significant for **closing** spreads (0.5224, p < 0.01) but **not** for spread movement. The GD+1 search differential predicted the realized margin (about 1.25, p < 0.01) even with the closing spread in the regression. But they "fail to find any strategy that consistently outperforms the hurdle rate of 52.38%" in out-of-sample 2016 data — [Gidumal & Muench 2017 PDF](https://sites.duke.edu/djepapers/files/2017/06/shivgidumalrolandmuench-dje.pdf). Evidence grade **B** (student paper, one holdout season).
- "Predicting the NFL using Twitter" (arXiv 1310.6998) used tweet volume and sentiment to predict NFL outcomes and betting results. It was found in search but not fetched, so its numbers are unverified — [arXiv](https://arxiv.org/pdf/1310.6998)
- A PLOS ONE article (PMC10431623 / RePEc pone00/0289213) appeared in the search for Google Trends / Wikipedia prediction of sports outcomes. It was not fetched, so its relevance is unverified — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10431623/)

### Inferences
- Gidumal & Muench's result, where search interest predicts the margin *beyond* the closing spread in-sample but fails out of sample, is the typical pattern for attention data: a real signal too small to beat the vig. If tested, frame it as a **"model + market" blend variant** (repo rule 3), judged by `vs_vegas.py`, not as a betting rule.
- **Concrete test (cheap, free):** pull Wikimedia pageviews (REST API, daily, free, no key) for the 32 NFL team articles and about 130 FBS team articles, 2015-25. Feature: z-scored pageview change from the previous game day to the next game's Wednesday, differenced between home and away. Add it to the `vs_vegas.py` blend. Pass if the 2020-25 blend log loss beats Vegas alone by ≥ 0.001 with a bootstrap CI excluding 0. Expect failure in the NFL; college attention gaps (mid-majors after a viral upset) are the more plausible place for mispricing.
- **Ticket resale prices** (StubHub/SeatGeek/TickPick): SeatGeek has a public API with prices, but the current terms and historical access were not verified. These could proxy fan demand (local public money), but no evidence links them to ATS results. Low priority.
- Reddit is excluded, matching the repo's earlier finding that it requires a login and blocks cloud IPs. Bluesky is already ingested.

### Gaps
- No evidence was found for ticket resale, TV audience, or Reddit/Bluesky sentiment against **closing** lines.
- The Twitter and PLOS ONE papers were not fetched, so their findings are unverified.

## College-specific (depth charts, NIL/portal, hot seats, bowl opt-outs) and overall evidence against the close

### Takeaway
College is where the information asymmetry is plausibly largest. 2025-26 conference availability reports are new official data. Depth charts released by schools and portal and opt-out news add to it. No source in this run showed any of them beating the closing line, so every proposal here is a test to pre-register, not a known edge. Overall, the reviewed literature points one way: alternative signals (bet percentages, search interest) are *already reflected in closing prices* and simple strategies on them do not beat -110 out of sample.

### Cited Findings
- Conference availability-report rules and their timing are listed in the practice-report section above. The integrity purpose stated by officials implies that information previously leaked unevenly — [CBS Sports](https://www.cbssports.com/college-football/news/big-12-to-mandate-player-availability-reports-for-football-basketball-starting-in-2025-26)
- Contrarian strategies on bet percentages are not significantly profitable — [Paul & Weinbach 2008](https://kylewoodward.com/blog-data/pdfs/references/paul+weinbach-international-journal-of-sport-finance-2008A.pdf). Search-attention strategies are not profitable out of sample — [Gidumal & Muench 2017](https://sites.duke.edu/djepapers/files/2017/06/shivgidumalrolandmuench-dje.pdf)

### Inferences
Priority ranking for this repo (researcher's judgment):
1. **College availability reports** (Big Ten/SEC/ACC/Big 12/MWC scrape, first-seen timing vs Pinnacle line, CLV-graded). Free, new and official, and it reuses the `news_audit` pattern.
2. **DK splits logging** (hourly, alongside odds snapshots). Free. Pre-register two rules (heavy-handle spread fade; over-heavy totals → under) judged on CLV. Expect a null result but it costs little.
3. **FTN charting "luck" features** (INT-worthy throws, drops, QB-fault sacks), 2022+. A new angle not covered by the NGS test. Small sample.
4. **Kalshi large-print lead/lag test**, reusing the `kalshi_maker` trade pulls.
5. **Props-implied team totals vs the main market**, reusing the stored props snapshots.
6. **Wikipedia pageviews blend test.** Cheap, likely null.
- Do not pursue: paid PFF/SIS without a trial backtest; charter flights, crowd noise and exam schedules (no evidence, too rare to test).

### Gaps
- Not verified: whether schools publish weekly depth charts in a consistent machine-readable format (many post PDF game notes), NIL-collective data sources, coaching hot-seat indices, or an academic-eligibility list source.
- No academic paper was found testing college injury/availability information against closing lines.
- Not reached in this budget: Unabated, Open Source Football and Journal of Prediction Markets material on these topics.
