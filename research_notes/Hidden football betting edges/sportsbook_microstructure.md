# Sportsbook microstructure: how US books set, copy and move NFL/college prices, and where the process can be exploited

Scope: what Arizona books do with NFL and college football prices (2025-26), for a bettor with DK, FD and theScore Bet accounts, hourly Odds API snapshots of about 25 books including Pinnacle, and paper tracks judged by CLV against Pinnacle.

Evidence labels:
- **[MEASURED]**: a peer-reviewed or systematic data study.
- **[INDUSTRY]**: a press release, regulator page or trade-press fact.
- **[PRACTITIONER]**: a claim by a named pro or tout without data.
- **[ANECDOTE]**: a single example.

Overall caveat: public, rigorous evidence on US sportsbook microstructure is thin. Most of the sources I could reach were trade press, marketing blogs or one-example articles. Many "edges" below are therefore testable hypotheses, not measured facts. They are marked as such and given numeric rules the bot can pre-register.

Already tested by the user, so not re-proposed:
- shop-vs-sharp (college passes, NFL fails)
- opener vs ratings
- live in-game overreaction
- teasers
- line-movement models
- first halves and team totals vs Pinnacle
- longshot moneylines
- receptions props line shopping
- devig methods

## Q1. Which line-setting vendors and platforms supply which Arizona books, and do vendor-shared books copy each other in testable ways?

### Takeaway
Arizona has two Kambi-powered books, BetRivers (Rush Street) and Desert Diamond, and almost certainly a third in Bally Bet. Kambi is the one clear "shared-feed" cluster where a single trading decision should appear at several books at once.
- DraftKings, FanDuel, Caesars, BetMGM, Fanatics and theScore Bet each run their own stack.
- A new market maker, Circa Sports, received an Arizona license effective 2026-07-27.
- Nobody has published a measured cross-book propagation study for US books. The Kambi cluster test has to be built from the user's own hourly snapshots, and hourly is too coarse to see propagation measured in seconds or minutes.

### Cited Findings
- Kambi signed a multi-year deal with Desert Diamond Casinos (Tohono O'odham Nation) on 2021-11-01. It covers retail sportsbooks at West Valley, Sahuarita and Tucson plus a mobile book under Desert Diamond Mobile LLC. The release does not say whether Kambi also runs managed trading. [INDUSTRY] — [iGaming Future](https://igamingfuture.com/kambi-group-plc-signs-multi-year-partnership-with-desert-diamond-casinos-an-enterprise-of-the-tohono-oodham-nation/); [CasinoBeats](https://casinobeats.com/2021/11/01/desert-diamond-casinos-inks-kambi-deal-for-arizona-sports-betting/)
- Bally's dropped its in-house product (Bet.Works and Monkey Knife Fight, over $215M in acquisitions, shut in Feb 2023) and moved to Kambi in May 2023, "the same platform BetRivers uses." Bally Bet relaunched in seven states by the end of 2023. The article does not name the states. It notes Arizona's Bally's handle share was below 1%. [INDUSTRY] — [Legal Sports Report, 2023-05-02](https://www.legalsportsreport.com/114299/2023-ballys-taps-kambi-sports-betting-app-restart/)
- The same article states three other platform moves. [INDUSTRY] — [Legal Sports Report](https://www.legalsportsreport.com/114299/2023-ballys-taps-kambi-sports-betting-app-restart/)
  - DraftKings left Kambi after its $3.3B SBTech merger, so DK runs in-house.
  - Penn (Barstool, later ESPN Bet, now theScore Bet) was moving off Kambi by 2023.
  - Fanatics signed with Amelco.
- ESPN Bet was rebranded theScore Bet on 2025-12-01 after Penn and ESPN ended their partnership. The app "looks, feels, and plays the same," which implies the same underlying (Penn/theScore) stack. [INDUSTRY] — [CDC Gaming](https://cdcgaming.com/brief/a-new-dawn-for-penn-espn-bet-is-replaced-with-thescore-bet/); [iGaming Future](https://igamingfuture.com/penn-and-espn-agree-to-end-u-s-online-betting-partnership/)
- Caesars bought ZeroFlucs in July 2024. ZeroFlucs is a same-game-parlay correlation pricing engine that Caesars had already integrated commercially, so Caesars' SGP pricing is in-sourced. [INDUSTRY] — [Caesars press release, 2024-07-03](https://s202.q4cdn.com/508919455/files/doc_news/Caesars-Entertainment-Acquires-Sports-Betting-Technology-Company-ZeroFlucs-2024.pdf)
- The San Juan Southern Paiute Tribe named Circa Sports Arizona LLC as its event-wagering operator. ADG approval is effective 2026-07-27, and ADG "will work with the licensee on operational launch." The source does not say whether the license covers online, retail or both. [INDUSTRY] — [Indian Gaming](https://www.indiangaming.com/?p=50865)
- Arizona had 14 licensed sports betting operators before the 2026 licensing round. ADG reopened applications from June 26 to July 10, 2026. [INDUSTRY] — [G3 Newswire](https://g3newswire.com/arizona-to-launch-tender-for-more-betting-licences/); [Yogonet](https://www.yogonet.com/international/news/2026/05/29/122156-arizona-to-reopen-sports-betting-license-application-window-in-june); [SBC Americas](https://sbcamericas.com/2026/07/14/arizona-new-retail-betting-licenses/)
- An industry consultancy describes how prices flow between book types. It names no vendors and gives no latency figures. [PRACTITIONER/INDUSTRY] — [SCCG](https://sccgmanagement.com/areas-of-expertise/2024/1/10/market-maker-vs-retail-sportsbook-business-models-and-the-impact-of-price-discovery/)
  - Retail books "source lines from third parties" and "typically follow" the market maker.
  - The market makers it names are Pinnacle and BetCRIS, with Circa noted for college.

### Inferences
- **Kambi cluster (BetRivers, Desert Diamond, Bally Bet).** If all three show the same main-line number and price at the same snapshot, nearly every time, they share one feed. Then a stale Kambi price is stale at three books at once.
  - Only one bet can be placed per cluster. Spreading it across three accounts does add capacity.
  - For CLV accounting, treat the cluster as one book. Otherwise the three copies count the same opportunity as three independent edges.
  - Test: for each NFL/CFB event and snapshot, compute the share of snapshots where all Kambi books carry identical spread, total and ML prices. More than 95% identical means one feed. Lower agreement means operator-level overrides, and the gaps are where a single operator's risk team moved a line, which is worth logging.
  - Check whether the Odds API carries all three: keys like `betrivers` and `ballybet` exist; Desert Diamond may not be present.
- **Kambi lag vs Pinnacle (hypothesis H1).** Kambi trades globally, at a lower priority on US college than a US-native book would give it. Hypothesis: Kambi's CFB prices move later than DK/FD after a Pinnacle move, especially for G5/FCS games and on Saturday mornings.
  - Test with existing hourly data: for each Pinnacle move of at least 1.0 point (spread) or 1.5 points (total) between consecutive snapshots, record whether each book had matched by the next snapshot (1 h) and by the one after (2 h). Compute the fraction of moves still unmatched at 1 h, by book cluster.
  - A cluster with more than 30% unmatched at 1 h on CFB, where DK/FD are below 10%, is a candidate for a "stale cluster" paper track. Bet the cluster's stale side when it shows EV of at least 2% vs Pinnacle Shin fair at the post-move price.
  - Hourly granularity will miss sub-hour lags. A finer test needs 5-minute polling of Pinnacle plus the cluster around known news windows (Odds API credits are the cost).
- **Circa as a reference and an account.** Circa is a sharp, high-limit originator. The repo notes say circasports never appears in the Odds API feed.
  - If Circa Arizona launches online, it gives a second sharp benchmark for college, where Circa is considered strong, and a high-limit account that tolerates winners.
  - Action: watch ADG's operator list and the Odds API bookmaker list for a Circa key.
- **Vendor map, compiled by me and partly from training knowledge; verify before relying on it.** Only the rows sourced above are confirmed here:
  - DK: in-house (SBTech). Confirmed.
  - FD: in-house (Flutter). Not sourced here.
  - BetMGM: Entain stack. Not sourced here.
  - Caesars: in-house, with ZeroFlucs for SGPs. Confirmed for SGPs.
  - Fanatics: Amelco originally, later in-house via the PointsBet US acquisition. Only the Amelco deal is confirmed.
  - theScore Bet: Penn in-house. Inferred.
  - Hard Rock Bet: in-house Hard Rock Digital. Not sourced.
  - BetRivers, Desert Diamond, Bally Bet: Kambi. Confirmed for BetRivers and Bally's corporate; Desert Diamond confirmed.

### Gaps
- No public study measures US cross-book propagation latency in seconds or minutes, Kambi to client or Pinnacle to US book. Not found in academic or trade sources.
- I could not confirm whether Kambi clients in Arizona apply their own risk overrides, or whether Desert Diamond uses Kambi managed trading. The press release is silent.
- The current Arizona operator list, which skins are live and which are in the Odds API feed was not verified. ADG's licensee page was not fetched.
- Whether Circa Arizona has gone live online as of Oct 2026 is unknown.
- I did not fetch Penn's FY2025 10-K ([SEC](https://www.sec.gov/Archives/edgar/data/921738/000092173826000008/penn-20251231.htm)) to confirm that theScore Bet runs on the in-house platform for trading and risk.

## Q2. Measured latency of US books copying Pinnacle/Circa moves, and which markets update slowest

### Takeaway
I found no rigorous public measurement of US-book copy latency for football. The best hard microstructure numbers concern prediction markets, not sportsbooks.
- Polymarket NBA single-market arbitrage lasted a median of 3.6 s, with 7 episodes in 173 games.
- One January 2026 industry snapshot reports US books suspending a lot during college games (uptime 65%+) and carrying overrounds above 5%, with BetRivers (Kambi) above 8%.
- Slow-updating markets are therefore a hypothesis to measure: alt lines, derivatives, small-conference college and props.

### Cited Findings
- EKG data for January 2026, as reported by iGaming Business: [INDUSTRY/MEASURED by a vendor; method not given] — [iGaming Business](https://igamingbusiness.com/sports-betting/market-making-opportunities-2026/)
  - Major US sportsbooks had uptimes of 65%+ during college football games, and DraftKings averaged 86%.
  - Average overround exceeded 5% at every book, and BetRivers exceeded 8%.
- Arbitrage on Polymarket NBA markets: [MEASURED, but a prediction market, not sportsbooks] — [arXiv 2605.00864 via EconPapers](https://econpapers.repec.org/paper/arxpapers/2605.00864.htm)
  - Data: 75M+ order-book snapshots across 173 games.
  - Single-market arbitrage: 7 executable episodes, median duration 3.6 s.
  - Combinatorial arbitrage: 290 episodes, median return 101 bp, mostly in the final minutes.
  - 76.9% of opportunities were limited to about 14.8 shares of depth.
  - Takeaway: mispricings exist but are capped by liquidity.
- Kalshi pricing, Mar to May 2026: [MEASURED, Kalshi only; NBA/MLB/NHL, not football] — [Moshrefi, arXiv 2607.14430](https://arxiv.org/pdf/2607.14430)
  - Data: 23M moneyline trades.
  - Prices are calibrated mid-life but strongly distorted in the final 0 to 10 minutes. The NHL Platt slope is about 4.6, a step-like curve the author attributes to insurance demand.
- Commercial odds-API vendors market "Pinnacle odds changed-at" timestamps and steam detectors. These are product docs, not studies. [PRACTITIONER/marketing] — [SharpAPI docs](https://docs.sharpapi.io/en/concepts/pinnacle-odds-changed-at/); [SharpAPI guide](https://sharpapi.io/learn/how-live-is-real-time-odds-data); [OddsPapi steam detector](https://oddspapi.io/blog/?p=2886)
- An industry consultancy holds that retail books "typically follow" market makers, and names Circa for college. [PRACTITIONER] — [SCCG](https://sccgmanagement.com/areas-of-expertise/2024/1/10/market-maker-vs-retail-sportsbook-business-models-and-the-impact-of-price-discovery/)

### Inferences
- **Overround as a proxy for attention.** BetRivers' 8%+ overround, against 5%+ elsewhere, fits a vendor book pricing defensively. A high-margin book is usually a lagging book: wide margins are a substitute for fast trading. Its main lines may lag less than its derivatives, though. Measure it rather than assume it.
- **Slowest-market ranking to test, using the existing snapshots.** Hypothesized order from slowest to fastest:
  1. FCS and small-G5 CFB derivatives (team totals, first halves at soft books vs Pinnacle). Already tested vs Pinnacle; the new angle is lag after news, not static price.
  2. Alt spreads and alt totals far from the main line.
  3. CFB main lines for low-handle games.
  4. NFL props.
  5. NFL main lines.
  - Metric: per market and book, the median number of snapshots until the price is within 1% (no-vig probability) of Pinnacle after a Pinnacle move of at least 2% no-vig probability.
  - The Odds API historical endpoint and the hourly cadence support this crudely. Alt-line lag needs the per-event alt-market calls (credit cost).
- **Rule to pre-register: "post-move stale soft price."** When Pinnacle's no-vig probability on a side moves by at least 3 percentage points between snapshots, bet any allowed book still offering that side at EV of at least 2% vs the new Pinnacle Shin fair. Bet only if the book's own line has not moved since the earlier snapshot.
  - This differs from static shop-vs-sharp because it conditions on a fresh move: the book is stale, not merely a different opinion.
  - Bet it in NFL even though static NFL shop-vs-sharp failed. The hypothesis is that NFL soft-book edges exist only briefly after moves, and the hourly watch rarely catches them.
  - Decay: minutes. Expect very few fills at hourly cadence, which argues for an event-driven poll (news reader fires, then a 5-minute poll for 30 min).

### Gaps
- No measured copy-latency figures exist for DK, FD, MGM or Caesars against Pinnacle or Circa in football. I found no academic paper, and the trade-press search found none.
- The EKG figures are methodologically opaque: which games were sampled, and whether uptime means all markets or main lines only.
- I have no data on how US books handle suspensions or "line freezes" around injury news, for example whether they suspend or keep offering stale prices. That would need the user's own snapshots, with news timestamps from news_llm.jsonl.

## Q3. Same-game and correlated parlay pricing: documented mispricings and how pros exploit them

### Takeaway
The only peer-reviewed football evidence is 2018 and covers college: favorites that cover go over more often (Davis, Dawson & Krieger, J. Prediction Markets). The paper argues books are overly conservative in banning such parlays, and reports no return figures. Since then, SGP engines (ZeroFlucs at Caesars and others) explicitly price correlation, and national hold has risen to about 10%, with parlays named as the driver. Generic SGPs are therefore likely deeply negative EV. Edges, if any, sit in specific mismodeled correlations, mainly in college and in boosted SGPs.

### Cited Findings
- Davis, Dawson & Krieger (2018), "Correlated parlay betting," *Journal of Prediction Markets* 12(2): 68-84, DOI 10.5750/jpm.v12i2.1562. [MEASURED] — [UWF IR Commons](https://ircommons.uwf.edu/esploro/outputs/journalArticle/Correlated-parlay-betting-An-analysis-of/99380090298406600)
  - Data: college football 2005-2015.
  - Favorites that cover are associated with the game going over.
  - The authors argue books are "too cautious" in refusing these parlays.
  - No return figures are reported on the abstract page.
- Caesars acquired ZeroFlucs (July 2024) as a "fully in-sourced solution for same-game parlay correlation pricing." It enabled in-play SGPs. [INDUSTRY] — [Caesars](https://s202.q4cdn.com/508919455/files/doc_news/Caesars-Entertainment-Acquires-Sports-Betting-Technology-Company-ZeroFlucs-2024.pdf)
- National implied hold rose from 6.9% (2019) to 8.0% (2022), 9.2% (2024) and about 10.2% (2025). LSR attributes this to the shift toward parlays, SGPs and in-play. Separate parlay hold is not given. [INDUSTRY/MEASURED aggregate] — [Legal Sports Report](https://www.legalsportsreport.com/263300/sportsbooks-make-more-money-from-slower-sports-betting-growth-in-2025/)
- On Kalshi, executed parlay price divided by the product of leg prices has a median of 0.996 at 2 to 4 legs. Above that it inflates about 3% per leg (R about 1.30 at 11 legs). Parlays win 2 to 10 percentage points less often than their price implies. These are cross-game parlays only, so correlation is excluded. [MEASURED, Kalshi] — [Moshrefi, arXiv 2607.14430](https://arxiv.org/pdf/2607.14430)
- An OpticOdds vendor blog says SGP correlation is hard to price and that underpricing creates sharp exposure swings. It gives no figures and is marketing. [PRACTITIONER/marketing] — [OpticOdds](https://opticodds.com/blog/correlation-in-same-game-parlays)
- Boosted SGP example: a BetMGM boost from +200 to +250 on Burnes 7+ K plus Brewers ML had an estimated fair price of about +320. FanDuel offered +291 and Caesars +280 unboosted. The boost was -EV, and other books' unboosted SGP prices were materially different for the same combination. [ANECDOTE] — [TheLines](https://www.thelines.com/sportsbook-promo-odds-boost-fool-bettors-nfl-mlb-nba-nhl-2023/)

### Inferences
- **Cross-book SGP price dispersion.** The same two-leg SGP was priced from +250 to +291 across three books. SGP engines disagree on correlation, so "SGP shopping" is the parlay analogue of line shopping.
  - Test: the Odds API does not carry SGP prices. It would need manual or scripted quotes from each book's SGP builder.
  - Simplest pre-registered study: each Tuesday, for every NFL game, request the same three canonical two-leg SGPs at DK, FD, theScore, Caesars and BetMGM:
    - favorite -X plus over
    - favorite ML plus under
    - QB passing yards over plus team over
  - Compare each against a fair price built from Pinnacle's main lines plus an empirical correlation measured on 2015-2025 nflverse results.
  - Log only, for 8 weeks.
  - This is labor-intensive and possibly against terms of service if automated. Manual entry is safest.
- **Where correlation is most likely mispriced: college.** The Davis et al. effect is college-specific, and college spread and total combinations have larger variance. SGP engines are likely tuned on NFL. Many books still ban "spread plus total" same-game parlays in college, or price them with heavy correlation tax. Verify per book.
  - Concrete test with existing data: compute the empirical P(fav covers AND over) on CFBD 2015-2025 results, bucketed by spread size and total. Compare it with independence.
  - If the joint probability exceeds independence by more than 3 percentage points in a bucket, check books' quoted SGP price for that combination. A book treating it as independent, or nearly so, is +EV in that bucket.
  - Decay: slow, since this is a structural model choice, but books fix it once they notice. Limit risk is high: SGP winners get flagged fast.
- **Leg-count lesson from Kalshi.** Even on an exchange, parlays beyond 4 legs are systematically overpriced. Any correlated-parlay play should stay at 2 or 3 legs.

### Gaps
- No public measurement of actual SGP holds at US books, by book, sport or leg count, was found. State reports do not break out SGPs.
- No documented 2024-26 case of a specific persistent +EV NFL SGP correlation was found in reachable sources. The practitioner sources searched (Rufus Peabody's Establish The Run episode page) had no transcript content. [Episode page](https://establishtherun.com/episode-67-one-on-one-with-pro-sports-bettor-rufus-peabody/)
- I did not verify which Arizona books currently allow college spread-plus-total SGPs.

## Q4. Are alternate lines and alt-total ladders priced off distributions that ignore key numbers?

### Takeaway
I found no published measurement of how US books price alt ladders, and no evidence that they ignore key numbers. The sources found were generic explainers and calculators.
- The user already holds the tools to test this directly: a key-number-aware margin and total distribution (margins.py, totals_dist.json, cfb_dist.json) and per-book half-point buy costs (buy_costs.json).
- buy_costs.json already established that buys are never +EV. The open question is the opposite side: selling through key numbers, and alt rungs far from the main line, especially in college.

### Cited Findings
- Explainers describe alt lines as books moving the price per half point along the ladder. None gives measured per-book ladder pricing. [PRACTITIONER, low quality] — [LSR alt lines explainer](https://www.legalsportsreport.com/sports-betting/alternate-lines/); [TheLines](https://www.thelines.com/alternate-lines/); [BettorEdge alt-line calculator](https://www.bettoredge.com/alternate-lines-calculator)
- Repo context (not an external source): CLAUDE.md records that buy_costs.json measured per-book half-point buy costs on 2023-25 alt lines and found buys "never +EV."

### Inferences
- **Hypothesis H4a: selling rungs through dead numbers is overpaid.** Books may charge a flat increment per half point, which overcharges for dead numbers (NFL margins like 5, 8, 9, 11 in the 4.5-to-12 range). If they also pay a flat increment when you sell, then selling across a dead number (e.g., -7.5 to -8.5) gives back more price than the probability lost.
  - Test with the user's own distribution: for each allowed book and each alt rung in the 2023-25 historical alt data, compute EV vs the Pinnacle main line moved along the key-number distribution.
  - Rule to pre-register: bet any alt rung with EV of at least 3% where the rung is at least 1.5 points from the main line and the move crosses only dead numbers.
- **Hypothesis H4b: college ladders use NFL-like shapes.** College margins have different key-number mass: 3 and 7 matter less, and totals are spread over a wider range. A book reusing NFL-shaped increments in college would misprice rungs near 3 and 7 in a predictable direction.
  - cfb_dist.json can price every rung.
  - Rule: when a soft book's alt rung in college differs from cfb_dist pricing by at least 4% EV, and the main line agrees with Pinnacle within 0.5 point, log it as a CFB alt-ladder bet.
- **Hypothesis H4c: alt totals on wind games.** Alt totals may be built from the main total with a fixed variance. In high-wind games the distribution narrows, so far-under rungs may be cheap. This would extend the existing forecast-wind under track to alt rungs.
- Decay: structural and slow, but it needs per-event alt-market Odds API calls, roughly 1 credit per market per region per event.
- Limit risk: moderate to high. Alt-rung bettors who cross dead numbers look unusual, and books profile market choice (see Q7).

### Gaps
- No source measures actual US-book alt-ladder construction or reports +EV rungs. This must be tested on the user's 2023-25 historical alt data.
- The Unabated content on alt lines and key numbers did not surface in search. Its own articles might contain practitioner claims.

## Q5. Promotions, boosts, bonus bets, refunds and insurance: systematic +EV extraction and annual value for an Arizona bettor

### Takeaway
Boosts and promos are the most reliably +EV, lowest-skill microstructure edge, because the book is knowingly giving up price. The examples found show 7% to 20% EV, and one mispriced boost was about 86% EV.
- Maximum stakes are tiny ($25 to $100), and DraftKings reportedly cuts boost maximums to $5 for profiled players.
- I found no credible public estimate of annual promo value for one bettor. It scales with the number of accounts (about 10 or more legal in Arizona) and with discipline, not with model quality.

### Cited Findings
- Method: devig the book's two-way price, or Pinnacle's, to fair probability, then compute EV of the boosted price. Examples: [ANECDOTE set, with self-reported results] — [Action Network](https://www.actionnetwork.com/education/bonus-hunting-for-existing-sportsbook-users)
  - FanDuel Avs/Knights/Kraken ML: +380 vs fair +314, 15.9% EV.
  - FanDuel Murray & Tsitsipas 3-0: +400 vs +316, 20% EV.
  - Barstool Nadal & Zverev: +125 vs +111, 7% EV.
  - Caesars Allen 199.5 yds & Bills: +125 vs +104, 9% EV, $25 max.
  - DraftKings Bills-Pats over 18.5: -110 vs about -8000, about 86% EV.
- Maximum stakes run $25 to $50. DraftKings may limit bettors below the stated $25 maximum, reportedly to $5. New York had "line-moving" crowd promos (FD moved a line one point per 500 bettors; DK moved a total half a point per 5,000 bettors). [ANECDOTE] — [Action Network](https://www.actionnetwork.com/education/bonus-hunting-for-existing-sportsbook-users)
- Not all boosts are +EV. A BetMGM SGP boost to +250 was still about 11% -EV vs an estimated fair of +320, and competitors' unboosted prices were better. Suggested threshold: at least 7% EV. Boost maximums are typically $100 or less. [ANECDOTE/PRACTITIONER] — [TheLines](https://www.thelines.com/sportsbook-promo-odds-boost-fool-bettors-nfl-mlb-nba-nhl-2023/)
- Action Network also has guides on promo value, live promo rankings and boost approaches. These are practitioner content and were not fetched in full. — [JuiceReel promo rankings](https://juicereel.beehiiv.com/p/ranking-live-sportsbook-promotions); [Establish The Run on odds boosts](https://establishtherun.com/how-to-approach-odds-boosts/)

### Inferences
- **Mechanism.** Boosts are marketing spend priced by a marketing team, often against the book's own (vigged) line. Parlay and SGP boosts are frequently still -EV because the base SGP hold is large. Single-leg main-market boosts and "max $X" profit-boost tokens on near-fair markets are usually +EV.
- **Bonus bets (stake not returned).** Standard practitioner math, from my general knowledge rather than a source fetched here: convert by betting the bonus on a long price, around +300 to +500, and hedging at a sharp or exchange-like book. This typically recovers 65% to 80% of face value. Treat the number as a [PRACTITIONER] figure needing verification; the Action Network page did not give it.
- **Profit-boost token math, testable automatically.**
  - EV per $1 = p_fair × (1 + b × (1 + boost)) − 1, where b is the decimal profit and boost is the percentage (e.g., 0.5 for 50%).
  - Use the token on the market where the book's own price is closest to Pinnacle Shin fair, that is, the lowest-hold main line at that book. Higher odds increase the dollar value of the boost per stake and the variance, but not the EV percentage once vig is held equal.
  - The bot can compute this from existing snapshots: rank, per book, the current lowest-vig market against Pinnacle.
- **Annual value (rough inference, not sourced).** A disciplined Arizona bettor with about 8 books, taking one +EV boost per book per NFL/CFB weekend at roughly $25 to $50 maximum and about 10% EV, earns about $2.5 to $5 per boost. That is about $20 to $40 per weekend, or roughly $500 to $1,000 per football season. Profit-boost tokens and reload bonus bets for "rec-looking" accounts may add similar amounts.
  - Edge size is large but capacity is tiny. The main value is camouflage (Q7): promo-heavy, recreational-looking activity extends the life of accounts used for the model's bets.
  - Promo limits and promo bans ("promo abuse" flags) are a distinct, separate risk from betting limits.
- **Automation.** Boost menus are not in the Odds API. Detection needs per-book scraping (likely against terms of service) or manual check-in. The bot's role is to price a boost the user types in, as a fast EV calculator against the Pinnacle/Shin fair price it already computes.

### Gaps
- No Arizona-specific inventory of current recurring promos (e.g., DK/FD/theScore NFL boosts and "no sweat" tokens) was gathered.
- No measured fraction of boosts that are +EV across a large sample was found.
- No credible annual-value figure was found. The estimate above is an inference.
- Arizona-specific promo rules were not checked. Arizona historically restricted "free bet" advertising and let operators deduct promotional credits from taxable revenue within a capped schedule.

## Q6. Market-specific soft spots: early-week openers, overnight lines, Saturday-morning college moves, line freezes during news

### Takeaway
I found no rigorous public study of US-book soft spots by day or time for football. The user's own hourly snapshot archive (history/odds_*.json.gz, plus the 2020-25 historical odds) is better data than anything public. These are hypotheses to pre-register, ranked by mechanism strength.

### Cited Findings
- Books suspend a lot during college games (uptime 65%+; DK 86%), and margins differ widely by book (above 5% at all, above 8% at BetRivers). [INDUSTRY/vendor data] — [iGaming Business, EKG data](https://igamingbusiness.com/sports-betting/market-making-opportunities-2026/)
- Market makers named for college: Circa (and BetCRIS for "other US sports"). Retail books follow them. [PRACTITIONER] — [SCCG](https://sccgmanagement.com/areas-of-expertise/2024/1/10/market-maker-vs-retail-sportsbook-business-models-and-the-impact-of-price-discovery/)
- Some offshore books partner with touts or shows to "hold the line" after a public pick is released. This shows that tout-driven moves are a known microstructure event that books manage manually. [ANECDOTE] — [RAS/BetOnline Substack](https://raspicks.substack.com/p/ras-live-release-show-partners-with)
- A bad BetCris NBA line was taken down after bettors hit it: an example of manual error correction at a market maker. [ANECDOTE] — [Legal Sports Report](https://www.legalsportsreport.com/43632/betcris-bad-nba-betting-line/)

### Inferences, as testable rules for the user's archive
- **H6a: first-mover soft openers.** For NFL look-ahead or Sunday-evening openers, identify which book posts first, then measure that book's opening line vs the Pinnacle line 24 h later. A book that posts before Pinnacle has settled is guessing.
  - Rule to log: bet the first-posting soft book's side when its number differs from Pinnacle's first available number by at least 1.0 point (spread), or by at least 3% no-vig ML.
  - Opener vs ratings was already tested. This is opener vs the first sharp number, a different comparison.
  - Decay: minutes to hours. Limits on openers are low, and early bettors are profiled.
- **H6b: overnight staleness.** Between about 01:00 and 08:00 AZ time, US trading desks are thin while Pinnacle keeps moving.
  - Metric: the share of snapshots where a soft book is at least 2% EV vs Pinnacle Shin, by hour of day, NFL vs CFB.
  - If the 02:00-07:00 share is at least 2 times the daytime share, run a dedicated overnight watch.
- **H6c: Saturday-morning college moves.** Sharp college money moves Pinnacle and Circa early Saturday. Soft books with low college attention (likely the Kambi cluster) may stay stale 1 to 2 h.
  - Measure Pinnacle moves of at least 1 point between 12:00 and 18:00 UTC on Saturdays, and soft-book catch-up by snapshot count. This is the Q2 test restricted to that window.
- **H6d: news freezes.** When news_llm.jsonl first logs a QB or star signal, check snapshots in the next 1 to 2 h. Does each book suspend (market missing), move, or keep the old price? A book that keeps the old price is the stale-news target.
  - The pre-registered news promotion rule (news_rules v1) gates using news for picks. This test is logging only, so it is compatible.
- **H6e: tout or public-pick steam.** Picks_log already timestamps published picks. Measure soft-book moves in the 30 minutes after a high-follower pick vs Pinnacle.
  - If soft books move toward the pick and Pinnacle does not, the soft book has gone off-market against the pick side, which creates value on the opposite side.

### Gaps
- No public measurements of day-of-week or time-of-day softness at US books for football were found.
- No documentation of US-book "line freeze" policies was found.

## Q7. Account longevity: how books profile and limit, and what pros do to extend account life

### Takeaway
CLV is the main limiting signal. Books also restrict by sport and market where the bettor has not shown "skill," and raise limits for recreational bettors. Once flagged, a bettor may be held to about 10% of a new account's limits. Regulators have barely looked: Massachusetts held a roundtable in May 2024, and every operator declined to attend.
- Practical longevity tactics are practitioner lore, not evidence.

### Cited Findings
- The Massachusetts Gaming Commission held the first public US roundtable on betting limits on 2024-05-21, and all Massachusetts operators declined to attend. [INDUSTRY] — [Closing Line Substack](https://closingline.substack.com/p/sportsbook-limits-get-a-closer-look)
  - CLV is described as the main metric. Repeated CLV gets accounts limited.
  - Books restrict markets where bettors haven't shown skill and raise limits for recreational bettors.
  - A flagged sharp may get 10% of a new account's limits.
  - Unabated co-founder Jack Andrews said DraftKings limited him so severely he could bet only $27 on the Super Bowl coin flip.
- Retail books use "lower betting limits" and "higher market holds." They may limit or close winners, and they review large bets case by case on the customer's wider spending behavior and perceived knowledge. [PRACTITIONER] — [SCCG](https://sccgmanagement.com/areas-of-expertise/2024/1/10/market-maker-vs-retail-sportsbook-business-models-and-the-impact-of-price-discovery/)
- DraftKings may limit boost maximums to $5 for some accounts. [ANECDOTE] — [Action Network](https://www.actionnetwork.com/education/bonus-hunting-for-existing-sportsbook-users)
- Other industry and vendor content describes operator-side sharp detection tools: CLV, bet timing, stake patterns and market selection. This is vendor marketing, not evidence of what DK or FD actually do. [PRACTITIONER/marketing] — [Track360 operator guide](https://track360.io/blog/sportsbook-bet-limiting-sharp-player-liability-operator-guide-2026); [SourceCodeLab](https://sourcecodelab.co/?p=113896); [BetSmart](https://betsmart.beehiiv.com/p/sportsbooks-limit-winning-bettors)

### Inferences, as practical rules and cautions within terms
- **The CLV paradox for this system.** The system's own success metric (CLV vs Pinnacle) is the books' primary limiting signal. Every passing paper track that goes live will shorten account life roughly in proportion to its CLV.
  - Allocate "CLV budget" per book: send the highest-CLV, lowest-capacity bets (e.g., CFB shop-vs-sharp, receptions props) to the books most tolerant of winners.
  - Consider whether Circa Arizona (if online) and exchange-like venues (Kalshi, whose Arizona status is contested per the repo notes) can carry volume that soft books won't.
- **Market choice is profiled.** Books limit by sport and market, so a props-and-college-alt-line bettor at a soft book looks sharp quickly. Mixing in main-line NFL sides at near-fair prices dilutes the profile but costs about 2% to 4.5% hold on the filler. Treat filler as a measurable cost per week, and compare it with the expected life extension.
- **Timing is the most visible tell.** Betting seconds after a Pinnacle move, on stale prices (the Q2/Q6 rules), is the classic arbitrage and steam fingerprint. If such rules go live, cap them to a small share of each account's bets.
- **Promos as camouflage.** Participating in boosts, SGP promos and bonus bets makes an account look recreational, and boosts are themselves +EV (Q5). Taking only +EV boosts is selective in its own way, and some books flag "promo-only" behavior.
- **Sizing.** Rounded, consistent stakes ($50, $100) look recreational. Stakes at exactly the maximum, or odd Kelly-sized amounts, look sharp. Practitioner lore, unverified.
- **Stay within law and terms.** Do not use other people's accounts (beards). That violates terms of service and Arizona regulations on account ownership. Stick to the bettor's own accounts at multiple licensed books.

### Gaps
- No published data was found on limit rates by book, the share of limited accounts, or the time to limit as a function of CLV in the US.
- Arizona's ADG position on limiting winners was not found.
- Whether any Arizona book (e.g., Circa if launched) offers a published minimum-bet-to-lose guarantee is unknown. Some states and books have discussed such guarantees. Not verified.

## Summary table of candidate edges, ranked by evidence plus fit to the user's stack

| # | Edge | Mechanism | Evidence | Decay | Detection with current data | Limit risk |
|---|---|---|---|---|---|---|
| 1 | +EV boosts and profit tokens | Marketing gives up price | Anecdotes (7-20% EV examples) | Per promo; steady supply | Manual input → EV vs Pinnacle Shin | Promo limits ($5-$100 max) |
| 2 | Post-move stale soft price (NFL + CFB) | Retail books follow market makers with lag | Practitioner; no measured lag | Minutes | Hourly snapshots catch few; needs event-driven 5-min polls | High (timing fingerprint) |
| 3 | Kambi cluster lag (BetRivers, Desert Diamond, Bally Bet) | One vendor feed, global trading priorities | Platform facts confirmed; lag unmeasured | Minutes-hours | Cluster-identity test plus move catch-up test on snapshots | Medium |
| 4 | College alt-ladder shape mismatch | NFL-shaped increments applied to CFB | None public; hypothesis | Slow (structural) | cfb_dist.json vs alt rungs; needs alt-market calls | Medium-high |
| 5 | Selling through dead numbers on alts | Flat per-half-point pricing | None public; complements the buy_costs finding | Slow | margins.py vs historical alt data (2023-25) | Medium |
| 6 | College correlated SGPs (fav cover + over) | Correlation under-taxed in CFB | Peer-reviewed 2018 correlation; no returns | Slow until noticed | CFBD joint-probability table; SGP quotes manual | High |
| 7 | Cross-book SGP shopping | Engines disagree on correlation | One anecdote (+250 to +291 spread) | Unknown | Manual SGP quotes only | Medium |
| 8 | First-mover openers vs first sharp number | Books guessing before Pinnacle settles | Hypothesis | Hours | Snapshot archive, first-post timestamps | High (openers have tiny limits) |
| 9 | Overnight / Saturday-morning staleness | Thin desks | Hypothesis | Hours | Hour-of-day EV share on snapshots | Medium |
| 10 | Circa Arizona as sharp reference and account | New Arizona license 2026-07-27 | Regulatory fact | n/a | Watch ADG and the Odds API bookmaker list | Low (Circa tolerates winners) |
