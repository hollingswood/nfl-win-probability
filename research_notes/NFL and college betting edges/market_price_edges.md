# Price-based and market-structure edges in NFL and FBS betting for a US (Arizona) retail bettor, 2025-2026

Scope note: researched 2026-10-04 with about 45 web searches and fetches. Many sources are industry blogs and data vendors, not peer-reviewed work, and each is labeled as such. Vendor numbers (latency in particular) are marketing claims, not independent measurements. I found almost no public, dated, quantified evidence on CFB-specific market structure (middle frequency, soft-book copy latency for Circa/BetCRIS). Those gaps are listed explicitly.

## 1. Softest markets, books and times; stale lines and steam; soft-book copy latency

### Takeaway
The best-documented structural softness is timing. Openers are posted at low limits by a market-setting book and then shaped by sharp action. For CFB, Circa opens the weekly market on Sunday. Soft US books follow sharp moves with a lag that a vendor puts at about 15-90 seconds. Exploiting that lag is real but burns accounts quickly, so experienced bettors say most stale-line pickoffs aren't worth the account. Your own measured 3-7-days-out CFB edge fits this: the market is least shaped while limits are low.

### Cited Findings
- Circa Sports sets the first weekly college football line every Sunday during the season. "When lines are opened, the limits are usually lower and the book that opens them is willing to be first." For NBA/NHL/MLB the first line "still happens offshore." The same piece says "the most respected professional bettors will dictate what the betting line is, not the bookmaker." — [Circa: How Lines Are Set](https://www.circalasvegas.com/blog/sportsbook/from-the-experts/article/how-lines-are-set-an-oddsmakers-perspective/)
- Vendor-stated update speeds after a market move (no methodology published): Pinnacle 1-5 s; Bet365 5-15 s; DraftKings 15-60 s; FanDuel 15-60 s; BetMGM 15-45 s; Caesars 30-90 s. "Recreational books like DraftKings may take 30+ seconds to follow. That window is where value exists." The Odds API's collection interval is about 30-60 s, so a pipeline polling The Odds API cannot see sub-minute stale windows. — [SharpAPI: How live is real-time odds data](https://sharpapi.io/learn/how-live-is-real-time-odds-data)
- Data Golf (golf matchups, 106k+ matchups since 2019) gives the cleanest public evidence of a sharp-vs-soft hierarchy. Pinnacle closing odds show a "1 to 1 relationship between expected and actual ROI": a 4% Pinnacle-implied edge realized about 4%. DraftKings and Bet365 showed "only a slightly positive relationship." BetOnline, 5Dimes and Bovada "largely copy Pinnacle or Bet365." Blind-betting returns: Pinnacle -3.31%, Bet365 -7.08%, DraftKings -7.01%. Pinnacle's closing odds incorporate about 50-70% of competing books' opening information. Betting other books' prices with Pinnacle *opening* odds as fair yielded about +1.8% ROI at a 0% threshold, falling at higher thresholds. — [Data Golf: How sharp are bookmakers](https://datagolf.com/how-sharp-are-bookmakers)
- Futures and awards markets left open at some books during standalone (TNF/SNF/MNF) games, and at "a couple of books" even during NFL Sundays, are a known stale-line source. The author advises against betting them: "It's not worth burning an account betting stale lines." Books may void, limit or close. — [Betting Life newsletter: Picking off stale lines](https://betting-life-newsletter.beehiiv.com/p/picking-off-stale-lines)
- Same-game-parlay and curated "specials" are a negative-structure market. In one example, FanDuel offered a TNF "Quick Bet" at +330 while the same legs built manually priced at +370 to +377, costing the bettor "more than 10% of winnings." — [JuiceReel: Ranking live sportsbook promotions](https://juicereel.beehiiv.com/p/ranking-live-sportsbook-promotions)
- Cross-venue arbitrage on prediction markets (Polymarket/Kalshi/PredictIt) "typically exist[s] only for a few seconds, at best a few minutes, and transaction costs significantly reduce potential profits." Polymarket tends to lead price discovery thanks to higher liquidity, especially in the final hours. — [QuantPedia: Systematic edges in prediction markets](https://quantpedia.com/systematic-edges-in-prediction-markets/)

### Inferences
- Polling at 30-60 s through The Odds API, you structurally cannot capture steam-lag edges (15-90 s windows). Your edge will come from slower-decaying mispricings: early-week CFB, derivatives, and soft-book openers that sit for hours. This matches your finding that the edge is largest 3-7 days out. Chasing steam would need a WebSocket feed and sub-10-second execution.
- The Data Golf result (Pinnacle open-as-fair gives about +1.8% vs other books) is the closest public analogue to your CFB result (+2-4% CLV vs Pinnacle close). The magnitude is plausible, and it shrinks if you raise thresholds, which suggests diminishing returns to filtering.
- Circa opens CFB on Sunday, with BetCRIS (also Latin-American-origin, CFB-heavy) and Pinnacle following. So the Sunday-to-Tuesday window, before soft-book limits rise and before sharp shaping is complete, is the most plausible softest window for FBS. Measure it in your odds history; I found no published quantification.

### Gaps
- No independent, published measurement of how long DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, Fanatics or Hard Rock take to copy *Circa or BetCRIS* moves (as opposed to Pinnacle). The SharpAPI table is vendor-asserted, not football-specific, and doesn't separate pregame from live.
- No public data found on which soft book is "most stale" for small-conference CFB (MAC, Sun Belt, C-USA) or on overnight/weekday softness by hour.
- Alt-line, derivative (1H/1Q/team total) and player-prop softness lacked quantified, dated public evidence beyond anecdote.

## 2. Sharp reference books (NFL vs CFB) and building a fair line: weighting and devig methods

### Takeaway
Pinnacle remains the empirically validated sharp anchor (Data Golf). Circa is the CFB market setter, and BetCRIS (a Latin-American market-making brand) is now legally available in Arizona. The devig literature agrees that the basic multiplicative method is the least accurate. Power, odds-ratio, margin-weights-proportional-to-odds and Shin all model the favorite-longshot bias and should be preferred for lopsided moneylines. For two-way markets near 50/50 (spreads, totals) the method barely matters.

### Cited Findings
- Devig methods in the R `implied` package: basic/multiplicative (pi = ri / sum r), additive, power (pi = ri^(1/k), Buchdahl's "logarithmic"), odds ratio (Keith Cheung), margin weights proportional to odds (WPO, from Buchdahl's "Wisdom of the Crowds"), Shin (insider-trader model, estimates insider share Z), Balanced Books, and Jensen-Shannon. The vignette says basic "tend[s] to be the least accurate of the methods in this package." WPO, OR and power explicitly account for favorite-longshot bias. No single method is declared best in all cases. — [implied package vignette (CRAN)](https://ftp.fau.de/cran/web/packages/implied/vignettes/introduction.html)
- Data Golf used plain proportional (multiplicative) devig and validated it by checking that blind margin-free betting returns about zero. They note it is adequate "for odds in the 35%-65% range where favorite-longshot bias doesn't significantly apply." — [Data Golf](https://datagolf.com/how-sharp-are-bookmakers)
- BetOnline (and historically 5Dimes and Bovada) "largely copy Pinnacle or Bet365" rather than price independently. — [Data Golf](https://datagolf.com/how-sharp-are-bookmakers)
- Circa sets the first weekly CFB line every Sunday. — [Circa](https://www.circalasvegas.com/blog/sportsbook/from-the-experts/article/how-lines-are-set-an-oddsmakers-perspective/)
- BetCRIS launched a legal Arizona sportsbook the week of 2025-01-27, through Plannatech's B2C license with the San Carlos Apache Tribal Gaming Enterprise. Arizona was chosen partly for its 10% tax and proximity to California. The COO aims for it to be "the third or fourth app on anybody's phone." The article gives no limits or sharp-friendliness policy. — [Covers, 2025-01-27](https://www.covers.com/industry/betcris-brand-returns-to-usa-with-sportsbook-launch-in-arizona-jan-27-2025)
- Circa Sports has been approved for an Arizona license (partner: San Juan Southern Paiute Tribe), with launch expected early 2027. — [LegalSportsReport Arizona page, Sept 2026](https://www.legalsportsreport.com/arizona/)
- Your project notes (CLAUDE.md) say circasports and bookmaker never appear in The Odds API feed. Your v3 fair price is the median of Pinnacle, LowVig and BetOnline. Data Golf's finding that BetOnline copies Pinnacle implies the median is not three independent opinions.

### Inferences
- For CFB, add BetCRIS's Arizona book as a reference quote if you can get its prices (it is legal and local). Add Circa once it launches in AZ (early 2027) or through a feed that carries Circa Nevada. Circa opens CFB, so Circa early-week prices are probably the most informative reference for FBS.
- Weighting: copier books (BetOnline, LowVig) add little independent information to Pinnacle, so a median of {Pinnacle, LowVig, BetOnline} is close to "Pinnacle with noise." Better options: weight by independent sharpness (e.g., Pinnacle at full weight, plus Circa/BetCRIS when available) or fit weights by regressing closing no-vig on each book's earlier no-vig in your own history. This is untested and is a suggestion for your data.
- For moneylines beyond about -250/+210, switch from multiplicative to power or Shin devig. The multiplicative method overstates the longshot's fair probability, which makes soft-book underdog prices look falsely +EV and favorite prices falsely −EV.

### Gaps
- I found no dated public head-to-head test of devig accuracy specifically on NFL or CFB moneylines (e.g., log loss of Shin vs power vs multiplicative on Pinnacle closes). This is a cheap test to run on your own 2020-25 historical odds.
- No public evidence ranking Circa vs Pinnacle vs BetCRIS vs Bookmaker accuracy specifically for FBS small-conference games.
- Sporttrade, Novig and ProphetX as reference prices: no published accuracy studies found.

## 3. US prediction markets and exchanges: fees, liquidity, mispricing, maker/taker, arbs vs books

### Takeaway
Fees are now probability-dependent everywhere and peak near 50¢, exactly where NFL/CFB spreads and totals trade. That makes takers expensive (Kalshi about 1.75¢ per contract at 50¢, about 3.5% of the stake) and makers privileged. The best empirical study of Kalshi (pre-sports data) shows makers lose far less than takers and a strong favorite-longshot bias. ProphetX and Novig are low- or zero-commission and available in most states.

### Cited Findings
- **Kalshi fee formula (effective 2025-07-01, announced 2025-06-26):** taker fee = round_up(0.07 × C × P × (1−P)); maker fee = round_up(0.0175 × C × P × (1−P)) where applicable. The maximum taker fee is 1.75¢ per contract at P = 0.50; at 10¢ or 90¢ it is 0.63¢ taker and 0.16¢ maker. Makers pay nothing to cancel resting orders. — [River Markets: Kalshi fees (updated 2026-08-09)](https://www.rivermarkets.com/insights/kalshi-fees.html); [InGame, 2025-06](https://www.ingame.com/kalshis-change-may-increase-fee-sports-traders/)
- The 2025 Kalshi fee change raised sports fees about 6.5% overall and about 15% on NBA/NHL/NFL, because those markets cluster near 50/50. The old fee was a flat 0.25¢ per $1 contract. — [InGame](https://www.ingame.com/kalshis-change-may-increase-fee-sports-traders/)
- Sports were about 87% of $39.7B traded on Kalshi in the year to February 2026. — [River Markets](https://www.rivermarkets.com/insights/kalshi-fees.html)
- **Polymarket sports fees (effective 2026-03-30 for newly launched markets):** peak effective rate 0.75% at 50% probability, declining toward 0/100. Example: a $50 trade at 50% costs about $0.38 (previously $0.22). Makers receive a 25% rebate funded by taker fees. — [iGaming Business](https://igamingbusiness.com/prediction-markets/polymarket-sports-fee-hike-2026/)
- **Robinhood (contracts now on Rothera, the Robinhood–Susquehanna JV exchange, formerly LedgerX):** a new pricing model from June 2026 with fees that "vary depending on contract prices, but will never exceed $.01 per contract," lower away from 50¢. Gold members get up to 50% off. — [Covers, 2026-06-05](https://www.covers.com/industry/robinhood-launches-world-cup-contracts-through-rothera-june-5-2026)
- **ProphetX:** "up to ~1% commission on net winnings," 0% on props. Available in 40+ states including Arizona. Offers NFL and NCAA football, and allows in-game exit/trading. **Novig:** described as commission-free P2P with spreads, totals, props, parlays and futures, in 35+ states; Arizona availability not confirmed by the source. Novig bets lock once matched, with no mid-game exit. (LSR, updated 2026-09-08.) — [LegalSportsReport: ProphetX vs Novig](https://www.legalsportsreport.com/dfs-sites/prophetx-vs-novig/)
- ProphetX raised $35M (2026-07-28) for B2B sports prediction markets. — [Axios](https://www.axios.com/pro/fintech-deals/2026/07/28/prophetx-35m-b2b-prediction-markets)
- **Kalshi maker vs taker economics (Bürgi, Deng, Whelan, UCD; data 2021-11 to 2025-04, 46,282 contracts, 313k price observations; mostly pre-sports):** average return makers −9.64% vs takers −31.46% (post-fee). Contracts at 1-10¢ lose over 60% on average. Contracts at 50¢+ earn small positive returns (about +2.6% for makers). The average pre-fee return is about −20%. The bias holds across categories and volume levels. Sports markets, which began in early 2025, are explicitly not driving these results. — [GWU working paper 2026-001](https://www2.gwu.edu/~forcpgm/2026-001.pdf); [VoxEU summary](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market)
- Sharp bettors limited by state-licensed books are moving to federally regulated prediction markets (Kalshi, Polymarket) and exchanges "where peer-to-peer matching eliminates practical betting caps" (Covers, September 2025). — [Covers](https://www.covers.com/industry/massachusetts-limiting-sports-betting-prediction-markets-exchanges-sharps-box-september-2025)

### Inferences
- Fee drag at typical NFL spread/total prices (about 50¢): Kalshi taker 1.75¢ per $0.50 stake is about 3.5% of stake, roughly equal to the vig on a −107 book line. Kalshi maker at about 0.44¢ is about 0.9%. Polymarket US at 50% is about 0.75%, ProphetX is about 1% of *net winnings* (≈0.5% of stake at even money), Robinhood is ≤1¢ on a 50¢ contract (≤2%), and Novig is 0% stated. So for a bettor with a fair line, **resting maker orders slightly better than the sharp no-vig price on Kalshi/Polymarket, or taking on Novig/ProphetX**, beats crossing the Kalshi spread as a taker.
- "Market making as a bettor": post limit orders at the Pinnacle no-vig ± a small margin. Your edge is then capturing the spread plus rebates, and the risk is adverse selection when news hits (resting orders get picked off by faster traders). Pulling quotes on news and before kickoff is essential. Your news_llm and hourly watch could drive quote cancellation, since cancels are free on Kalshi.
- The Whelan et al. favorite-longshot finding on Kalshi suggests longshot sides (big-underdog moneylines, at ≤15¢) are systematically overpriced by retail takers. Selling them as a maker, or buying the favorite side, is the structural trade. Verify on sports contracts specifically before trusting it, because the paper's data is mostly non-sports.
- Exchange ↔ sportsbook arbitrage is feasible in principle: buy the Kalshi or Polymarket side when it is below the soft book's devigged opposite. Windows are short, and transaction costs consume much of the edge (QuantPedia).

### Gaps
- No published sports-specific (NFL/CFB) study of Kalshi or Polymarket mispricing vs Pinnacle was found. Whelan et al. exclude or downplay sports.
- Liquidity depth figures (e.g., $ available within 1¢ of the mid on a CFB G5 spread on Kalshi/Novig/ProphetX) were not found.
- Sporttrade, Crypto.com and Underdog prediction-market fee schedules were not retrieved. Novig's monetization and Arizona availability are unconfirmed.

## 4. Promotions, boosts and bonus bets: EV math and annual extraction in Arizona

### Takeaway
Unconditioned profit boosts and bonus bets are reliably +EV. With optimal play a bonus bet converts to cash at about 50-70% of face. Parlay insurance and curated specials are usually −EV. I found no credible dated estimate of total annual promo value for an Arizona bettor.

### Cited Findings
- An unconditioned 10% profit boost on a $100 bet at −110 is worth about +$4.55 EV ("win $9.09 more ... half the time"). "Any straight up boost is good value." — [JuiceReel](https://juicereel.beehiiv.com/p/ranking-live-sportsbook-promotions)
- Bonus bet conversion: the author uses 50% as a base estimate and says expert strategy gets "often ... less than 70%" of face. — [JuiceReel](https://juicereel.beehiiv.com/p/ranking-live-sportsbook-promotions)
- A 4-leg parlay insurance example remained −EV (−$7.20 vs −$27.20 on $160 without insurance): it reduces the house edge but does not flip it. Curated bet specials are the worst category (>10% of winnings lost vs building the legs manually). — [JuiceReel](https://juicereel.beehiiv.com/p/ranking-live-sportsbook-promotions)
- Arizona has 12 live online books: Bally Bet, bet365, Betcris, BetMGM, BetRivers, Caesars, Desert Diamond, DraftKings, Fanatics, FanDuel, Hard Rock Bet, theScore Bet. Circa is approved for early 2027. Online tax is 10%. — [LegalSportsReport Arizona](https://www.legalsportsreport.com/arizona/)

### Inferences
- Bonus-bet conversion math: the stake isn't returned, so the value of a bonus bet B at decimal odds d is B × (d − 1) × p. Long odds maximize the retained fraction. Hedging on a sharp-priced venue (Novig, ProphetX, Pinnacle-like) locks in about 65-75% when the hedge side is low-vig. A zero- or low-fee exchange hedge, such as Novig or a Kalshi maker order, raises the locked fraction compared with hedging at another soft book.
- A profit boost of b (e.g., 0.25) at decimal odds d has EV per unit stake = p_fair × (1 + (d − 1)(1 + b)) − 1, where p_fair comes from the sharp no-vig line. Apply it to the bet with the best no-vig edge against your fair line, not to a random favorite. Your existing fair-line engine is the right tool to pick boost targets.
- Annual extraction: with 12+ AZ books (plus bet365 and Desert Diamond, which you may not be tracking), sign-up bonuses are one-time. Recurring value comes from daily or weekly boosts, which are capped (often $25-$50 max stake), so the per-boost EV is small. This is an inference; no quantified source was found.

### Gaps
- No credible, dated estimate of annual promo profit per bettor in Arizona (or any state) was found. Figures in affiliate articles are marketing.
- Book-by-book boost frequency and max-stake caps for AZ in 2025-26 were not found.

## 5. Middles, arbitrage, hedging, key-number scalping, buying points

### Takeaway
NFL margins remain concentrated at 3 (about 15%) and 7 (about 9%), so moving across 3 or 7 carries most of the value, and middling 2.5/3.5 is the classic play. Totals key numbers are much flatter (each about 3-4%). No public quantification of middle availability in CFB was found. Your own measured buy costs (buy_costs.json: buys never +EV) agree with the general point that books overprice half-points onto key numbers.

### Cited Findings
- NFL: "three occurs in around 15% of all games, and seven in about 9%." After the 2015 PAT change (2015-2023 vs 2006-2014), 6- and 8-point games rose 2.4% combined and 5-point games rose 1.4%; 3 and 7 remain "king" with slightly diminished dominance. — [Covers: NFL key numbers 2025-26](https://www.covers.com/nfl/key-numbers)
- 3-point margins occurred in 14.8% of games since 2003. Getting −2.5 and +3.5 when the market is at 3 is "extremely important." — [Action Network: NFL key numbers](https://www.actionnetwork.com/nfl/nfl-key-betting-numbers-spread-margins-of-victory-line-value)
- NFL totals key numbers (2014-2023): 41 at 3.5%, 43 at 3.2%, 44 at 3.7%, 47 at 3.1%, 51 at 3.9%. The 43-51 range is the meaningful zone. — [Covers](https://www.covers.com/nfl/key-numbers)
- Cross-venue arbs "exist only for a few seconds, at best a few minutes," and costs reduce profits. — [QuantPedia](https://quantpedia.com/systematic-edges-in-prediction-markets/)

### Inferences
- Middle EV at 2.5/3.5 in the NFL: the middle hits about 15% of the time (the 3 frequency). With both legs at −110 the cost of a miss is about $10 on a $220 outlay, and a hit wins about $200. Break-even hit rate ≈ 10/(10+200) ≈ 4.8%, so a 2.5/3.5 middle at −110/−110 is strongly +EV in principle. The binding constraint is availability: books rarely hang 2.5 and 3.5 simultaneously without juice adjustments, and line movement through 3 is what creates it. Your margins.py (key-number distribution) can price any spread pair exactly.
- CFB middles: CFB margins are less concentrated on 3/7 and spreads are larger and more dispersed across books (especially G5), so price discrepancies of 1.5-3 points are more common but each point is worth less. I found no source quantifying CFB middle frequency; measure it from your historical odds.
- Hedging is never +EV by itself. It is a variance tool, except when converting bonus bets or locking boost EV.

### Gaps
- CFB key-number frequencies and middle availability were not found in public 2024-26 sources.
- Exact per-half-point win-probability values (e.g., 2.5→3 is about X%) were not in the fetched sources. Use your own margins.py estimates.

## 6. Account limits: speed, AZ tolerance, extending account life, legality

### Takeaway
Limiting is officially rare in the US (under 1% of accounts in the regulator data) but concentrated precisely on CLV-positive bettors. Operators told Massachusetts regulators that bettors who "consistently beat the closing line" get lower stake factors. Limited accounts are commonly cut to 1-24% of default limits. Limiting is legal in Arizona. Massachusetts is the leading regulator on transparency rules.

### Cited Findings
- Massachusetts: 0.64% of sports bettor accounts were limited as of December 2024. Of those, 57.6% were restricted to 1-24% of default bet amounts. "Players who consistently beat the closing line are more likely to have a lower stake factor," and losing bettors "are more likely to have a higher stake factor." Limited players rarely reach VIP status. The MGC is considering mandatory notification of limits and semiannual manual reviews of limited accounts (Covers, September 2025). — [Covers](https://www.covers.com/industry/massachusetts-limiting-sports-betting-prediction-markets-exchanges-sharps-box-september-2025)
- MGC Sports Wagering Division head Carrie Torrisi presented operator admissions that CLV-beaters get lower stake factors (reported 2025-10-02). — [CDC Gaming](https://cdcgaming.com/brief/winners-face-betting-limits-losers-made-vips-mass-gaming-commission-learns/)
- In Massachusetts, BetMGM reported about 1% of customers limited, and FanDuel reported that 0.043% of wagers were placed at maximum amounts. Wyoming: "less than 1% of bettors are limited." UK Gambling Commission (July 2025, about 15M accounts): 4% limited (643,779), 62.7% with stake restrictions, and over half of restricted accounts subsequently closed. — [SBC Americas, 2025-07-25](https://sbcamericas.com/2025/07/25/uk-regulator-sports-betting-limits/)
- Timeline: in May 2024 all but one licensed MA sportsbook declined a regulator roundtable on limits. In November 2024 the MGC approved data collection on limit and VIP practices. — [Straight to the Point newsletter](https://straighttothepoint.substack.com/p/take-it-to-the-limit)
- Account-preservation tactics from an unsourced industry blog (low reliability): round stakes, avoid betting immediately at market open, mix in recreational-looking action, spread volume across books (called "the most effective single tactic"), and don't bet only promos. Example drop: from $500 to $22 max on a market. "Some accounts are restricted within a dozen bets, others operate for years." — [DeucesCracked, 2026](https://www.deucescracked.com/blog/sportsbook-account-limits-2026-why-winners-get-limited)

### Inferences
- Your CLV-positive CFB edge (+2-4% vs Pinnacle close) is exactly the signal books profile on. Expect stake factors to fall at the soft AZ books that hang the stale prices. Betting at open, in odd amounts, or only on lines that are about to move are the classic tells.
- Diversification across the 12 AZ books is the main lever. Exchanges (ProphetX in AZ; Novig if available) and prediction markets don't limit winners structurally and are the natural destination once soft books restrict you.
- BetCRIS-AZ and Circa-AZ (2027) are the most likely AZ books to tolerate winners, given their market-maker heritage. This is not confirmed by any source on their AZ limit policy.

### Gaps
- No public data on how many bets or how much $ CLV it takes for specific books (DraftKings, FanDuel, BetMGM, Caesars, etc.) to limit an account, and no AZ-specific tolerance ranking.
- Arizona Department of Gaming has no published limiting data or rules (none found).

## 7. Market efficiency: when closing lines are least efficient (CFB small conferences, early season, bowls); favorite-longshot bias in NFL/CFB moneylines

### Takeaway
The academic evidence is old-ish and mixed. NFL moneylines (2007-2016) show a *reverse* favorite-longshot bias (favorites overbet) that is unstable across seasons. CFB spreads predict less well in bowls than in the regular season. Prediction-market data shows the classic favorite-longshot bias. None of this is a robust standalone strategy, so price comparison against sharp books remains the more reliable edge.

### Cited Findings
- NFL money line market, 2007-2016: inefficiencies exist at open and close. The pattern is a reverse favorite-longshot bias ("bettors tend to overbet the favorite and underbet the underdog"). Favorites in precipitation games were profitable. The inefficiencies "are unpredictable and vary by season." — [Journal of Economic Insight: An Examination of the Money Line Market for NFL Games](https://journalofeconomicinsight.com/index.php/joei/article/view/1335)
- CFB: spreads are more predictive in the regular season than in bowl games. The market is "an inefficient processor of momentum effects, particularly negative momentum," though largely efficient within transaction costs (Cox, Schwartz, Van Ness & Van Ness, Journal of Sports Economics 22(3), 2021). — [IDEAS/RePEc](https://ideas.repec.org/a/sae/jospec/v22y2021i3p251-273.html)
- Data Golf: books' blind-betting returns and sharpness vary widely, and only Pinnacle's closing odds are well calibrated. — [Data Golf](https://datagolf.com/how-sharp-are-bookmakers)
- Prediction markets (Kalshi, 2021-2025, mostly non-sports): strong classic favorite-longshot bias, with 1-10¢ contracts losing over 60%. — [GWU WP 2026-001](https://www2.gwu.edu/~forcpgm/2026-001.pdf)
- Classic football-betting data (12,084 matches, February-May 2017, soccer): favorites −3.64% vs outsiders −26.08% average returns. — [QuantPedia](https://quantpedia.com/systematic-edges-in-prediction-markets/)

### Inferences
- Bowls are a plausible window of lower closing-line efficiency (opt-outs, motivation, long layoffs). Cox et al. support lower predictive power in bowls, though not necessarily an exploitable price edge.
- US sportsbook NFL moneylines may show the reverse bias (favorites overbet), while exchanges and prediction markets show the classic bias (longshots overbet). That contrast creates a potential structural cross-venue trade: buy longshots at sportsbooks and sell longshots on exchanges when both deviate from sharp no-vig. Test it on your data before acting.

### Gaps
- No recent (2020-2026) peer-reviewed study of CFB closing-line efficiency by conference (P4 vs G5) or by season week was found.
- No study of favorite-longshot bias on CFB moneylines specifically.
- No public evidence quantifying early-season CFB closing-line inefficiency.
