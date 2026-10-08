# Exchange and Prediction-Market Edges for an Arizona NFL/CFB Bettor (as of 2026-10-07)

Scope: venues, fees, incentive programs, documented mispricings, cross-venue arbitrage, market-maker economics, and DFS pick'em, all from the point of view of an automated Arizona-based system. These are already tested in the repo and are not repeated here: taker price after fees vs Pinnacle no-vig (paper record 1-5), and simulated Kalshi maker bids at Pinnacle fair minus 3c.

Evidence-quality tags:
- **[A]** primary document: venue rules, fee schedule, court filing, peer-reviewed or working paper.
- **[B]** reputable trade or news report.
- **[C]** affiliate or review site, or secondary summary.
- **[H]** headline or search snippet only, not read in full.

---

## 1. Which venues are open to Arizona residents, their fees, incentives and API access?

### Takeaway
Kalshi's Arizona status is legally unstable. Arizona brought a 20-count criminal case in March 2026, and Kalshi won a federal injunction in May 2026. The Ninth Circuit's Aug 28, 2026 *KalshiEX v. Assad* ruling (Nevada) held that sports contracts are not protected by federal law. Arizona is now asking the court to lift the injunction, but the ruling's mandate is paused while Kalshi seeks en banc rehearing. Of the newer venues, ProphetX (CFTC DCM) is listed as available in Arizona. Novig blocks Arizona, Sporttrade left Arizona, and Polymarket US availability in Arizona is unconfirmed. Kalshi's fee schedule effective July 7, 2026 charges NFL and NCAAF game markets both a taker fee (0.07) and a maker fee (0.0175). Combo makers pay double the maker fee.

### Cited Findings
**Kalshi: legal status in Arizona**
- March 2026: AG Kris Mayes filed a 20-count criminal information against Kalshi in Maricopa County Superior Court for operating an illegal gambling business and for election wagering. It was the first criminal case any U.S. state has brought against Kalshi. [B] — [Yogonet, 2026-09-02](https://www.yogonet.com/international/news/2026/09/02/126190-arizona-ninth-circuit-ruling-backs-state-authority-over-kalshi-sports-contracts-but-election-dispute-remains)
- May 2026: U.S. District Judge Michael Liburdi held that the Commodity Exchange Act preempts Arizona's gambling statutes as applied to Kalshi. He enjoined Mayes and the Arizona Department of Gaming from enforcing the charges. [B] — [Yogonet](https://www.yogonet.com/international/news/2026/09/02/126190-arizona-ninth-circuit-ruling-backs-state-authority-over-kalshi-sports-contracts-but-election-dispute-remains); [Arizona Mirror, 2026-09-25](https://azmirror.com/2026/09/25/after-9th-circuit-ruling-mayes-pushes-to-revive-her-gambling-case-against-kalshi/)
- Aug 28, 2026: In *KalshiEX, LLC v. Assad* (Nevada; opinion by Judge Ryan Nelson, unanimous panel), the Ninth Circuit held that the CEA does not stop states from applying gambling law to Kalshi's online sports-event contracts. Its reasoning: whether a game occurs is an "occurrence", but who wins is not. Quote: "Congress did not take a wrecking ball to all sports gambling regulations built up over decades." The election-contract question was remanded. [A] — [Ninth Circuit opinion 25-7516](https://cdn.ca9.uscourts.gov/datastore/opinions/2026/08/28/25-7516.pdf); [B] [Arizona Mirror](https://azmirror.com/2026/09/25/after-9th-circuit-ruling-mayes-pushes-to-revive-her-gambling-case-against-kalshi/); [Reason](https://reason.com/2026/09/01/kalshi-says-its-a-prediction-market-the-9th-circuit-says-its-gambling/)
- Sept 2026: Arizona asked the Ninth Circuit to dissolve the Liburdi injunction ("Assad is now the law of this Circuit"). The CFTC asked the court to wait until Kalshi's en banc petition is decided. Under the appellate rules, issuance of the Assad mandate is automatically paused while that petition is pending. [B] — [Arizona Mirror, 2026-09-25](https://azmirror.com/2026/09/25/after-9th-circuit-ruling-mayes-pushes-to-revive-her-gambling-case-against-kalshi/)
- April 2026: the U.S. government sued Arizona, Connecticut and Illinois, claiming exclusive CFTC jurisdiction over event contracts. [B] — [Yogonet](https://www.yogonet.com/international/news/2026/09/02/126190-arizona-ninth-circuit-ruling-backs-state-authority-over-kalshi-sports-contracts-but-election-dispute-remains)
- July 9, 2026: Arizona Mirror headline: "Hobbs bars Arizona state employees from using government info to bet on prediction markets." [H] — [Arizona Mirror](https://azmirror.com/2026/09/25/after-9th-circuit-ruling-mayes-pushes-to-revive-her-gambling-case-against-kalshi/)
- **Unverified:** None of the sources I read says whether Kalshi currently accepts trades from users located in Arizona. Because the Liburdi injunction is still in force, Kalshi presumably still serves Arizona, but I did not confirm it.

**Kalshi: fees (fee schedule effective July 7, 2026)** [A] — [Kalshi fee schedule PDF](https://kalshi.com/docs/kalshi-fee-schedule.pdf?)
- Taker fee = roundup(M × 0.07 × C × P × (1−P)), with default M = 1.
- Maker fee = roundup(M × 0.0175 × C × P × (1−P)). The default maker M is 0 unless the market is listed in the Non-Standard Fees table.
- Sports game markets carry maker multiplier 1. Listed series include KXNFLGAME, KXNCAAFGAME, KXNBA, KXNHL, KXMLBGAME and others.
- Combos/multivariate (KXMVE): maker multiplier 2, taker multiplier 1. The row excludes "uncorrelated NFL combos", and the schedule gives no separate rate for those.
- Before April 2025, makers paid nothing. Kalshi began charging makers after April 2025. [A] — [Bürgi, Deng & Whelan, "Makers and Takers"](https://www.karlwhelan.com/Papers/Kalshi.pdf)
- S&P/Nasdaq range markets use a 0.035 taker coefficient. [C] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)

**Kalshi: incentive programs**
- **Liquidity Incentive Program (LIP)** [A] — [Kalshi Help: Liquidity Incentive Program](https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program)
  - Runs through Jan 1, 2027. Any market may be eligible, and active reward periods are shown on each market page.
  - Pays $1–$1,000 per market per day, in periods of up to 31 days.
  - The order book is sampled once per second at a random moment.
  - Target Size is between 100 and 20,000 contracts. Each side must reach Target Size for a snapshot to count.
  - The Reference Price is the first level where cumulative depth reaches Target/5.
  - Orders at or better than the Reference Price get multiplier 1.0. Orders behind it get Discount Factor^ticks.
  - Snapshot score = your yes-side share plus your no-side share (maximum 2.0).
  - Payout = (your score / total score) × period reward × (counted snapshots / total snapshots).
  - Minimum payout is $1. The SSN/tax threshold applies.
  - Eligible: U.S. members, excluding affiliates and IB/FCM customers.
- **Sports Prop Combo Component-Leg LIP** [A] — [Kalshi Help](https://help.kalshi.com/en/articles/17184676-sports-prop-combo-market-component-legs-liquidity-incentive-program)
  - The pool (CLIFF) is 25% of total fees on each combo market, scaled by the fraction of legs that are eligible.
  - Score = ∏(1+Fᵢ) − 1, where Fᵢ is your share of eligible maker volume in each component leg.
  - Minimum score 0.01. For live events, only maker volume after the scheduled start counts.
  - Paid monthly as bonus credit. No opt-in is required.
- **Sportsbook Hedging Rebate**: 100% rebate of taker and RFQ fees for members hedging sportsbook liabilities who trade more than 300,000 hedging contracts per month. Runs from Feb 23, 2026 to Feb 1, 2027. Members under a Market Maker Agreement are excluded. It is not relevant to an individual. [B] — [Gaming America](https://gamingamerica.com/news/1006858/kalshi-launches-prediction-market-rebate-program-for-sports-event-contracts)
- **Designated Market Maker / Liquidity Provider programs**: these require a Market Maker Agreement. Designated makers get reduced fees and must quote two-sided with 98% uptime per hour. Kalshi calls the program "highly selective". Rewards are allocated by auction among designated providers. [C] — [Turbine, citing Kalshi](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)

**Polymarket US**
- The page says "Sports event trading is the only market currently available at Polymarket in the US." The venue is a DCM under a November 2025 CFTC amended order. Arizona is named among six states that "may get access later" (with NV, MA, TN, IL, CT); the page does not confirm Arizona availability. [C] — [SailGP prediction-markets guide, 2026-09-17](https://sailgp.com/prediction-markets/polymarket/legal)
- Fees, per secondary sources:
  - Kairos: Polymarket US uses a single theta of 0.06 (max $1.50 per 100 shares at 50c). Makers pay no fee and receive a 25% rebate of the matched taker fee.
  - International Polymarket sports: theta 0.05, raised from 0.03 in July 2026.
  - The Kairos page contradicts itself internally.
  - [C] — [Kairos](https://kairos.trade/compare/polymarket-fees)
- Turbine, citing Polymarket docs, gives a different picture:
  - Global Polymarket has category-based taker fees since March 30, 2026, about 0.75% near 50/50 for sports.
  - Makers pay 0% and receive 25% of taker fees outside crypto.
  - Global liquidity rewards score every minute and pay daily.
  - For Polymarket US it mentions only a "small fixed maker rebate", with no figure.
  - [C] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)
- **Conflict:** Polymarket US fee figures differ between secondary sources. Check docs.polymarket.us directly.

**ProphetX**
- A CFTC-approved DCM and DCO, "no longer only a sweepstakes exchange". Listed as available in all states except Nevada, including Arizona. Peer-to-peer. Fees were not stated on the page I read. [C] — [OddsAssist, ProphetX states (modified 2026-09-05)](https://oddsassist.com/prediction-markets/prophetx-states/)

**Novig**
- CFTC DCM. It dropped the sweepstakes model in August 2026. Available in 47 states plus DC; **restricted in Arizona**, Michigan and Nevada. Peer-to-peer limit orders across ML, spread, totals and props, advertised as having "no traditional vig". [C] — [Legal Sports Report, updated 2026-10-05](https://www.legalsportsreport.com/prediction-markets/novig-promo-code/)
- Reported valuation: $2B. [H] — [Northeast Times, 2026-09-30](https://northeasttimes.com/2026/09/30/sports-prediction-market-novig-open-in-pennsylvania-is-now-valued-at-2-billion/)

**Sporttrade**
- Left state-licensed sports betting in NJ, AZ, CO, IA and VA to pivot to prediction markets. Arizona users had until June 25, 2026 to withdraw. The structure and Arizona availability of its new product are not stated. [B] — [Gambling.com, 2026-05-19](https://www.gambling.com/us/news/sporttrade-exits-us-sports-betting-market-in-pivot-to-prediction-markets)

**Crypto.com and Underdog Predict**
- Dec 5, 2025: the Arizona Department of Gaming issued Underdog a Notice of Violation and Intent to Revoke its fantasy-contest license. The grounds were that Underdog is a technology provider to Crypto.com's event contracts, which the department called "illegal conduct in Arizona". Underdog Predict itself does not allow event-contract trading in Arizona. [B] — [Legal Sports Report, 2025-12-15](https://www.legalsportsreport.com/?p=249145)

**Robinhood**
- Its sports contracts are powered by Kalshi. [B] — [SBC Americas, 2025-03-17](https://sbcamericas.com/2025/03/17/robinhood-kalshi-sports-event-contracts/)
- Fee per contract and Arizona availability were not verified.

### Inferences
- **Kalshi break-even arithmetic at P = 0.50:**
  - Taker fee is 1.75c per contract, 3.5% of the stake.
  - Maker fee is 0.4375c per contract, which rounds up; on single contracts that is effectively 1c.
  - A resting bid at fair − 3c therefore keeps at most about 2.56c of edge before adverse selection, and less on small orders because of rounding.
  - Practical rule: fees should be computed on the batch size actually posted, not per contract.
- **Combos are expensive to make on Kalshi.** Maker multiplier 2 means 0.875c per contract at 50c. A combo-making strategy must clear roughly 1c per contract plus adverse selection.
- **Legal-risk handling:** If the Ninth Circuit denies en banc and lifts the Liburdi injunction, Kalshi could geofence Arizona with little notice. Live money on Kalshi or Robinhood should be treated as exposed to a sudden lockout.
  - Suggested rule: hold no more than the bankroll you would accept being frozen for 30 days or more.
  - Suggested automation: watch PACER/CourtListener for the docket and pause Kalshi tracks automatically when the mandate issues.
- **ProphetX is the most clearly Arizona-available new peer-to-peer venue.** Its prices should be logged with the same depth and timestamps as Kalshi (the system already logs ProphetX).

### Gaps
- Whether Kalshi, Robinhood, Polymarket US and Crypto.com currently accept Arizona-located users. The sources I read do not say directly. Check each venue's state list or geolocation help page.
- ProphetX commission or fee schedule and API terms.
- Robinhood's per-contract fee. I recall $0.01 per contract plus exchange fees, but this is unverified and goes here as a gap.
- The exact Polymarket US fee schedule and US liquidity-reward program. Secondary sources conflict.
- Whether NFL and NCAAF game markets currently have active LIP reward periods, and their dollar sizes. These are shown per market page; scrape them.
- Status of the Underdog license revocation appeal. PrizePicks' current Arizona status was not found (see section 5).

---

## 2. Documented mispricings on exchanges and prediction markets

### Takeaway
The evidence of systematic mispricing is in three places, all on Kalshi and none NFL-specific:
1. A classic favorite-longshot bias. Contracts at 10c or less lose more than 60%, while contracts at 50c and above earn small positive returns. The bias is weakening in 2025.
2. Mispricing in the final 10 minutes before expiry: prices are too compressed, so favorites are underpriced late.
3. Combo/parlay overpricing of about 3% per leg beyond 4 legs. Two-to-four-leg combos are priced near fair.

Sub-1c combos lost buyers 86% of stake.

### Cited Findings
**Favorite-longshot bias: Bürgi, Deng & Whelan, "Makers and Takers"** [A] — [Whelan PDF](https://www.karlwhelan.com/Papers/Kalshi.pdf); [CESifo WP](https://www.ifo.de/en/cesifo/publications/2025/working-paper/makers-and-takers-economics-kalshi-prediction-market); [VoxEU summary](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market)
- Sample:
  - 46,282 contracts from 12,403 events, 313,972 prices, covering 2021 to April 2025.
  - Only contracts with at least $1,000 volume and a final spread of 20c or less were included.
  - Sports markets (from Jan 2025) are included but not broken out, and excluding them does not change the results.
- Returns:
  - Average return is about −20%.
  - Contracts at 10c or less lose more than 60% on average.
  - Contracts at 50c and above earn small positive returns. Above 70c, post-fee returns are small but significant.
  - Example: a 95c contract that wins 98% of the time returns +3.1% pre-fee.
- By side:
  - Makers average −9.64% and takers −31.46%.
  - Makers buying at 50c or above earn about +2.6%.
  - The bias is present for both sides and is stronger for takers.
- The bias is weakening: the 2025 Mincer-Zarnowitz price coefficient is 0.021 (SE 0.011), versus 0.034 (SE 0.005) for the full sample.
- Unbiasedness is rejected on every day from 0 to 10 days before close.

**Near-expiry miscalibration and combo pricing: arXiv 2607.14430, "Prices, Probabilities, and Parlays"** [A, preprint] — [arXiv](https://arxiv.org/html/2607.14430v1)
- Sample: about 23M Kalshi trades on NBA, MLB and NHL moneylines, March to May 2026.
- Prices are near-calibrated 30–240 minutes before expiry.
- In the final 0–10 minutes, Platt slopes are NBA 1.62, MLB 2.05 and NHL about 4.56. A slope above 1 means true probabilities are more extreme than prices. In the NHL, a 0.40 contract near expiry wins close to 0%.
- Combo overpricing ratio R = executed price / independence product, over 12,639 cross-game parlays:

| Legs | Median R |
|---|---|
| 2 | 0.991 |
| 3 | 0.995 |
| 4 | 0.999 |
| 5 | 1.005 |
| 6 | 1.013 |
| 7 | 1.037 |
| 8 | 1.066 |
| 10 | 1.223 |

- Overpricing grows about 2.9% per leg (log-linear, R² ≈ 0.94).
- Win rates run 2–10 percentage points below price: a 0.30 parlay wins about 24%, a 0.40 parlay about 30%.
- The paper has no NFL or college data and no sportsbook comparison.

**Combo volume and losses**
- Combos were over 68% of Kalshi's $3.7B record-Sunday notional, but taker stake that day was only $90.7M.
- Gaming America estimates the implied edge at about 2.5% on 2-leg combos and about 25% on combos of more than 10 legs.
- On Sept 3, 2026, Kalshi cut the minimum combo price from $0.01 to $0.0001. Since then, users have lost $4.6M, or 86% of stake, on combos priced below 1c.
- The average Sunday combo now has more than 10 legs, up from 6 in May.
- [B/C] — [RG.org, 2026-10-07](https://rg.org/news/gambling-industry/kalshi-record-volume-longshot-parlays-house-edge), citing Gaming America/Predict Charts and CNBC/Dune.
- CNBC: combos were 58% of September volume but under 13% of transactions. [B, via RG.org] — [RG.org](https://rg.org/news/gambling-industry/kalshi-record-volume-longshot-parlays-house-edge)
- NFL Week 1 2026 (Citizens analysis): Kalshi combos had about 8% worse implied vig than FanDuel and DraftKings before fees. [B] — [Covers, 2026-09-16](https://www.covers.com/industry/kalshi-edges-fanduel-draftkings-sportsbooks-in-nfl-week-1-pricing-sept-16-2026)

**Single-market pricing vs sportsbooks**
- Implied vig on NFL Week 1 2026 moneylines and totals combined: Kalshi 4.32%, FanDuel 4.44%, DraftKings 4.51%. In 2025 Kalshi was worse at 4.84%. Pricing data is from Citizens. [B] — [Covers](https://www.covers.com/industry/kalshi-edges-fanduel-draftkings-sportsbooks-in-nfl-week-1-pricing-sept-16-2026)
- Headline: "Kalshi Beats the Books on NFL Totals, Loses to FanDuel on Moneylines". [H] — [Bitcoin.com](https://news.bitcoin.com/igaming/kalshi-beats-the-books-on-nfl-totals-loses-to-fanduel-on-moneylines/)

### Inferences
These are proposed testable rules. All are untested on NFL or CFB.

- **R-FLB (sell longshots, buy favorites as maker).**
  - Rule: rest bids only on the ≥ 0.70 side of NFL and NCAAF game markets, at Pinnacle Shin fair − 1c, with a 1c maker fee buffer.
  - Never buy Yes below 0.15.
  - Prior evidence: makers at 50c or above earned +2.6% (Whelan).
  - Test: paper-trade for one season. Grade on realized ROI after fees and on CLV against Pinnacle close.
  - Pass criterion: ROI > 0 at p < 0.05 over ≥ 300 fills.
  - Data: Kalshi public trades for KXNFLGAME and KXNCAAFGAME (to simulate fills, as `kalshi_maker.py` already does).
- **R-LATE (late-game compression).**
  - Rule: in the final 10 minutes of game clock, compare the Kalshi in-game price with a win-probability model (nflfastR `vegas_wp` is available in the PBP data).
  - Buy the favorite if the model probability exceeds the price by 4c or more after fees.
  - The NBA/MLB/NHL evidence of Platt slope above 1 suggests favorites are underpriced late.
  - Data: Kalshi trade prints with timestamps aligned to nflverse PBP `game_seconds_remaining`.
  - Test this offline first on archived 2025–26 Kalshi trades. This is a historical backtest, so no new data feed is needed if the trade history API covers past markets.
- **R-COMBO-SELL.**
  - The edge for combos sits on the side that sells 5+ leg combos, which is what Kalshi's RFQ makers and market makers do. A small automated system could respond to RFQs if the API allows it.
  - Quote price: the product of leg no-vig probabilities (Pinnacle Shin) × (1 + 0.03 × max(0, legs − 4)), plus 1c for the maker fee at multiplier 2.
  - Correlation (same-game legs) must be modeled. `margins.py` and `margin_total.py` give joint spread and total distributions for same-game NFL legs.
  - Test: paper-log RFQs received and simulated fills.
  - Risk: tail exposure on many-leg parlays. Use a hard cap on worst-case loss per slate.
- **Avoid:** buying combos of 5+ legs, and any contract below 10c. Both are documented negative-EV zones.

### Gaps
- No NFL- or NCAAF-specific calibration study of Kalshi, Polymarket or ProphetX was found. The near-expiry and combo studies cover NBA, MLB and NHL only.
- No documented study of retail overreaction after news on exchanges, or of slow updating in player props, specials or college markets.
- Settlement-rule quirks (overtime inclusion, ties, push handling on spread contracts, voids on postponement) were not researched in primary rulebooks. Read Kalshi's KXNFLSPREAD/KXNFLGAME contract terms and Polymarket US market rules before any spread-ladder strategy.
- Whether Kalshi's RFQ/combo API is open to retail accounts, or only to registered market makers.

---

## 3. Arbitrage and middles between exchanges and sportsbooks

### Takeaway
I found no rigorous, sourced measurement of how often NFL arbitrage or middles between exchanges and sportsbooks appear at real size. Kalshi's two-way vig on NFL moneylines and totals (about 4.3%) is similar to major books, so pure arbitrage needs one stale side. The cleanest testable ideas are internal-consistency checks: monotonicity within Kalshi's own spread ladders, and Kalshi ladder vs sportsbook alt-line middles priced with the repo's key-number margin model.

### Cited Findings
- Kalshi's NFL implied vig is close to the major books: 4.32% vs FanDuel 4.44% and DraftKings 4.51% in Week 1 2026. [B] — [Covers](https://www.covers.com/industry/kalshi-edges-fanduel-draftkings-sportsbooks-in-nfl-week-1-pricing-sept-16-2026)
- Kalshi set new single-day volume records in NFL Week 1 2026: $2.46B notional, about $1.5B of it combos. Kalshi listed 15 of 16 Week 1 games. NFL was 67.2% of league trading across nine tracked operators (Aldrin Research). [B] — [Covers](https://www.covers.com/industry/kalshi-edges-fanduel-draftkings-sportsbooks-in-nfl-week-1-pricing-sept-16-2026)
- A 30-day Polymarket/Kalshi vs sportsbook arbitrage log exists for esports. It is practitioner-grade and I did not read it in full. [H] — [DEV Community](https://dev.to/dozor/polymarket-kalshi-arbitrage-vs-sportsbooks-30-days-of-esports-data-1dc2)
- Dimers headline: "Kalshi's record NFL weekend shows why bettors need to shop around." [H] — [Dimers](https://www.dimers.com/industry/news/kalshi-record-nfl-week-1-2026)

### Inferences
- **R-LADDER (internal arbitrage).**
  - Kalshi "win by more than X" contracts must be monotone in X. That is, the price of Yes(>X) must be at least the price of Yes(>X+k).
  - Log every orderbook snapshot. Flag any case where the ask on the lower strike is below the bid on the higher strike, after both taker fees.
  - Each flag is riskless only if settlement rules match across the strikes, including overtime and push handling.
  - Data: Kalshi spread-series orderbooks, hourly in the existing `pipeline watch`. The ticker family name is unverified.
- **R-MIDDLE (exchange ladder vs sportsbook alt line).**
  - Use `margins.py` to price the probability that the margin lands between the Kalshi strike and a sportsbook alt spread at an allowed Arizona book.
  - Bet when P(middle) × payout − fees − vig > 0, with EV of at least 2% against the Pinnacle-implied distribution.
  - This reuses existing infrastructure (`buy_costs.json`, `my_books.json`).
- **Measure frequency before building.**
  - Add a counter to the hourly watch: how many cross-venue two-way arbitrages exist with ≥ $100 fillable on both sides.
  - With a 1-5 taker record and vig similar to the books, expect them to be rare.

### Gaps
- No primary measurement of NFL arbitrage or middle frequency at size between Kalshi, Polymarket US or ProphetX and Arizona sportsbooks.
- The Kalshi spread-ladder contract specifications (strikes, overtime inclusion) were not verified.
- Liquidity (top-of-book size) on NFL spread ladders and on college markets was not found.

---

## 4. Market-maker economics: who provides liquidity, spreads, adverse selection, and what a small maker earns

### Takeaway
Most Kalshi liquidity comes from retail-scale makers: about 95% of matches come from 2,000 or more small makers, plus institutions such as Susquehanna. Makers lose less than takers, and on Kalshi's historical data they make money only on the favorite side. Adverse selection is concentrated in one-sided "single-name" flow. Kalshi's own in-house market maker reportedly loses money. For a small automated maker, the realistic extra income is LIP rewards: up to $1,000 per market per day, split pro rata by time-weighted resting depth near the top of book.

### Cited Findings
- Susquehanna became Kalshi's first dedicated institutional market maker in 2024 and reportedly added about 30x the liquidity in select markets. About 95% of Kalshi bid matches come from 2,000+ smaller makers and about 5% from institutions (co-founder Luana Lopes Lara, via eFinancialCareers). [C] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)
- Kalshi's in-house market maker is "not profitable" and handles under 6% of sports making volume (Bloomberg via Yahoo, 2026). [C, secondary] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)
- A Stanford Law study of 41.6M Kalshi trades found that makers earn about twice as much per contract in single-name markets, and that one-sided, toxic flow predicted maker losses in those same markets. [C, secondary] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)
- Polymarket (Akey et al., via CNBC): 68.8% of users lost money, and winning accounts tend to provide liquidity with resting orders. [C, secondary] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)
- Kalshi makers averaged −9.64% vs takers −31.46%, and makers buying at 50c or above earned +2.6% (pre-2025 data, when makers paid no fees). [A] — [Whelan](https://www.karlwhelan.com/Papers/Kalshi.pdf)
- LIP mechanics: see section 1. Kalshi samples once per second, requires Target Size depth on both sides, and pays $1–$1,000 per market per day. [A] — [Kalshi Help](https://help.kalshi.com/en/articles/13823851-liquidity-incentive-program)
- Kalshi filed prop-market liquidity incentives to "fuel sports combos". [H] — [NextPredict](https://nextpredict.io/market-news/industry/kalshi-files-prop-market-liquidity-incentives/)
- A CFTC rule filing exists, dated Feb 2026. [H] — [CFTC filing](https://www.cftc.gov/sites/default/files/filings/orgrules/26/02/rules02112639183.pdf)
- An August 2025 "Volume and Liquidity Incentive Program" PDF exists. [H] — [InGame PDF](https://www.ingame.com/wp-content/uploads/2025/08/Volume-and-Liquidity-Incentive-Program-August-2025.pdf)

### Inferences
- **LIP earnings model for a small maker.**
  - Reward ≈ (your depth share near the reference price, averaged over snapshots) × daily pool × fraction of valid snapshots.
  - Example: a $500/day NFL market where you keep 5% of qualifying depth on both sides about 80% of the time earns about $20/day per market.
  - Capital needed is roughly Target Size × price per side.
  - All of this is illustrative. Actual pools for NFL markets must be scraped.
- **R-LIP (reward farming with fair-value guardrails).**
  - Rest two-sided quotes at Pinnacle Shin fair ± (maker fee + 1c), sized to reach Target Size/5, in NFL and NCAAF markets with active LIP periods.
  - Pull quotes within 60 seconds of a Pinnacle move of 1.5c or more, or of a QB/injury signal from `news_llm.py`. Use the existing hourly-watch plus news infrastructure, but this needs a faster loop.
  - Accounting: track three numbers separately: spread P&L, adverse-selection P&L (markout 5 and 60 minutes after each fill against Pinnacle), and LIP rewards.
  - Pass criterion: total > 0 over ≥ 4 weeks, with adverse-selection loss smaller than rewards.
  - Data needed:
    - LIP reward-period amounts per market (market pages or API).
    - Kalshi orderbook WebSocket.
    - A Pinnacle feed at 1-minute resolution or better. The Odds API at hourly resolution is too slow for the cancel logic.
- **R-PROP-COMBO-LEG LIP.**
  - The component-leg program pays 25% of combo fees to makers in prop legs, counting live-event volume only.
  - Given combo volume of $2.5B notional on a Sunday, this pool could be significant. A small maker quoting NFL player-prop legs in-game could test its share.
  - Risk: in-game adverse selection is severe.
- Kalshi's maker fee for sports (0.0175 × P(1−P)) took about 0.44c at 50c out of the pre-2025 maker edge documented by Whelan. The +2.6% favorite-side maker return should be re-checked net of the fee.

### Gaps
- No public figure for the actual LIP pools on NFL or college markets, or for total payouts to date.
- No figures for typical NFL bid-ask spreads in cents on Kalshi, Polymarket US or ProphetX.
- Polymarket US maker rebate and reward program details (conflicting secondary sources).
- Primary versions of the Stanford Law and Bloomberg items were not read. Treat their numbers as secondary.

---

## 5. DFS pick'em (PrizePicks, Underdog): +EV strategy and Arizona status

### Takeaway
The +EV method for pick'em is well documented. Compare each fixed pick'em line with the no-vig probability from sharp sportsbook props, and play only legs that beat the slip's break-even rate: about 58% per leg for a 2-pick Power (-137) and about 54% for a 5-pick Flex (-119). Arizona status is uncertain and hostile. In December 2025 the Arizona Department of Gaming moved to revoke Underdog's fantasy license over its Crypto.com predictions tie-up. Earlier headlines report that Arizona approved Underdog's pick'em and that Underdog switched to a peer-to-peer format in Arizona.

### Cited Findings
- PrizePicks 5-pick Flex pays 10x for 5/5 and 2x for 4/5. Break-even is about 54% per leg (about -119) for 5-Flex and about 58% per leg (about -137) for 2-Power. [C] — [BettingPros, 2026-08-07](https://www.bettingpros.com/articles/how-to-make-money-on-prizepicks-picking-the-right-slip-size-props/)
- Optimizer method: rank props against market consensus from sharp books and prediction markets. Keep only props above the slip break-even. Exclude "demons", "goblins" and juiced lines. [C] — [BettingPros](https://www.bettingpros.com/articles/how-to-make-money-on-prizepicks-picking-the-right-slip-size-props/); [BettingPros optimizer launch](https://www.bettingpros.com/articles/bettingpros-launches-new-prizepicks-and-underdog-optimizers/)
- Dec 5, 2025: the Arizona Department of Gaming issued Underdog a Notice of Violation and Intent to Revoke its fantasy-contest license under A.R.S. §5-1209(A)(2), (12). The license is not stayed on appeal without a court order. [B] — [Legal Sports Report](https://www.legalsportsreport.com/?p=249145)
- 2023: Arizona and Mississippi questioned fantasy pick'em games (letters). [H] — [LSR](https://www.legalsportsreport.com/148266/letters-pickem-arizona-mississippi/); [SBC Americas](https://sbcamericas.com/2023/11/02/arizona-mississippi-fantasy-letter/)
- Sept 2024: "Arizona approves Underdog Fantasy's pick'em-style contest." [H] — [SBC Americas](https://sbcamericas.com/2024/09/11/arizona-underdog-fantasys-pickem/)
- "Underdog Changes To Peer-To-Peer Fantasy Sports In Arizona." [H] — [LSR](https://www.legalsportsreport.com/?p=169027)

### Inferences
- **R-PICKEM.**
  - Rule: for each NFL pick'em leg, compute the Shin no-vig probability from at least 2 sharp prop quotes (the repo already pulls props through `odds.fetch_event_props`).
  - Include a leg only if p ≥ 0.58 for a 2-Power slip, or p ≥ 0.55 for a 5-Flex slip (0.54 break-even plus 1 point of margin).
  - Grade with `player_stats.stat_value`.
  - Extra data: the PrizePicks and Underdog board lines. There is no official API; third-party feeds exist, for example [SharpAPI](https://sharpapi.io/sportsbooks/prizepicks-odds-api) [H].
- Before any real play, confirm which pick'em products currently operate legally for Arizona-located users. PrizePicks' Arizona status was not found, and Underdog's license may be revoked.

### Gaps
- PrizePicks' Arizona status in 2026.
- The outcome of Underdog's Arizona appeal.
- Underdog payout tables and peer-to-peer format details in Arizona.
- Independent, audited ROI studies of pick'em optimizers. None were found; all sources are affiliates or tool vendors.

---

## 6. Quantitative studies and practitioner reports of returns

### Takeaway
The quantitative literature on these venues is thin and recent, and none of it is NFL- or CFB-specific:
- Whelan et al. on Kalshi 2021–25: favorite-longshot bias, makers outperform takers.
- arXiv 2607.14430: late-game compression and combo overpricing.
- A Stanford Law maker study and an Akey et al. Polymarket study, both secondary.
- Trade-press vig comparisons from Citizens via Covers, and Gaming America.

Every proposed edge here (favorite-side making, late-game favorites, selling combos, LIP farming, ladder/middle checks, pick'em vs sharp props) is a hypothesis that should be pre-registered in a new rules JSON with a version number, as `betting_rules.json` does.

### Cited Findings
- Kalshi returns by price bucket and maker/taker split. [A] — [Whelan](https://www.karlwhelan.com/Papers/Kalshi.pdf)
- Kalshi calibration in the last 10 minutes and combo overpricing per leg. [A, preprint] — [arXiv 2607.14430](https://arxiv.org/html/2607.14430v1)
- Combo edge estimates: about 2.5% on 2 legs, about 25% on more than 10 legs; buyers of sub-1c combos lost 86%. [B/C] — [RG.org](https://rg.org/news/gambling-industry/kalshi-record-volume-longshot-parlays-house-edge)
- Polymarket: 68.8% of users lose, and winners skew toward makers. [C] — [Turbine](https://www.turbinefi.com/blog/how-to-market-make-prediction-markets-2026)
- A related working paper on Kalshi exists (GWU Forecasting Program 2026-001). Not read. [H] — [GWU PDF](https://www2.gwu.edu/~forcpgm/2026-001.pdf)
- UCD WP2025_19 is likely a version of the Whelan paper. Not read. [H] — [UCD](https://www.ucd.ie/economics/t4media/WP2025_19.pdf)

### Inferences
Suggested pre-registration order, cheapest data first:

| Order | Rule | Why this position |
|---|---|---|
| 1 | R-LADDER and R-MIDDLE counters | Logging only; uses existing hourly watch and `margins.py` |
| 2 | R-FLB favorite-side maker variant | Modification of the existing `kalshi_maker.py` simulation: restrict to ≥ 0.70 and net the 0.0175 fee with rounding |
| 3 | R-LATE | Historical backtest on Kalshi trade history aligned with nflverse PBP |
| 4 | R-LIP | Needs WebSocket and a faster sharp feed; highest engineering cost |
| 5 | R-COMBO-SELL | Needs RFQ access confirmation and a correlation model |
| 6 | R-PICKEM | Only after Arizona legality is confirmed |

- The existing taker track's 1-5 record (6 bets) is far too small to infer anything. This is consistent with literature showing takers do worst.

### Gaps
- No NFL or college football exchange-efficiency study.
- No practitioner P&L disclosures, for example from Unabated or Circles Off. Podcasts and X/Bluesky threads were not reached; one X article was blocked by robots.txt.
- No primary data on ProphetX fees, liquidity or API.
