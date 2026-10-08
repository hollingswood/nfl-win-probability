# Pro bettor playbooks: how long-term winners make money (NFL/CFB, 2020-2026)

Evidence labels used below: **[1st-hand]** = the bettor's own words in an interview or book, **[journalism]** = a reporter's account, **[summary]** = a third-party book summary or aggregator (weaker), **[claim]** = an unverified assertion by an interested party.

Research scope note: about 35 tool calls. Several primary sources (podcast audio for Unabated/Circles Off/Bet the Process, full text of *Gambler*, *The Logic of Sports Betting*, *Interception*, *Weighing the Odds*) could not be read directly. Where only summaries were available, that is flagged.

---

## 1. Billy Walters and the Computer Group: what methods he describes, and what still applies

### Takeaway
Walters' own system is power ratings (a 90/10 weekly update), player values in points (QB about 7, the best non-QBs 2.5-3, at least 60% of players worth about 0), injuries as the second-biggest edge, aggressive line shopping, and a hard 3%-of-bankroll cap. Scale came from proxies ("beards" and runners), which is the part that is not legally replicable now. Rating, player-value, and bankroll discipline still apply; a lone modeler copying the "model vs. line" approach is what the user's own backtest already shows fails against closing lines.

### Cited Findings
- **[1st-hand, via Covers excerpt of *Gambler*]** Power ratings: each team gets a neutral-field number, and the spread is the difference plus adjustments. Weekly update = 90% old rating + 10% "True Game Performance Level", so only 10% of the new rating comes from last week's game. — [Covers: Walters shares the secrets behind his system](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- **[1st-hand]** Player values: QB "about a touchdown" (more for the best), top non-QBs 2.5-3 points, "at least 60%" of players basically 0. Injuries are the "second-most important factor" for an edge. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- **[1st-hand]** Home field: conventionally 3, but about 2.5 for 1974-2022 and **under 1 point over the last four years**. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- **[1st-hand]** Situational adjustments he still uses: bounceback after losing by 19+ (more at 29+), Super Bowl winner up and loser down for the first 4 games of the next season, turf matching, and road team off Monday Night Football as "one of the biggest downgrades." — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- **[1st-hand]** Execution: accounts at many books; watch Circa, MGM, Caesars and Pinnacle movement; **"bet favorites early and underdogs late"**; don't chase. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- **[1st-hand]** Money management: max 3% of bankroll on one event; bets in half units from 0.5 to 3 units by edge; "start with the assumption that you'll lose it all." — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters). **[summary]** 1 unit = 1% of bankroll, and a 1-2 point edge earns about half a unit. — [Shortform](https://www.shortform.com/blog/billy-walters-sports-betting/). **[summary]** The 3% rule came from poker player Chip Reese. — [SuperSummary](https://www.supersummary.com/gambler/summary/)
- **[summary]** Scale math: a 5% edge on a $1M bet is about $50K expected, versus $500 on $10K. Walters used beard teams that bet "several million dollars a week." — [Shortform](https://www.shortform.com/blog/billy-walters-sports-betting/)
- **[summary]** Computer Group: model by Michael Kent (a former Westinghouse programmer) produced power-rated numbers and reportedly won "more than 60 percent" of bets. Walters moved money through runners and beards. FBI raided 45 locations in Jan 1985; at the 1992 trial the jury acquitted on 64 counts. Sierra Sports Consulting had about 30 staff and **more than 1,600 accounts worldwide**, was raided in 1996, and the case ended in 2002 with $4.6M returned. — [SuperSummary](https://www.supersummary.com/gambler/summary/)
- **[summary/court record]** Insider-trading case (stocks, not sports): convicted on 10 counts, 60-month sentence plus $44.2M; released May 2020; sentence commuted Jan 19, 2021 (the conviction stands). — [SuperSummary](https://www.supersummary.com/gambler/summary/); [Wikipedia](https://en.wikipedia.org/wiki/Billy_Walters_(gambler))

### Inferences
- What still applies to an automated individual: (a) a slow-moving rating (a 10% update is far stickier than most public models); (b) a player-value table in points with most players at 0 and QBs dominant (the repo's `qb_availability` and injury feature already do a version of this); (c) the "favorites early, dogs late" timing rule, which is already in the repo's backlog and testable from `history/odds_*.json`; (d) the 3% cap.
- What doesn't: beards and runners (illegal or ToS-violating in regulated US markets, see §4), and the 1980s-90s soft lines that let a single model win 60%.
- Walters' sub-1-point home field over 2019-2022 is a concrete check for the repo's home-field term.

### Gaps
- I could not find Walters' half-point / key-number percentage table from *Gambler* in any readable source. The Covers excerpt says key numbers matter but gives no values. **Don't cite a specific Walters percentage** without the book.
- "Move and bet" (bet a smaller sharp-known account first to move the market, then hit the other side at the moved number) is widely attributed to Walters, but no page I read describes it. Unverified here.

---

## 2. Modern pros' documented playbooks

### Takeaway
The modern winners are mostly not "better handicappers of NFL sides." They (a) originate in thin, derivative markets (props, golf, early college numbers) where books lack data, (b) trade information and speed (injury filtering, Asian/Pinnacle moves → slower books), (c) run at roughly a 1-2% hold on very large volume, and (d) treat account access as the binding constraint. NFL sides are, by their own account, a small or zero part of their profit.

### Cited Findings
**Rufus Peabody (props, originator)**
- **[journalism]** He builds models for hundreds of props, bets "as much as he can" when he finds value, and is described as an "originator" whose bets move lines that others react to. His operation is teams in the US and UK that share information and split action. For one Super Bowl, the crew split about $170K in cash into backpacks, and total risk was "well into the seven figures." He deposited about $100K into Arizona online books and drove to just inside the state line to bet. "A bettor like Rufus cannot even bet $10,000 at one place." — [The Ringer, 2022](https://www.theringer.com/2022/9/21/23363621/gamblers-super-bowl-props-rufus-peabody-nfl)
- **[1st-hand]** Group earns "over $1 million per year." NFL betting is "a small percentage" of what he makes; he mainly bets NFL props in the playoffs and shifted to golf and NFL as public data made MLB more efficient. "The same model I used to bet baseball then wouldn't make money now." Line shopping: 1-2% per bet "will add up and turn into real money quickly" (even -101/-102 vs. even money). — [TheLines, 2022](https://www.thelines.com/professional-sports-betting-journey-bettor-rufus-peabody-2022/)
- **[1st-hand]** Super Bowl prop tactics: the public "almost always fires on Over and Yes", so he takes Unders/No (e.g., "They always play Yes on the safety"; plans to bet No later). He bets openers only on props with value "that I don't think is gonna last", holds others for game day after public money pushes them, and is "not blasting everything at openers" to avoid moving markets. — [Covers](https://www.covers.com/industry/sharp-bettors-fire-early). **[1st-hand, book side]** Westgate's Kornegay: a 30-cent to $1 gap between book and sharp model draws action in "the first couple of days"; props are 60-65% of Super Bowl handle. — same source

**Gadoon "Spanky" Kyrollos (CFB/CBB, information + automation)**
- **[1st-hand]** Calls it "trading": "We don't watch any games." Custom software logs in, grades, and flags line moves and injury news. The edge is "injuries, numbers, and line movement," plus information shared with sharp groups worldwide. He watches Asian books and takes a number a slower market hasn't moved to yet (example: Asia -7 vs. the world -6). — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- **[1st-hand]** "Beating the closing number... the most important thing." Hold is **about 1-2% of handle**. He **rarely bets NFL ("not profitable for him")** and makes occasional NFL bets only as a courtesy to keep book relationships. Best CFB prices are often gone by the time lines open widely; his window is **Monday about 9-11 a.m. ET**. Injury-news filtering software took "seven, eight years to fine tune." Weather is secondary (extremes only). — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- **[journalism]** Middling across 80+ books (e.g., Steelers +4.5 at one book vs. +2.5 at another), steam ("board cleaners"), "betting robots", hundreds of accounts worldwide, and offshore credit. Books that cut him became "partners" lending him accounts (a ToS/legal problem, see §4). DraftKings cut him after a weekend he held 32%. — [The Ringer, 2019](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)

**Ed Miller & Matthew Davidow (*The Logic of Sports Betting*, *Interception*)**
- **[1st-hand excerpt]** Opening lines carry only the oddsmaker's opinion, so value is easier to find early. Treat post-bet movement as "market agreement" (good) or "resistance" (stop adding and investigate why). Average CLV above half the hold over hundreds of bets is "promising" (e.g., with a 4% hold, moving from a 42% to a 45% break-even is +3%). Applies only to liquid two-way markets; in thin markets an unmoved line means nothing. — [LSR excerpt](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- **[summary]** Market makers discover price at low opening limits; retail books copy settled lines, run higher holds, and limit winners. The weak spots: thin markets, derivatives and props (1H, player props), correlated parlays where allowed, in-play feed errors and timeouts, and promos. — [book summary](https://sobrief-staging.onrender.com/books/the-logic-of-sports-betting)
- **[summary/reviews]** *Interception* (2024) covers the retail tech stack, odds feed providers, and per-customer profiling; reviewers say the focus is keeping accounts open and exploiting automated odds models. — [Goodreads](https://www.goodreads.com/en/book/show/199657133-interception)

**Market-maker benchmark (Pinnacle) / Buchdahl-style CLV**
- **[analysis, 3rd party]** Using Pinnacle as ground truth and betting soft books above Pinnacle's implied probability (European football): 31,247 bets at 3.8% expected vs. 3.6% actual ROI since 2012/13. Since 2023/24, 6,806 bets at 4.2% expected vs. **1.9% actual**, with the author unsure whether Pinnacle's close has degraded or it's variance. — [Networked substack](https://networked.substack.com/p/a-view-from-the-pinnacle)
- **[journalism/advice]** Track at least 200-300 bets; positive CLV of "around 2% or more" against a sharp close is meaningful, while beating a soft-book close tells "very little." — [TheSpread](https://www.thespread.com/?p=550721)

**Promo grinders / matched betting**
- **[journalism]** Matched betting: bet the promo on side A and hedge with cash on B. Colorado grinders reportedly made "tens of thousands" from intro bonuses; profiled grinders came from couponing and credit-card churning. — [Bloomberg, 2022](https://prod.cm.bloomberg.com/news/features/2022-03-30/sports-betting-app-promos-make-bettors-look-for-new-edge)

### Inferences
- Common thread: **pros originate where the book's model is weakest and liquidity is low (props, early CFB numbers), and they take sharp-to-soft price gaps fast**. Both are already in the user's system (props shop, soft vs. Pinnacle). What is less mainstream:
  1. **Avoiding NFL sides entirely** (Spanky, Peabody), so the user's "model vs. Vegas fails" result matches the pros.
  2. **Sequencing bets to protect later prices** (Peabody holds some props until after public money moves them, which is the opposite of "always bet the opener").
  3. **Contrarian prop sides** where the public structurally takes Over/Yes. Testable: for each prop type, compare opener-to-close drift on Over vs. Under.
  4. **Relationship betting / "courtesy" action** to extend account life (§4).
- Haralabos Voulgaris, Starlizard/Tony Bloom, and Priomha (Brendan Poots) are mostly soccer/NBA. I found no first-hand NFL/CFB methods from them in this pass.

### Gaps
- Captain Jack Andrews / Unabated: the podcast pages have only topic lists (NFL simulator, derivative markets, Super Bowl prop arbitrage, NJ exchanges), not transcripts. — [LVA podcast page](https://www.lasvegasadvisor.com/gambling-with-an-edge/podcast-captain-jack-andrews-and-rufus-peabody/). His "originating vs. market-making / pick-off" framing is not verified here.
- Buchdahl's *Weighing the Odds* and his favourite-longshot findings were not read directly. The repo's own devig research (Shin) covers the favourite-longshot point empirically.
- Voulgaris, Starlizard, Priomha (a Chat With Traders episode exists: [Brendan Poots EP 105](https://chatwithtraders.com/episode/105-brendan-poots-how-a-former-punter-pioneered-a-premier-sports-betting-hedge-fund)), Thorp/Kelly specifics: not fetched.
- No credible 2024-26 figures found on promo-grinder income or arb/middle frequencies in US markets.

---

## 3. Information edges: originators vs. followers, news speed, and whether steam-chasing still works

### Takeaway
First-hand accounts say information edges come from **filtering noisy beat-reporter news with software** (Spanky's took 7-8 years) and from **watching faster markets (Asian books, Pinnacle) to bet slower books before they move**. Following originators works only at books that lag. Pros say the windows are short, and books identify followers quickly ("laying 5 on a game that closes at 8 marks you").

### Cited Findings
- **[1st-hand]** Spanky: beat writers at every school make CFB injury news plentiful but noisy, so his team built filtering software ("seven, eight years to fine tune"). Pre-Twitter, he called college newspapers posing as a student to get practice info. — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- **[1st-hand]** Spanky takes Asian-book numbers before North American books catch up (Asia -7 vs. the world -6). — same source
- **[journalism]** Ringer: Spanky's team "watched line movements across books worldwide and used models to get ahead of them". Books call these players "board cleaners." Pinnacle lets certain sharps bet before lines go public and uses their action to tighten odds. — [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)
- **[1st-hand, quoted in journalism]** Alan Denkenson: most people are limited **after about four bets**; "laying 5 on a game that closes at 8" identifies a sharp. Peabody: "If you beat the line you get banned." — [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)
- **[3rd party]** Odds-alert products sell Pinnacle-move notifications so users can act first elsewhere. The author speculates arbers may distort Pinnacle's close. — [Networked](https://networked.substack.com/p/a-view-from-the-pinnacle)
- **[1st-hand]** Andrews: NFL numbers are "hammered into shape" by the weekend. — [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)

### Inferences
- An AI news reader is the modern equivalent of Spanky's filter. Its value depends on **(a) speed vs. the sharp line (not vs. the soft book's line) and (b) precision on who is actually out**. The repo's `news_audit` lead-time vs. 6-hour spread move is exactly the right test. The pros' accounts suggest CFB (more teams, noisier coverage, slower books) is where this pays, not NFL.
- Steam-chasing at soft books is the fastest way to get limited (Denkenson's "four bets"), so its expected value comes from bet count before the limit, not per-bet edge. Treat it as an account-burning strategy and budget accounts accordingly.

### Gaps
- No first-hand source quantified how fast US retail books now follow Pinnacle/Circa moves in 2025-26 (seconds vs. minutes). The user's hourly snapshots cannot measure this; only finer-grained polling can.
- Paid injury newsletters, weather services, and beat-reporter networks used by NFL pros: no named, sourced examples found.

---

## 4. Account management, limits, beards, and exchanges / prediction markets

### Takeaway
Limits are the binding constraint for every documented winner. Legal levers: spread action across many legal books, keep some losing or "courtesy" action, avoid hitting only the softest markets, and use venues that don't limit (Pinnacle is not legal in AZ; exchanges and prediction markets). Proxies and beards, account sharing, and "partner" accounts are named by books as grounds for bans and can violate state law. **Kalshi faces 20 criminal counts from Arizona's AG (March 2026)**, so it is not a clean option for an Arizona resident.

### Cited Findings
- **[1st-hand]** Spanky: limits were his biggest problem by 2019; "Guys taking $500 bets and stuff, that doesn't work." Bettors who "go for the jugular" on props and obscure markets get restricted fastest. He keeps book relationships with courtesy NFL bets ("It doesn't have to be adversarial"). — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- **[journalism]** Example limits: Andrews was capped at $100 on inning props and $30.26 on baseball at DraftKings; a test bettor was cut from $5,000 to $1,000 after three bets. William Hill lists account sharing, betting for third parties, screen scraping and compliance as ban reasons. Denkenson: books can be beaten by "sending people with new names and fresh money" (that is, proxies). — [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)
- **[1st-hand]** Spanky to Wyoming regulators (2024): limiting is "an epidemic" and hits customers who "do their homework." Massachusetts operator data said **under 1% of players** are affected. — [SBC Americas](https://sbcamericas.com/2024/11/25/regulators-wyoming-talk-betting-limits/)
- **[journalism]** Massachusetts (Feb 2026) ruled operators must tell customers why their limits were cut; a NJ bill would require disclosure of limits. — [Covers, May 2026](https://www.covers.com/industry/kalshi-investor-kyle-kuzma-criticizes-draftkings-and-sportsbook-model-may-26-2026)
- **[journalism/claim]** Prediction markets "aren't limiting successful customers," which has drawn pros. Market maker Matt Kalish says ordinary users lose "at a 90%-plus rate" to professional market makers there. — [Covers](https://www.covers.com/industry/kalshi-investor-kyle-kuzma-criticizes-draftkings-and-sportsbook-model-may-26-2026)
- **[analysis]** NFL Week 2 2025 straight-bet vig: Kalshi 4.22%, FanDuel 4.43%, DraftKings 4.50%. Combos/parlays are worse on Kalshi (26.4% vs. about 23.9%). Takers pay higher fees than makers (average $1.62 per 100 contracts). In a survey, 31% chose prediction markets for liquidity/no max bet, and 14% because winners aren't restricted. — [RotoWire](https://www.rotowire.com/article/kalshi-beats-draftkings-fanduel-on-nfl-week-2-pricing-134248)
- **[journalism]** Arizona AG filed 20 criminal counts against Kalshi (Maricopa County, March 2026), including illegal gambling business (college/pro sports, player props) and election wagering. Kalshi says the CFTC preempts state law. The article does not address individual users. — [Covers, Mar 2026](https://www.covers.com/industry/arizona-attorney-general-criminal-charges-at-kalshi-march-17-2026); [Morgan Lewis](https://www.morganlewis.com/pubs/2026/03/arizona-files-first-criminal-charges-against-a-prediction-market). The litigation was still active in Sept 2026 ("Arizona sues Kalshi after Kalshi sued Arizona"): [Front Office Sports](https://frontofficesports.com/article/arizona-sues-kalshi-after-kalshi-sued-arizona/) (headline only; not read).
- **[1st-hand]** Peabody had to be physically in Arizona to bet AZ online books. — [The Ringer 2022](https://www.theringer.com/2022/9/21/23363621/gamblers-super-bowl-props-rufus-peabody-nfl)

### Inferences
- **Do not recommend:** beards, runners, using others' accounts, "partner" accounts, or third-party betting. Books list these as ban grounds, and in regulated states they may breach terms or law. Walters' and Spanky's historical scale relied on these.
- Legal account-life tactics consistent with pros' statements: favor main markets over obscure ones for the volume of bets; keep some non-edge action (Spanky's "courtesy" bets); avoid patterns books flag (always taking the stale number, laying 5 into an 8 close); spread across all legal AZ books; and track per-book "time to limit" as a resource. The repo's paper tracks could log simulated "bets until limit" assumptions.
- Prediction markets: lower straight-bet vig and no limits, but Arizona's criminal case makes this a legal risk for an AZ resident. The repo's existing "paper only" stance on Kalshi is consistent with this.

### Gaps
- No sourced evidence for "limits are higher near game time at US retail books" (common lore). The Ringer article does not say it.
- No data found on how much "recreational camouflage" (parlays, losing bets) extends account life.

---

## 5. Bankroll and variance: realistic numbers

### Takeaway
Documented pros run at about **1-2% hold on very large handle** (Spanky), cap single-event risk at 3% of bankroll (Walters), and see large swings (Peabody's group: +$1M then -$700K in two months). A positive CLV of about 2% against a sharp close over 200-300+ bets is the signal threshold cited. ROI against Pinnacle-as-truth in soccer has ranged from 3.6% down to 1.9% realized.

### Cited Findings
- Spanky: hold about 1-2% of handle. — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor); [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)
- Krackomberger: needs to bet "five and six figures on a weekend" to make a living. — [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)
- Peabody: group earns over $1M/yr; 2010 MLB group was +$1M through June, then -$700K over the next two months. — [TheLines](https://www.thelines.com/professional-sports-betting-journey-bettor-rufus-peabody-2022/)
- Walters: max 3% per event, units of 0.5-3% by edge. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Miller/Davidow: average CLV above half the hold is promising. — [LSR](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- Pinnacle-as-truth EV betting: 3.6% realized ROI (31K bets), but 1.9% (6.8K bets, 2023/24+). — [Networked](https://networked.substack.com/p/a-view-from-the-pinnacle)
- At least 200-300 bets; about 2%+ average CLV vs. a sharp close is meaningful. — [TheSpread](https://www.thespread.com/?p=550721)

### Inferences
- At a 1-2% hold, a bettor needs roughly $5-10M handle per year to clear $100K, which the user's legal AZ accounts will not allow under typical retail limits. So the realistic individual profile is **higher per-bet edge on low-limit soft/derivative markets** until limited, not a pro-style low-margin, high-volume operation. (This inference is derived from the cited hold figure.)

### Gaps
- No first-hand 2023-26 NFL/CFB CLV distributions from named pros were found.

---

## 6. What pros say the public and "mainstream analytics bettors" get wrong

### Takeaway
Recurring pro claims: the public bets close to game time, overbets Over/Yes, chases obscure props that get them limited, relies on the eye test, treats a model's disagreement with the line as edge, and keeps adding to positions the market resists.

### Cited Findings
- Waiting until near game time (Spanky implies this is wrong; the best CFB prices are gone at the open, so bet Monday 9-11 a.m. ET). — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- "Going for the jugular" in props and obscure markets gets you restricted quickly. — same source
- The eye test is excluded; "We don't watch any games." — same source
- The public "almost always fires on Over and Yes", so fade with Unders/No, often later after public money moves the price. — [Covers (Peabody)](https://www.covers.com/industry/sharp-bettors-fire-early)
- Loading up when the market resists instead of investigating why. — [LSR excerpt](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- Treating odds as probability, ignoring that it is a zero-sum contest, and letting promos mask a lack of edge. — [book summary](https://sobrief-staging.onrender.com/books/the-logic-of-sports-betting)
- Overweighting last week: Walters gives the latest game only 10% weight. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Assuming 3-point home field (under 1 point recently, per Walters). — same source
- Using a soft book's close as a CLV benchmark. — [TheSpread](https://www.thespread.com/?p=550721)
- Reusing an old model ("the same model... wouldn't make money now"), since markets adapt. — [TheLines (Peabody)](https://www.thelines.com/professional-sports-betting-journey-bettor-rufus-peabody-2022/)

### Inferences
- Counter-strategies that map to the repo: Over/Yes bias tests in props (Peabody), a timing study of favorites early / dogs late (Walters), Monday-morning CFB execution (Spanky; the repo's cfb_shop could add a Monday 13:00-15:00 UTC window), and a check that the AI news signal has a lead over the sharp line, not just the soft one.

### Gaps
- No direct sourced critiques of "analytics bettors" (EPA models) from named pros were found.

---

## 7. Edges pros publicly say still work for NFL/CFB (2020-2026)

### Takeaway
The edges pros name publicly: **early-week college numbers (Monday morning)**, **injury/news in college** filtered by software, **Super Bowl/playoff props** (contrarian to Over/Yes, staged timing), **derivatives and thin markets** (1H, props, correlated parlays where allowed), **middles across books**, **promos**, and **taking sharp-book moves at slower books**. NFL full-game sides are repeatedly described as near-efficient or unprofitable.

### Cited Findings
- CFB Monday 9-11 a.m. ET window; NFL "not profitable" for Spanky. — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- Super Bowl props: sharps hit gaps of 30 cents to $1 in "the first couple of days"; contrarian Unders/No. — [Covers](https://www.covers.com/industry/sharp-bettors-fire-early)
- Derivatives, props, correlated parlays, in-play feed errors, and thin markets are the soft spots; resistance/agreement reading. — [book summary](https://sobrief-staging.onrender.com/books/the-logic-of-sports-betting); [LSR](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- Middling 2-point gaps across 80+ books (pre-2020 account; still structurally possible). — [The Ringer](https://www.theringer.com/2019/6/5/18644504/sports-betting-bettors-sharps-kicked-out-spanky-william-hill-new-jersey)
- Kalshi straight bets were cheaper than DK/FD in early 2025 NFL weeks (but see the AZ legal status). — [RotoWire](https://www.rotowire.com/article/kalshi-beats-draftkings-fanduel-on-nfl-week-2-pricing-134248)
- Matched betting on promos. — [Bloomberg](https://prod.cm.bloomberg.com/news/features/2022-03-30/sports-betting-app-promos-make-bettors-look-for-new-edge)

### Inferences (strategy matrix for the report writer)
| Strategy | How it makes money | Resources | Scales? | Book response | AZ-legal individual + automation? |
|---|---|---|---|---|---|
| Originating in props/derivatives (Peabody) | Model beats thin, data-poor prices | Per-market models, fast execution | Low limits; needs many books | Fast limits on props | Yes, already partly built (props tracks); add Over/Yes bias and staged timing |
| Early-week CFB (Spanky) | Openers carry only the oddsmaker's opinion | Ratings + Monday-morning execution | Medium (limits low at open) | Moves line; pro identified | Yes; add a Monday morning window to cfb_shop |
| Software news filter (Spanky) | Bet before injury news is priced | Years of tuning; source precision tracking | Limited by speed | Books cut or delay | Yes; news_audit lead-time is the right test |
| Sharp-to-soft lag / steam (Spanky, "board cleaners") | Soft book lags Pinnacle/Asia | Sub-minute polling, many accounts | Short account life | Fastest limiting trigger | Legal; treat accounts as a consumable budget |
| Middles/arbs | Cross-book gaps around key numbers | Many accounts, scanning | Small, account-burning | Flags arbers | Legal; margins.py can price middles |
| Promos/matched betting | Book subsidies | Accounts, hedging calculator | One-off per book | Promo restrictions | Legal; finite |
| Prediction markets / maker bids | Lower vig, no limits, maker rebates | Order management | High liquidity | No limits (claimed) | **Legally contested in AZ** (criminal charges vs. Kalshi); paper only |
| Beards/runners, account sharing (Walters, Spanky's partners) | Bypass limits | Network of people | High historically | Ban, possible legal exposure | **Do not use** |

### Gaps
- No 2025-26 first-hand NFL-specific edge disclosures from named pros were found (pros rarely disclose live edges). Circles Off and Unabated podcast transcripts were not accessible.
- Live-betting NFL/CFB specifics, futures timing, and alt-line/teaser edges from pros: not found in this pass. The repo already tested Wong teasers.
