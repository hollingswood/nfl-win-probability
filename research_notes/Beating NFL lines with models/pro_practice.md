# How Professional NFL/CFB Bettors and Public Modelers Build Models That Beat the Market

Scope note: ~24 searches/fetches. Many primary sources (podcast audio for Bet the Process, Establish The Run, Gambling With an Edge, Business of Betting) only had show notes online, no transcripts, so several items below are gaps. Every claim below carries the source it came from; self-reported records are flagged as such.

## 1. Billy Walters ("Gambler", 2023): power ratings, player values, injuries, home field, timing

### Takeaway
Walters' published system is a classic power-rating-plus-adjustments approach: a slowly updated team rating (90% old / 10% new game performance), explicit point values for individual players (QB about 7 points, top non-QBs 2.5 to 3, most players about 0), large weight on injury "clusters," a shrinking home-field value, and situational factors. On the execution side his edge came as much from getting the best number, timing (favorites early, dogs late) and moving lines through proxy bettors as from the ratings.

### Cited Findings
- Ratings update rule: the new rating is 90% of the previous rating plus 10% of the "True Game Performance Level" from the latest game, i.e. very heavy shrinkage toward the prior rating — [Covers: Billy Walters Shares the Secrets Behind His Betting System](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Power ratings are used to predict a score, which is then compared against the posted spread — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Player values: QBs are "worth about a touchdown. The best ones are worth more"; top non-QBs are worth "between 2.5-3 points"; he keeps a separate QB-only rating system; roughly 60% of players have "basically zero" value — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Injuries are the "second-most important factor in gaining a handicapping advantage." Clustered injuries matter most, in this priority order: pass catchers, defensive line, offensive line, defensive backs, linebackers, running backs — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Home field: the traditional assumption is 3 points; 1974–2022 data shows "closer to 2.5 points"; in the last four years (as of the 2023 book) it was "less than one point" — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Situational adjustments cover travel distance, turf, schedule quirks, weather, and emotional/situational "S-, W-, E-factors." Examples: Super Bowl winners are upgraded for the first four games of the next season, and Monday Night road teams get "one of the biggest downgrades" — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Execution: "Get the best odds on every bet you make," watch the market-leading sportsbooks, and "bet favorites early and dogs late" — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Staking: maximum 3% of bankroll on one bet, sized in half-units from 0.5 to 3 units, with bigger edges getting more units — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters); the 3% rule came from Chip Reese — [SuperSummary: Gambler](https://www.supersummary.com/gambler/summary/)
- Origins: the Computer Group used a model built by Michael Kent (a former Westinghouse programmer) to generate power-rated numbers, and the syndicate reportedly won "more than 60 percent of its bets" (claim from the memoir) — [SuperSummary](https://www.supersummary.com/gambler/summary/)
- Operation: he moved money through networks of "runners" and "beards" (proxy bettors), which let him move lines without revealing who was betting — [SuperSummary](https://www.supersummary.com/gambler/summary/); [Wikipedia](https://en.wikipedia.org/wiki/Billy_Walters_(gambler))
- Track-record claims (all self-reported or press-reported, never audited): a "30-year winning streak," only one losing year in 39, about 57% of bets won overall, $3.5M won on Super Bowl XLIV, and a claimed $50–60M a year in good years — [Wikipedia](https://en.wikipedia.org/wiki/Billy_Walters_(gambler))
- Credibility context: convicted of insider trading in 2017 (sentence commuted in 2021) — [Wikipedia](https://en.wikipedia.org/wiki/Billy_Walters_(gambler))

### Inferences
- The 90/10 update is a very low learning rate. Walters treats a single game's result as mostly noise and puts his real effort into week-specific adjustments (injuries, situational spots) that the base rating doesn't capture. For the project, this suggests the edge sits in fast, accurate personnel adjustments rather than in a better team-strength estimate.
- A player value chart where most players are worth 0, top non-QBs 2.5–3 and QBs about 7 implies a sparse, position-weighted player model. Player RAPM spreads value across all 22 players and probably can't recover those magnitudes from public data.
- "Favorites early, dogs late" is a statement about the market's order flow (public money comes in on favorites late), not about the model. It is a timing edge that sits on top of the ratings.
- His record is unverifiable, and his ability to move lines with very large proxy bets is not something an individual bettor can copy.

### Gaps
- The full player value tables (per-position numbers for OL, CB and so on, and how they scale with backups) and the exact "True Game Performance Level" calculation are in the book's two methodology chapters, but I could not find a full-text excerpt online. The Covers summary is the most detailed secondary source.
- No numbers found for his weather adjustments.

## 2. Ed Miller and Matthew Davidow ("The Logic of Sports Betting", "Interception"): where edges are, market-making vs taking, opener vs closer, market as prior

### Takeaway
Miller and Davidow treat the market as the authority. Line movement after you bet is a consensus verdict from sharp bettors, CLV is the feedback signal, and edges are most likely in derivative and prop markets rather than the heavily traded main spreads and totals.

### Cited Findings
- Line movement after your bet "isn't random. It reflects something of a consensus of all the serious bettors" — [LSR excerpt: Market Agreement and Resistance](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- "When the price moves lower on a bet you've made, that is market resistance. That's smart people telling you that you made a mistake. Listen to them." — [LSR excerpt](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- CLV as the metric: "If in general you are getting a substantially better price on your bets than what's available at closing, that's good." — [LSR excerpt](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)
- Where to look: avoid heavily traded main markets (moneylines, spreads, totals) and instead look in "related, derivative and prop markets with less picked-through prices" — [LSR excerpt](https://www.legalsportsreport.com/33079/the-logic-of-sports-betting-market-agreement-and-resistance/)

### Inferences
- In this framework, a model that disagrees with the close on NFL sides is wrong by default unless the bettor has information the market lacks. That fits the project's finding that public-data team-strength models don't beat closing lines.
- The authors' advice points toward alternate lines, derivative markets (first half, team totals) and props priced off the main line, where books may derive prices mechanically.

### Gaps
- I found no accessible text of "Interception" (Davidow), or of Miller/Davidow's explicit discussion of market-making vs market-taking and opener betting. Their podcast ("Bettor Thinking") has no transcripts I could reach.

## 3. Rufus Peabody (Massey-Peabody, Unabated, Bet the Process): structure, inputs, market blending, track record

### Takeaway
Massey-Peabody is a play-by-play-derived, opponent- and situation-adjusted team model with weights chosen for out-of-sample prediction. Peabody explicitly says the best forecast blends the model with the market (about 45% model / 55% market), and that bookmakers are better than his model at player-specific and qualitative information. The published pick record (about 55–56% on roughly 500–580 picks) is the closest thing to a verifiable public record among the people covered here, but it is graded against the line at posting time, not the close.

### Cited Findings
- Inputs: rushing, passing, scoring and play success, adjusted for home field and game situation. The model deliberately ignores "personnel, coaching, and motivation" — [Wharton: Prof. Cade Massey explains the analytics behind his NFL rankings](https://experience.wharton.upenn.edu/story/prof-cade-massey-explains-the-analytics-behind-his-nfl-rankings/)
- Weights are set by "out-of-sample predictive ability" rather than in-sample fit; "the task...is to predict out-of-sample performance" — [Wharton](https://experience.wharton.upenn.edu/story/prof-cade-massey-explains-the-analytics-behind-his-nfl-rankings/)
- Noise removal: fumble-recovery luck is discounted, strength of schedule is controlled for, garbage time is filtered, weather is put in context, and recent games are deliberately not overweighted (he says "most people" overweight them) — [RotoWire interview](https://www.rotowire.com/football/article.php?id=40172)
- Market blend: "the best combination of the Massey-Peabody prediction and the market prediction is 45 percent MP, 55 percent market" — [RotoWire](https://www.rotowire.com/football/article.php?id=40172)
- Betting threshold: a disagreement of about 2 points with the market generally gives +EV in the NFL, adjusted for key numbers (3, 7) — [RotoWire](https://www.rotowire.com/football/article.php?id=40172)
- He acknowledges that bookmakers excel at "rating specific players and factoring in qualitative information" — [RotoWire](https://www.rotowire.com/football/article.php?id=40172)
- Injury limitation of the ratings: "the ratings know if a starting QB is out — but they don't know if he is playing through an injury" — [Fantasy Life / Betting Life newsletter](https://betting-life-newsletter.beehiiv.com/p/rufus-dawgs-name)
- Track record, official full-game picks: 314-256-18 (55.4%) over 578 games. Unofficial "break-even or better" leans went only 47.1% — [RotoWire](https://www.rotowire.com/football/article.php?id=40172). A Wharton piece gives 56.4% ATS over 500+ public predictions since 2011 — [Wharton](https://experience.wharton.upenn.edu/story/prof-cade-massey-explains-the-analytics-behind-his-nfl-rankings/). The two figures come from different time windows and are roughly consistent.
- Unabated's position: "We're not creating their power rankings for them"; it supplies tools to "leverage their own numbers." Its NFL simulator uses Massey-Peabody as a baseline that users adjust (for example for QB injuries) and compare against 30+ books — [The Handle: Unabated](https://thehandle.substack.com/p/unabated)
- Podcast topics that may hold more detail (show notes only, no transcripts): "Does Defense Matter?", player props — [Establish The Run ep. 67](https://establishtherun.com/episode-67-one-on-one-with-pro-sports-bettor-rufus-peabody/); "How does Rufus get NFL data?", Super Bowl betting process — [Gambling With an Edge](https://www.lasvegasadvisor.com/gambling-with-an-edge/podcast-rufus-peabody/)

### Inferences
- The 45/55 blend is the clearest statement from a top pro that a good public-data model is only worth roughly equal weight with the market. Bets come only from the residual disagreement (about 2+ points), and the edge is small.
- The weak "lean" record (47.1%) is evidence that small model-vs-market disagreements carry no value. Only the large-disagreement subset showed an edge.
- Peabody's prominence is more in props and derivative markets (his Unabated and podcast topics) than in NFL sides, which fits Miller's view of where edges are.

### Gaps
- No Massey-Peabody CLV record was found. The 55–56% figure is graded against lines at posting time (likely mid-week), not the close.
- No Bet the Process transcripts were found to confirm his statements on which markets are beatable.

## 4. Public models with tracked records (nfelo, Warren Sharp, others)

### Takeaway
nfelo publishes a detailed performance page. It beats the opening line by a small margin and is roughly level with the closing line on accuracy (−0.12% accuracy differential vs close), so it does not beat the close on prediction. Its ATS record against openers is better than against closers. Warren Sharp's large records are self-reported, with no CLV. I found no audited multi-season CLV record for any public NFL side model.

### Cited Findings
- nfelo: 56.81% ATS against opening lines and 54.34% against closing lines; +169.4 cumulative units against openers since 2009, against 113.5 "expected units" from CLV; average CLV of 6.11% per play — [nfelo model performance](https://nfeloapp.com/games/nfl-model-performance/)
- nfelo accuracy vs market: +0.38% accuracy differential vs the opening line, −0.12% vs the closing line. Model MAE is 10.1 points. Seasons ranged from +28.8 units (2023) and +23.5 (2015) to −7.1 (2014) and −8.1 (2024) — [nfelo model performance](https://nfeloapp.com/games/nfl-model-performance/)
- The nfelo performance page says the model is built on the nflfastR dataset and "custom open source models," but gives no methodology details there — [nfelo model performance](https://nfeloapp.com/games/nfl-model-performance/)
- Warren Sharp: lifetime NFL 1,781-1,295 (58%), +391.6 units since 2006; "computer totals" 739-431 (63%), overs 66-18; 2024: 438-326. All self-reported on his own site, with no third-party verification and no CLV — [Sharp Football Analysis lifetime records](https://www.sharpfootballanalysis.com/lifetime-betting-records/amp/)
- R.J. White (CBS) builds starting power ratings from win-total markets (de-juiced), uses 3 points for home field adjusted by venue, adjusts 10+ points for elite-QB-to-backup changes and 0.5–1 point for a star WR, and stresses betting into Sunday-night openers. He reports a 58% SuperContest hit rate over four years (contest standings are public, but this claim was not independently checked here) — [CBS Sports](https://www.cbssports.com/nfl/news/nfl-betting-on-opening-lines-how-to-build-a-power-rating-system-and-why-its-crucial-to-beating-the-odds)

### Inferences
- nfelo's own numbers show the pattern the project found: a public-data model plus market information has value against the opener but not against the close.
- The nfelo record "since 2009" almost certainly includes a backtest from before the model was published, so it is not a pre-registered live record. Its "6.11% CLV" metric also needs careful reading: it measures CLV for the model's picks against openers, which is a different thing from beating the close.
- Self-reported tout records (Sharp) at 58% over 3,000+ picks would be extraordinary. Without CLV or third-party audit they shouldn't be treated as evidence.

### Gaps
- Not covered due to the tool-call budget or lack of sources: PFF betting model tracked records, ESPN FPI vs market, 538, inpredictable, Austin Mock (The Athletic), FTN/DVOA as a betting tool, Unabated NFL line records, and the records of Fezzik, Krackomberger, Sevransky and Kelly Stewart. Their contest results (Westgate SuperContest, Circa) are public but were not retrieved here. I found no reliable, independently audited CLV record for any of them.
- I could not verify how much of nfelo's history is a live record and how much is backtest, or nfelo's exact market-regression method.

## 5. Common professional practices: market-informed ratings, information the market lacks, timing, sharp consensus, player-level ratings

### Takeaway
Across sources, pros (a) start from or blend heavily with the market (win totals, sharp-book prices); (b) spend their effort on fast, accurate personnel and injury information, often automated; (c) treat beating the close as the scoreboard; (d) choose timing deliberately (openers for rating-based edges, Monday-morning or later windows for liquidity, favorites early and dogs late); and (e) accept thin margins of 1–2% hold on turnover. Their edge is usually informational or executional, not a better team-strength regression.

### Cited Findings
- Spanky (Gadoon Kyrollos): "Trading all day. We don't watch any games." His team spent "seven, eight years" building software to detect injuries and line moves. "We're looking at line movement and trying to beat that closing number...if you beat the closing number, that means you're beating the market." — [Covers: What's the difference between you and a sharp?](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- Spanky on timing: "You kinda wanna wait for Monday morning, around 9 to 11 a.m. ET...where you're gonna get enough down, you're gonna get a good number." His sustainable hold is roughly "1 and 2 percent" of handle. He takes unfavorable NFL action to keep access to profitable college limits — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- Spanky on injury information: "You're not going to get the nuggets unless you go through the bullshit. You have to keep digging." — [Covers](https://www.covers.com/industry/whats-the-difference-between-you-and-a-sharp-step-inside-the-mind-of-a-professional-bettor)
- Market-derived starting ratings: build preseason power ratings from de-juiced win-total markets — [CBS Sports, R.J. White](https://www.cbssports.com/nfl/news/nfl-betting-on-opening-lines-how-to-build-a-power-rating-system-and-why-its-crucial-to-beating-the-odds)
- Model/market blending (45/55) and a roughly 2-point bet threshold — [RotoWire, Peabody](https://www.rotowire.com/football/article.php?id=40172)
- Player-level sparse values and injury clusters — [Covers, Walters](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Caveat on CLV: Karl Whelan's study of 3,670 NBA games (2022–25, ten books) found that only the top CLV decile made meaningful profit (+11.4%). Three of five positive-CLV deciles lost money, and the 9th decile had 5% CLV but only +0.2% profit, because CLV has to exceed the book's margin to pay. The study is NBA moneylines only, not NFL — [Karl Whelan: The Truth about Closing Line Value](https://www.karlwhelan.com/?p=2595)

### Inferences
- What the project may be missing:
  - **Information speed and quality on personnel.** Spanky automated this, Walters ranks it second in importance, and Peabody concedes books do it better than his model. A public-data model built on weekly snapshots structurally can't capture it.
  - **Timing and execution.** This means betting openers where the rating edge exists, shopping the best number across books, and reading market resistance. It matches the project's finding that only cross-book price edges survived.
  - **Market selection.** Props and derivatives over main NFL sides.
- Using the market as a prior (a 55% weight, or ratings seeded from win totals) and modeling only the residual is standard practice. Under this framing, the project's pre-registered tests vs the close were a test the pros themselves say public team-strength models should fail.
- CFB: Spanky's comment on keeping college limits suggests college markets are more profitable than NFL for sharps (likely because of more games and less efficient pricing). This is inferred from one quote, not documented.

### Gaps
- No direct sources retrieved for Captain Jack Andrews' specific NFL modeling views beyond Unabated's tool and CLV education framing; for Voulgaris (the ESPN page returned empty); or for Pinnacle betting-resources articles.
- No quantitative public evidence found on opener-vs-closer edge sizes in the NFL from the pros' own records. The only numeric opener/closer comparison is nfelo's page.
- College-football-specific modeling practices (talent composites, transfer portal, returning production) were not covered by any source found.
