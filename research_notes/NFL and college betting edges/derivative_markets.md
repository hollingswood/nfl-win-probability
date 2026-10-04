# Edges in Derivative and Secondary Football Betting Markets (NFL + FBS)

Scope: player props, 1H/quarter lines, team totals, alternate lines, teasers, same-game parlays (SGPs), futures (win totals, awards, Heisman), pick'em/DFS and exchange prop markets. Research date 2026-10-04. Note: about 22 tool calls were used. Several prop-specialist sources (Unabated, ETR, 4for4) are practitioner writing, not peer-reviewed; quantified evidence for derivative markets is thin compared with sides and totals. Where a claim comes from my own domain reasoning rather than a fetched source, it sits under **Inferences** or **Gaps**, never under Cited Findings.

---

## 1. Player props (NFL and CFB): which types are mispriced, under bias, how sharps model them, books and limits

### Takeaway
The best-documented prop edge is structural. Player outcomes are right-skewed and books hang lines near the median, so bettors who project the mean get pulled toward overs. Season-long prop unders have a multi-year record of winning about 60% or more. For weekly props there is no single sharp book: Caesars and FanDuel are the domestic books that originate and move numbers, Pinnacle's limits are small, and opportunities are being priced out over time. I found no rigorous public study of weekly prop under hit rates by prop type.

### Cited Findings
- **Median vs mean.** Player performance is positively skewed: a player can have huge games but cannot go below zero, so the mean sits above the median. Books set prop lines at the median (50th percentile). Using mean projections leads to "betting far too many overs, and is almost guaranteed to be unprofitable." (Jack Andrews, 2026-07-31) — [Unabated](https://unabated.com/post/the-biggest-mistake-youre-making-when-betting-nfl-player-props)
- **Limits (2026).** Pinnacle prop limits are about "$250 before gameday, and maybe $500 or above on gameday." Even sharp bettors find "$500 bet sizes fly under the radar at many of the bigger U.S. books," which "a few years ago [was] unheard of." FanDuel deals many gameday props at -110/-110. DraftKings outsources props to Swish Analytics. The article says "There is no sharpest sportsbook for props. At least, not yet." — [Unabated](https://unabated.com/post/the-biggest-mistake-youre-making-when-betting-nfl-player-props)
- **Prop ecosystem (2023).** No single book is sharp for props. Pinnacle, Betcris and Circa are sharp on sides and totals but post limited prop sheets late in the week. Caesars gives "$500 limits for sharp accounts, even on openers" and moves aggressively, sometimes overreacting, which creates value on the other side. FanDuel originates its own numbers with respectable limits near kickoff. DraftKings had high limits early in 2021 but now follows other books. Comparing Caesars and DraftKings juice shows whether sharp action has landed. Books "cannot efficiently price all props simultaneously," which is why limits stay low. (Jack Miller, 2023-08-25) — [Establish The Run](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/)
- **Props are getting more efficient (2025).** Player props have shifted from "one of the easiest markets to beat to an efficient market," and FanDuel is called "one of the more efficient books with its player prop lines." These are editorial claims with no CLV data or sample sizes. — [BetSmart, 2025-07-11](https://betsmart.beehiiv.com/p/efficient-prop-markets)
- **Season-long prop unders (2021-22).** Unders won 63% of 269 props in 2021 and 60% of 335 in 2022, for 61% of 604 combined. By category over the two years:
  - QB passing yards unders: 74% (43/58)
  - Interception unders: 66% (21/32)
  - RB rushing TD unders: 66% (39/59)
  - QB rushing TD unders: 64%

  The mechanism is that an under has many "outs" (injury, benching, a trade, the QB getting hurt), and interception unders win both ways (a bad QB gets benched, a good QB throws few). The author recommends unders on "90%+ of season-long props." (Connor Allen, 2023-06-15) — [4for4](https://www.4for4.com/2023/preseason/key-winning-season-long-player-props)
- **Season-long unders (5 seasons).** The same author later reported tracking 1,392 NFL season-long props over 5 years: "Blindly betting every under has been profitable in 5 CONSECUTIVE seasons," with a stated angle for overs. Only the post snippet was seen; the full content was blocked by robots.txt, so ROI is unknown. — [Connor Allen on X](https://x.com/ConnorAllenNFL/status/2081746792849244434)
- **Season-long unders, other sources.** These are corroborating headlines only; the pages were not fetched. — [Yahoo Sports](https://sports.yahoo.com/nfl-betting-betting-unders-on-season-long-props-is-the-way-to-go-172913329.html); [Stealing Lines](https://stealinglines.substack.com/p/best-season-long-player-props-to)
- **Under drift in game totals.** Through Week 6 of 2023, unders hit 61.3%, the highest rate through six weeks since 1991. Unders hit 57.6% in 2022 and 57.7% in 2017, against 48.5% on average since 2003. Explanation: scoring was trending down (43.4 PPG in 2023; 218.6 passing yards per game vs 235.7 over 2012-21) and "the betting market is slow to catch up." — [theScore (Eric Patterson)](https://www.thescore.com/nfl/news/2740613)
- **CFB props.** Odds aggregators list CFB player props across many books ("22 markets across 63 books"). This shows the markets exist but says nothing on limits or efficiency (headline only). — [OddsPapi](https://oddspapi.io/blog/?p=3723)

### Inferences
- **Prop types most likely mispriced.** These follow from the skew and limits evidence above, not from a direct study:
  - Highly skewed, low-median props (longest reception, receiving yards for low-target players, rush+rec yards for change-of-pace backs): the gap between mean and median is largest, so unders are favored when the line sits near the mean.
  - Low-limit, rarely bet markets (defensive tackles and sacks, kicker points, CFB props): books spend the least modeling effort here.

  The season-long result (QB passing-yards unders 74%) is consistent with injury and benching mass sitting in the left tail.
- **How to model a counting prop.**
  1. Build a projection: snap share, then routes or carries, then target or carry share, then yards per opportunity.
  2. Condition on pace and projected game script. The project's spread and total already give implied team points and a margin distribution (margins.py).
  3. Convert to a full distribution. Receptions fit well with a binomial on targets times catch rate, or a negative binomial. Yards fit well with a gamma or log-normal, with a point mass at zero for DNP or early injury.
  4. Price P(X > line) at the median, never by comparing a mean projection to the line.

  Data: nflverse `load_player_stats`, snap counts, `load_participation` (routes are approximate), ftn charting. The current receptions line-shopping track can be extended to receiving and rushing yards using the same median-of-books fair price, then priced against a skewed distribution.
- **Steam signal.** Caesars moving aggressively on sharp action and overreacting suggests a "Caesars steam" filter: compare the Caesars price with the median of other books and fade Caesars-only moves. This is testable with the existing hourly odds watch if the Odds API returns Caesars props.
- **Season-long prop unders as a new paper track.** This is a cheap, pre-registerable rule: bet every under at the best allowed-book price, using about 1,400 props over 5 years as the prior. The NFL scoring-trend under drift is a reminder that market lag after a regime shift (rule changes, scoring decline) is a recurring source of under value.

### Gaps
- No public, rigorous study of **weekly** NFL prop under hit rates or ROI by prop type for 2018-2026 was found. The "unders hit about 55% or more" folklore remains unverified here.
- No sourced evidence on CFB prop limits, which books post CFB props first, or CFB prop efficiency.
- No quantified evidence on anytime-TD hold or mispricing. My search hit only touts. It is widely believed that ATD holds are high (often 20%+ on the yes side), but I have no source for that.
- I did not verify current Circa or BetOnline prop limits or opening times.

---

## 2. Same-game parlays: correlation mispricing, which books price it worst, +EV evidence, how to compute

### Takeaway
Books price SGPs with Gaussian copulas or empirical joint frequencies, and add a large margin (15-25% house edge vs 4-5% on singles). That margin swamps most correlation errors. I found no credible public evidence of persistently +EV SGPs. The only theoretical openings are (a) underpriced *negative* correlation and (b) leg combinations a book's correlation matrix doesn't cover well.

### Cited Findings
- **Pricing methods.** Books use either (1) a Gaussian copula, mapping binary legs to latent normals with a correlation matrix and integrating by Monte Carlo, or (2) empirical frequencies of the combination in comparable games. Sophisticated books use a hybrid of the two. (Joey Shackelford, updated 2026-08-03) — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- **Example.** In a 3-leg parlay, independence gives a joint probability of 16.0% and the copula gives about 21.2%. Across 500 comparable games all three legs hit 20.4% of the time, vs 15.7% under independence. — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- **Illustrative correlation matrix.**

  | Pair | ρ |
  |---|---|
  | Team win, QB over 275 yards | 0.35 |
  | Team win, game over | 0.28 |
  | QB over 275 yards, game over | 0.42 |

  — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- **House edge.** SGPs "typically carry house edges of 15-25%, compared to 4-5% for single bets." In the example, a 3-leg SGP pays +350 against a fair +429, about a 14.9% edge, while a plain -110 3-leg parlay carries about 0.7% extra edge. The article finds **no evidence of +EV SGPs** and advises advantage players to avoid them. — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- **Negative correlation.** With ρ = -0.30, the joint probability falls from 24.8% under independence to 19.2%, a fair price of +421. A typical SGP price is +450. Bettors seldom build such parlays, and sharp books adjust for them. — [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)

### Inferences
- **How to compute a fair SGP price here.** Simulate games jointly from play-by-play-derived distributions (team points, margin, player stats conditional on game script), or fit a copula to historical nflverse player-game and team-game outcomes. Then compare with the SGP price after removing each leg's vig.
  - The realistic +EV case is **a leg the book misprices as a single**, for example a stale prop, combined with correlated legs priced at a small correlation markup.
  - A pure correlation edge is unlikely at a 15-25% hold.
- **Do not build an SGP paper track first.** The evidence is too weak to justify it. If you test anything, test "SGP price vs simulated fair price" offline on recorded SGP quotes. The Odds API does not provide SGP quotes, so this would need manual capture or a third-party SGP pricing feed.

### Gaps
- I found no source ranking US books by SGP correlation pricing quality (for example FanDuel vs DraftKings vs Caesars vs bet365), and no measured SGP hold by book.
- No academic paper on SGP pricing was found in this pass.

---

## 3. First half, quarters and team totals: derivatives of the full game? Known edges and half key numbers

### Takeaway
Books largely derive 1H, quarter and team-total lines from the full-game line. The documented recent pattern, 1H overs being undervalued, is being priced out through juice (-118 to -122 on overs in 2026). There is no rigorous public study of half/quarter key numbers or edges. Build these from the project's own play-by-play data.

### Cited Findings
- **1H overs being priced out (2026).** A tout-style analyst reports books "catching on" to 1H over value: lines stay put, but favored overs now carry -118 to -122 juice with unders at plus money. The analyst reported a 19-9-1 record, about +9u, through 2026-09-27. This is a small, unverified sample. — [Jeff's Edge (Substack)](https://jeffsedge.substack.com/p/sunday-the-books-are-pricing-the)
- **Both teams scoring a TD in each half.** Over 11 seasons, in games with high totals (about 53.5), both teams scored a TD in each half about 55% of the time, equivalent to roughly -120. — [Jeff's Edge](https://jeffsedge.substack.com/p/sunday-the-books-are-pricing-the)
- **Same author on key numbers.** He has a separate post questioning whether NFL key numbers are real; its content was not fetched. — [Jeff's Edge](https://jeffsedge.substack.com/p/are-key-numbers-in-the-nfl-real-or)
- **Unfetched explainers.** Pages on 1H team totals and how 1H lines differ from full-game lines appeared in search, but were not fetched or verified. — [Bettorsworld](https://www.bettorsworld.com/nfl-first-half-team-totals/); [BettorEdge](https://www.bettoredge.com/post/nfl-first-half-betting)

### Inferences
- **How books likely derive these lines.** Rules of thumb are 1H spread ≈ 0.5-0.6 × the full-game spread and 1H total ≈ 0.5-0.52 × the full-game total, adjusted by heuristics. This is unsourced here and should be verified empirically.
- **Testable angles from nflverse pbp.** Halftime score is derivable from `total_home_score` at the end of Q2. Two angles are cheap to test:
  - Teams with strong scripted openings or 1H/2H splits in EPA (coach tendencies, slow-starting offenses) vs the fractional 1H line.
  - Team totals derived from spread and total by books that don't account for total-dependent margin distributions. The project's `margin_total.py` already models this, so it can price team totals and 1H lines and compare them to quotes.
- **Key numbers by market.** 1H spread key numbers are 3, 7, 0 (a tie at the half is common), 4, 6 and 10. Quarter lines cluster heavily at 0, 3 and 7. Team totals cluster at 17, 20, 24, 27, 10 and 13. These should be measured from pbp to build key-number-aware distributions, as was done for full-game margins.
- **CFB tempo edge (hypothesis).** Hurry-up teams and slow, run-heavy service academies may distort fractional derivations of 1H and team totals. This is untested and unsourced; CFBD play-by-play data (collegefootballdata.com) would support it.
- **Data constraint.** The Odds API carries 1H, quarter and team-total markets for NFL and NCAAF on its event-odds endpoint (as I understand its documentation; not verified this session). Historical derivative quotes would be needed for any backtest.

### Gaps
- No rigorous public study (with sample size and CLV) of NFL or CFB 1H/quarter/team-total mispricing was found.
- No sourced evidence on halftime (live 2H) line edges.
- No sourced half/quarter key-number frequency tables were found.

---

## 4. Alternate lines and spread/ML/total consistency inside a book

### Takeaway
I found no quantified public evidence of systematic within-book inconsistency. The best tool is the project's own key-number-aware margin and total distribution: price every alt spread, alt total and ML at one book against each other and against the sharp fair price.

### Cited Findings
- **Not verified.** Generic explainers on alternate lines and buying or selling points were found but not fetched; they contain no quantified edges. — [Legal Sports Report](https://www.legalsportsreport.com/sports-betting/alternate-lines/); [DeucesCracked](https://www.deucescracked.com/blog/alternate-lines-betting-guide-buying-selling-points)
- **Line-shopping leakage (unverified).** A blog claims that tracking odds across 10 books found "$47k in annual leakage." The headline was not fetched or verified. — [DEV Community](https://dev.to/edgelab/line-shopping-actually-works-i-tracked-odds-across-10-sportsbooks-and-found-47k-in-annual-leakage-2h4a)
- **Older academic work.** An academic paper examines "arbitrage" and market efficiency in NFL wagering; only its title and abstract listing were seen. — [NC A&T profile](https://profiles.ncat.edu/en/publications/on-arbitrageand-market-efficiency-an-examination-of-nfl-wagering-4/)

### Inferences
- **Highest-leverage cheap test.** The project already has `margins.py`, `margin_total.py` and `buy_costs.json` (half-point buys measured as never +EV). Extend that analysis to alt lines far from the main line. Books often apply flat curves that underweight key numbers 3, 7, 10 and 14 at alternate spreads, and that can misprice alts crossing 7 or 10 relative to the main line.
- **Same check for totals and CFB.** Alt totals crossing common totals (41, 44, 47, 51) can be checked the same way. In CFB, margin distributions are flatter and key numbers weaker, so books copying NFL-style curves for CFB alt lines is a hypothesis to test.

### Gaps
- No source quantified alt-line mispricing by book or by distance from the main line.

---

## 5. Futures: NFL win totals, division, playoffs, awards; CFB win totals, conference, CFP, Heisman

### Takeaway
NFL win-total holds run about 3.8-5.9% on 2026 lines, much lower than awards or outright futures. Season-long player prop unders (Section 1) are the best-documented futures-style edge. I found no quantified evidence on award or Heisman longshot bias or on win-total over/under rates, so the remaining futures angles are modeling hypotheses: simulation from game-level ratings plus injury and QB-news timing.

### Cited Findings
- **NFL win-total hold and data.** 2026 holds range from about 3.8% to 5.9%. nfelo converts win totals into "implied strength" ratings adjusted for hold and schedule, and keeps win-total records back to 2003. — [nfelo](https://www.nfeloapp.com/nfl-power-ratings/nfl-win-totals/)
- **Simulation approach.** A modeling guide on +EV season win totals ("two easy ways and one fun way") appeared in search but was not fetched. — [analytics.bet](https://analytics.bet/articles/nfl-season-win-totals-two-easy-ways-and-one-fun-way-to-find-ev-bets/)
- **Season-long player prop unders.** About 61% under across 604 props in 2021-22, and reported profitable over 5 straight seasons (1,392 props). See Section 1 — [4for4](https://www.4for4.com/2023/preseason/key-winning-season-long-player-props); [X](https://x.com/ConnorAllenNFL/status/2081746792849244434)
- **CFB win totals.** How-to guides and academic gambler's-fallacy and efficiency papers surfaced in search but were not fetched, so they are not usable as evidence. — [The Spread](https://www.thespread.com/cfb-regular-season-win-totals-2025-how-to-bet/); [Kennesaw](https://digitalcommons.kennesaw.edu/facpubs/2523)

### Inferences
- **NFL win totals.** Simulate the season by Monte Carlo with game-level win probabilities from the project's MarginModel, adding rating uncertainty (QB injury probability over the full season). Compare with the no-vig over/under. Season-long QB injury risk adds left-tail mass, consistent with the under edge in season-long player props.
- **Timing.** The project logs futures after QB news. A natural test is whether win-total and division prices react more slowly than game spreads to QB injuries. Compare implied win shifts against the model's re-simulated shift at each logged snapshot.
- **Awards (MVP, OPOY, Heisman).** Outright markets usually carry very high overround. A narrative or voter model (QB on a top-2 seed, stat thresholds) plus no-vig comparison across books is the standard approach. No quantified evidence was found here, so treat this as low priority.
- **CFB.** The preseason win-total prior failed as a CFB ATS feature in this project. That does not imply CFB win totals themselves are mispriced, but it argues against investing there without new evidence.

### Gaps
- No source quantified NFL or CFB win-total over/under historical rates or ROI, MVP or Heisman favorite-longshot bias, or award-market holds.

---

## 6. Teasers beyond Wong (CFB teasers, 6-point teasers after rule changes, totals teasers)

### Takeaway
The one CFB teaser study found (2021, 166 games) shows no subset clearing the 74% per-leg break-even for 2-team teasers at -120. CFB margins lack the 3 and 7 concentration that makes NFL Wong teasers work. Combined with the project's own null result on NFL Wong teasers, teasers are a low-priority area.

### Cited Findings
- **CFB teasers, 2021 Weeks 5-13 (166 games).** Per-leg teaser win rates:

  | Subset | Win rate | Record |
  |---|---|---|
  | All underdogs | 56.4% | 104-61-1 |
  | All favorites | 66.3% | 110-53-3 |
  | Favorites, total > 65 | 73.6% | 40-12-3 |
  | Underdogs, total < 50 | 65.5% | 72-39 |
  | "Wong" dogs +1.5 to +2.5 | 76.5% | 13-4 (tiny sample) |

  A 2-team teaser at -120 needs 74% per leg, so no statistically significant +EV strategy was found, and straight bets beat teasers in nearly all categories. (Carson Mundy, 2021-08-26) — [TheLines](https://www.thelines.com/college-football-teasers-study-2021/)
- **NFL teaser key-number background.** These pages were found but not fetched. — [TeamRankings](https://www.teamrankings.com/blog/sports-betting/teasers-betting-nfl-key-numbers-wong); [Grantland](https://grantland.com/the-triangle/viva-las-vegas-making-sense-of-teaser-bets-in-week-4-of-the-nfl-season/)

### Inferences
- **Totals teasers.** With weak key numbers in totals, they are almost certainly -EV at standard prices. Under-teasers in low-total games could be checked cheaply from the existing totals key-number distribution (`totals_dist.json`) against teaser pricing at allowed books.

### Gaps
- No post-2015 (post-PAT-rule) quantified CFB or NFL totals-teaser studies were found. No current Arizona teaser pricing by book was verified.

---

## 7. Novel: pick'em DFS (Underdog/PrizePicks) and prediction-exchange props (Kalshi)

### Takeaway
Pick'em apps convert fixed multipliers into an implied per-leg price of about -122 to -136. Any leg whose no-vig sportsbook probability exceeds the break-even rate (54.9% for Underdog 5-pick, 55.0% for 3-pick) is +EV, so juiced props at sharp books flag value directly. Kalshi lists a full NFL prop grid, but adoption is modest and its legal status in Arizona is contested.

### Cited Findings
- **Underdog payouts and break-evens.**

  | Entry | Payout | Break-even per leg | Implied price |
  |---|---|---|---|
  | 2-pick | 3x (+200) | 57.7% | -136 |
  | 3-pick | 6x (+500) | 55.0% | -122 |
  | 4-pick | 10x (+900) | 56.2% | -128 |
  | 5-pick | 20x (+1900) | 54.9% | -122 |

  A prop juiced to -140 at books but offered flat on Underdog (implied -122) is value. Restrictions: at least 2 teams per entry, $500 max per entry, $750 max exposure per player across entries. (Jack Miller, updated 2025-08-30) — [Establish The Run](https://establishtherun.com/how-to-beat-pick-em-on-underdog-fantasy/)
- **Optimizers.** BettingPros launched PrizePicks and Underdog optimizers that compare pick'em lines with sportsbook odds. — [BettingPros](https://www.bettingpros.com/articles/bettingpros-launches-new-prizepicks-and-underdog-optimizers/)
- **Kalshi NFL props.** Self-certified with the CFTC in September 2025, Kalshi lists:
  - Passing: completions, attempts, yards, TDs, INTs
  - Rushing and receiving: attempts or receptions, yards, TDs
  - Combined: yards from scrimmage, total TDs
  - Defensive: tackles, sacks, INTs, passes defended
  - Special teams: FGs and XPs, punts, punt yards, return TDs

  Any of these can be split by half, quarter or custom period. "Adoption has been modest for touchdown props." Kalshi had over $2B in 2025 sports volume. No college props appear in that filing. — [Legal Sports Report](https://www.legalsportsreport.com/241859/kalshi-self-certifies-more-nfl-prop-markets/)
- **Arizona legal status (relevant to the user).**
  - March 2026: the Arizona AG filed 20 criminal counts against Kalshi.
  - May 2026: a federal judge blocked prosecution on CFTC-preemption grounds.
  - Late August 2026: the Ninth Circuit held that states can regulate Kalshi sports contracts.

  Availability to Arizona residents is unsettled. — [Public Gaming](https://publicgaming.com/news-categories-m/legal/16348-arizona-ninth-circuit-ruling-backs-state-authority-over-kalshi-sports-contracts-but-election-dispute-remains); see also [Front Office Sports](https://frontofficesports.com/article/arizona-sues-kalshi-after-kalshi-sued-arizona/)

### Inferences
- **Pick'em legs are an easy addition to the props track.** The receptions track already computes a median-of-books fair probability, so tagging any prop where the no-vig probability for one side is at least 55-56% gives pick'em candidates. Paper-track them as "flat -122 equivalent" bets.
  - Caveats: entries must be graded all-or-nothing per slip, pushes reduce entry size, and the multi-team rule constrains correlation.
  - Pick'em operators shade or remove lines that sportsbooks have moved. Speed matters, which fits the hourly watch.
  - Check whether Underdog and PrizePicks are currently legal in Arizona before going live; I did not verify this.
- **Exchange props (Kalshi).** If they are accessible, they may be thin, which suggests a maker strategy: post limit orders at the model's fair price. Fees and the legal situation must be resolved first.

### Gaps
- I did not verify current PrizePicks payout tables (including Flex) or 2026 Arizona availability of pick'em apps.
- Kalshi's fee schedule for prop contracts and its liquidity and spreads were not found.

---

## Summary priority list for this project (inferred, ranked by evidence strength and cost to test)
1. **Season-long prop unders paper track.** Strongest public evidence: about 61% across 604 props in 2021-22, and profitable for 5 consecutive seasons by one tracker. Low effort.
2. **Extend the props track to skewed yardage props, priced at the median.** Use skewed distributions with zero-inflation. Add pick'em leg flags (no-vig ≥ 55%) and Caesars-steam fading.
3. **Derivative consistency checks with the existing margin/total distributions.** Cover team totals, 1H lines and alt spreads/totals against the full-game sharp line. These need the Odds API derivative markets plus historical captures.
4. **Futures simulation and QB-news reaction lag** for win totals and divisions, using existing logged futures.
5. **Low priority (evidence negative or absent):** SGPs (15-25% hold, no +EV evidence), CFB and totals teasers (no subset beat 74%), awards and Heisman.
