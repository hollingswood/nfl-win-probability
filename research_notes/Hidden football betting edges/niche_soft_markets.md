# Least efficient (softest) NFL and college football betting markets and niches for an Arizona bettor, 2025-2026

Scope note: these notes try not to re-propose what this repo has already tested. Local evidence is cited by repo path. That includes NFL and college main lines vs Pinnacle, NFL and college 1H spreads and team totals, the major NFL player props, longshot moneylines, Super Bowl futures shopping and preseason win totals.

Evidence quality labels:
- **[A]** peer-reviewed or a large documented sample
- **[B]** a reputable practitioner with numbers
- **[C]** anecdote, opinion or a small sample
- **[D]** low-quality or likely AI-generated page

Web research for this pass was limited to about 30 searches and fetches. Many claims below are therefore marked as gaps instead of being filled from memory.

---

## 1. College player props: how soft are they, which books offer them, and do they key off season averages?

### Takeaway
At Arizona-regulated books this niche is effectively closed. The Arizona Department of Gaming lists "collegiate proposition bets" as prohibited, and Action Network's state table lists Arizona as "No Props" for college players. The only paths are research on out-of-state Odds API quotes, which is not bettable here, or DFS pick'em apps. Arizona's legal status for those apps was not verified. Deprioritize this niche.

### Cited Findings
- The Arizona Department of Gaming FAQ says: "Wagers will be prohibited on injuries, penalties, collegiate proposition bets, and high school events." **[A, regulator]** — [AZ Dept of Gaming](https://gaming.az.gov/node/1255)
- Action Network's state-by-state table (Apr 27, 2024) lists Arizona among the states with no college player props, alongside CO, IA, LA, MD, MA, NY, OH, OR, PA, TN, VT, VA and WV. Only KS, MI, NC, DC, WY and IN (pre-game only) allowed them. **[B]** — [Action Network](https://www.actionnetwork.com/ncaab/college-basketball-player-prop-rules-by-state)
  - **Contradicted by** a DeucesCracked page (May 8, 2026), which says Arizona bars props only on in-state athletes (ASU, UA, NAU). It cites no statute or regulator, and the site's style suggests AI-generated content. **[D]** — [DeucesCracked](https://www.deucescracked.com/blog/arizona-prop-bet-restrictions-rules-what-you-can-and-cannot-bet). Treat the regulator's wording as controlling until a real AZ book menu is checked.
- Ohio banned college player props at the NCAA's request. Low, transparent limits have been proposed as the alternative. College player props stay available offshore and through pick'em apps (PrizePicks, Sleeper) in states where those are unregulated. **[B/C]** — [Closing Line (Gouker)](https://closingline.substack.com/p/problem-with-banning-college-prop-betting)
- Regulatory pressure continues. An investment-firm estimate put the cost of a nationwide college-prop ban at about $200M a year for sportsbooks. **[C]** — [Covers](https://www.covers.com/industry/investment-firm-estimates-200-million-loss-if-college-player-props-banned); [LSR](https://www.legalsportsreport.com/174227/could-college-player-props-betting-ban-really-cost-sportsbooks-200m-a-year/)
- A data-vendor blog advertises "22 markets across 63 books" for college football player props, which would make an out-of-state or offshore research dataset feasible. **[C, vendor marketing]** — [OddsPapi](https://oddspapi.io/blog/?p=3723)

### Inferences
- An Arizona bettor cannot act on college player props at DK, FD, theScore or any regulated book. The question of whether college props key off season averages is academic here. Don't spend Odds API credits on a college-props paper track.
- If a research-only check is still wanted (for example, to learn something about vendor pricing that carries over to NFL), use this design:
  1. Pull Odds API `americanfootball_ncaaf` event props from non-AZ US books for 2025 (historical endpoint, about 1 credit per market per event).
  2. Regress (actual − line) on opponent-adjusted pass or rush defense and the pace gap from CFBD (plays per game, opponent SP+/EPA allowed).
  3. If |t| > 3 on a pre-declared feature, the "season-average" hypothesis has support. It is still not bettable in AZ.
- No cited evidence was found that college prop lines ignore opponent or pace. That remains a common practitioner claim only.

### Gaps
- No primary source from 2025-26 confirming the current Arizona menu. Check the DK, FD and theScore AZ apps on a college Saturday to settle the conflict.
- Whether PrizePicks, Underdog or Sleeper pick'em operate legally in Arizona in 2026 and offer college players was not verified.
- No quantitative study of college prop softness (CLV or hold vs NFL props) was found.

---

## 2. College niches: FCS-vs-FBS, small conferences, MACtion, Weeks 0-2, bowls and opt-outs, CCG/CFP, service academies

### Takeaway
Two documented biases exist:
- **Service academy unders.** The record is large, but the market is visibly adjusting.
- **Power-conference-vs-mid-major spread mispricing.** This comes from a 2013 academic paper on 2002-11 data and is probably arbitraged since.

Both are cheap to re-test with CFBD lines the repo already holds. Bowl and opt-out games are the most plausible information-driven edge, but they need news-timing data, which the repo's AI news reader now logs.

### Cited Findings
- **Service academies (Army, Navy, Air Force playing each other)** — [Covers, Oct 2024](https://www.covers.com/ncaaf/service-academy-betting-trends-over-under-total) **[B: real records, small n, trend article]**
  - The under is 44-9-1 (83%) in service academy games since 2006.
  - 2006-17: under 29-8, with an average total of 51 and average scoring of 42.1.
  - 2018-20: the average total fell to 42.3 and the under went 8-1 (34 ppg).
  - 2021 on: the average total was 35.4 and the under went 7-1-1 (27.6 ppg).
  - The author warns that books are adjusting and that unders may "start swinging the other way." Example: the 2024 Navy-Air Force total opened 36.5 and fell to 34.5.
- **Power vs mid-major spreads** — Krieger, Girdner and Fodor, "The Power of Wagering on Power Conferences," *Journal of Prediction Markets* 7(1), 2013. **[A, but old data]** — [UWF repository](https://ircommons.uwf.edu/esploro/outputs/journalArticle/The-power-of-wagering-on-power/99380090342306600)
  - Data: FBS regular seasons 2002-2011.
  - Spreads in power-conference vs mid-major games were "set statistically irrationally." Backing the power team returned about +2.94%.
  - Conditioning on AP rank, spread size and week raised profits further.
- **Bowl opt-outs.** The ESPN Chalk story on fluctuating bowl odds (the Kenny Pickett case) could not be retrieved; the fetch returned empty. PFF runs recurring "bowl games betting market update" pieces, which suggests sharp attention to bowl line moves. Neither shows an exploitable bias. — [PFF](https://www.pff.com/news/bet-college-football-bowl-games-betting-market-update); [ESPN Chalk (not retrieved)](https://www.espn.com/chalk/story/_/id/32868941/everchanging-bowl-game-odds-kenny-pickett-high-stakes-guessing-game-bookmakers-bettors)
- **Local evidence on college news.** In 965 games with a starting-QB change, Pinnacle moved only +0.24 points against the team from Sunday/Monday to kickoff, so most QB changes are priced by Sunday. The AI news reader now logs opt-outs, interim coaches and similar items. **[A, local]** — `output/research/cfb/round2.md`
- **Local evidence on situational factors.** All 10 pre-declared college situational factors failed. College 1H spreads and team totals vs Pinnacle or derived lines also failed: Arizona books almost never beat Pinnacle's derivative at the same number (6-19 cases in three seasons). **[A, local]** — `output/research/cfb/factor_screen.md`, `output/research/cfb/deriv_screen.md`

### Inferences
- **Service academy unders: not adopted.**
  - Re-test as a pre-declared rule: academy-vs-academy games, plus academy vs a triple-option or flexbone opponent, betting the under at the best AZ book's total against the Pinnacle close.
  - Data: CFBD lines 2014-2025, plus 2026 live.
  - At about 3-4 academy-vs-academy games a year, the sample is too small for p < 0.05 within any reasonable horizon. Widen the scope to every game involving a triple-option team (Army, Navy, Air Force, plus Kennesaw State and Georgia Southern historically) for about 35-40 games a year. Still expect a low-n rule.
  - Pass bar: CLV > 0 vs the Pinnacle close in at least 3 of 4 seasons, and ROI p < 0.05 over at least 150 bets.
- **Power vs mid-major spreads.** CFBD closing lines for 2014-2025 make an out-of-sample check of the 2002-11 result trivial.
  - Rule: back the P4/P5 side in non-conference P5-vs-G5 games, all spreads, and split the results by spread bucket (≤7, 7-17, >17) and week (0-4 vs 5+).
  - The repo's college shop track already beats Pinnacle on main lines. This would be a directional test of whether the closing line itself is biased. Expect failure; the cost is about one hour.
- **FCS-vs-FBS and Weeks 0-2.** No sourced evidence of a persistent bias was found. Opening lines in these games rely on thin information (transfer portal, new coaches), but they are also the lowest-limit markets.
  - Test: opener-to-close move size and direction by week, using CFBD opener and close.
  - If week 0-2 openers move ≥ 2x more than week 5+ openers and the move is predictable from a transfer-adjusted preseason rating (the repo has `transfer_prior.json`), an "opener-only" track at Arizona books could be considered.
  - Caveat: `round2.md` notes openers-vs-ratings failed for college in general. This would be a week 0-2-only re-cut, which is post-hoc, so pre-register it.
- **MACtion (Tue/Wed/Thu November games).** No sourced bias was found. Lines in these games are set with full information, and the games get national TV attention. Low priority.
- **CCG and CFP.** These are among the most heavily bet and sharp-scrutinized college games. Expect efficiency. No evidence was found otherwise.

### Gaps
- No 2018-2025 academic or large-sample study was found on FCS-vs-FBS cover rates, MAC/Sun Belt/CUSA/MWC efficiency, HBCU lines, or bowl opt-out line under-reaction.
- The ESPN bowl-odds article could not be fetched, so there are no book quotes on opt-out limits.

---

## 3. Futures and season markets: win totals, division and conference winners, playoff yes/no, awards, in-season futures after injuries, hedging

### Takeaway
Futures carry very high hold and long capital lockup. The only consistently documented mispricing is mean-reversion in win totals: teams that beat their total by 3+ wins tend to go under the next year. Awards markets are narrative-driven, but the sourced evidence is thin.

The repo's preseason win-total prior already passes and Super Bowl shopping is a live paper track. The incremental ideas are:
- shop division and playoff yes/no props against the repo's own season simulator
- re-price awards from a voter-model

Both need an odds archive the repo does not have. Per `season_sim.md`, no playoff-odds or in-season futures archive was found.

### Cited Findings
- **Win-total regression since 2000** — [Action Network](https://www.actionnetwork.com/nfl/nfl-season-win-totals-public-betting-biases-bears-saints-2019) **[B: real records, 2000-2018, but a trend article with no vig or CLV analysis]**
  - Teams that beat the prior total by 3+ wins went under the next year 54-30-6 (64.3%). By 2+ wins: 83-58-10 (58.9%).
  - Teams that missed by 2+ wins went over 95-70-6 (57.6%).
  - The author attributes this to the public bidding up last year's over-performers and to book shading.
- **Local:** the repo's season simulator and futures test used nfelo win totals for 2013-2026 with juice. Division odds were only top-2 favorites from an unverified source, and no playoff or in-season futures archive was found. **[A, local]** — `output/research/season_sim.md`
- **Awards markets.** The search turned up only practitioner guides (Establish The Run, "NFL awards: who wins and why") and a Kalshi NFL-awards odds hub, which indicates prediction markets now list award races. — [Establish The Run](https://establishtherun.com/nfl-awards-who-wins-and-why/); [Stokastic on Kalshi awards](https://www.stokastic.com/articles/prediction-markets/kalshi-nfl-awards-odds) **[C, not fetched]**
- **Prediction markets.** Bloomberg Law reports that Kalshi and Polymarket "rewrite Super Bowl playbook for pro gamblers," meaning pros now use exchange-style markets for Super Bowl positions. **[B, headline only, not fetched]** — [Bloomberg Law](https://news.bloomberglaw.com/tech-and-telecom-law/gambling-pros-adjust-to-a-super-bowl-on-the-prediction-markets)
  - The Arizona Department of Gaming wrote to the CFTC on Jun 2, 2025 about event contracts, and the repo notes that Kalshi's Arizona status is contested. — [ADG letter PDF](https://www.ingame.com/wp-content/uploads/2025/06/Arizona-Dept-Gaming-Letter-CFTC-June-2-2025.pdf)
- The Action Network explainer covers how to remove hold from multi-outcome futures; it was not fetched for numbers. — [Action Network](https://www.actionnetwork.com/education/calculate-remove-hold-from-futures)

### Inferences
- **Division winner and "to make playoffs" yes/no shopping (new angle vs the Super Bowl shop).**
  - Rule: each Tuesday, run the repo's season simulator, which is market-rating based and calibrated on 2015-19. Compare its probabilities with the power-devigged consensus of all US books for division winner and playoff yes/no.
  - Bet only when the best AZ allowed-book price shows EV ≥ 5% against the average of the simulator and consensus. The simulator alone overfits, so this is a blend, not a pure model.
  - Grade on CLV against the same market a week later and at season end, not on results. Futures resolve too rarely.
  - Data need: does the Odds API carry `americanfootball_nfl_division_winner` and playoff markets? Verify in the Odds API docs. If not, scrape book JSON or log daily screenshots, which is costly.
- **In-season futures after injuries.** When a QB injury is announced, books move game lines within minutes, but futures and division boards at retail books plausibly lag.
  - Test: with hourly futures snapshots (if the Odds API carries them), measure the lag between a ≥ 3-point move in the team's next-game spread and the division-odds move at each book.
  - Rule: if any AZ book's division price is unchanged ≥ 60 min after a confirmed QB-out news event, bet the division rival. Log only until there are ≥ 30 events.
  - This mirrors the repo's "news vs line timing" infrastructure in `news.log_first_seen`.
- **Win-total regression.** The repo's preseason win-total prior "passes." Check whether it already contains last-year residual vs total. If not, add `prior_year_wins − prior_year_total` as one pre-declared feature and test it on 2013-2026 nfelo totals. It is a cheap ablation.
- **Awards.** MVP and OROY outcomes are driven by voter narratives (QB on a top seed). A voter model (seed, QB EPA, team wins) could price MVP probabilities. With about one resolution per year per award, there will never be a statistically valid test. Treat it as entertainment or CLV-only. Low priority.
- **Hedging.** No sourced hedging strategy evidence was found. Futures plus hedge is arithmetically break-even at fair prices. Hedging adds value only if the bettor's bankroll utility is concave (Kelly). Not an edge source.

### Gaps
- No academic paper on NFL award-market efficiency or a favorite-longshot bias in NFL futures was retrieved. The search returned generic soccer and racing FLB papers.
- Hold figures for division, playoff and award markets at AZ books were not found.
- Whether the Odds API's NFL futures coverage includes division winners and awards in 2026 is unverified.

---

## 4. Specials: first TD scorer, first scoring play, 2+ TDs, kicker props, overtime, race to points, winning margins, highest scoring half/quarter

### Takeaway
Specials carry enormous hold. First TD scorer hold ran 22-65% by book in 2022, so blanket betting is hopeless. The edge, if any, comes from:
- pricing structure: deriving first-TD prices from team-first-TD probability times player TD share, shopping across books with different rule sets, and "no" or "field" sides that intuition misprices
- recurring "yes" props that the public underrates (Peabody: three unanswered scores "Yes" up to −170)

Arizona books rarely offer two-way specials, which limits this.

### Cited Findings
- **First TD scorer hold, 2022 playoff average:** SuperBook 6.9%, Circa 22%, BetMGM 23%, Kambi/BetRivers 34.7%, DraftKings 42.3%, FanDuel 42.5%, bet365 47.9%, Hard Rock group 65.5%. **[B, 2022 data, prices have likely changed]** — [Action Network](https://www.actionnetwork.com/nfl/best-sportsbooks-first-touchdown-anytime-td-props)
  - BetMGM often had the best first-TD prices (best on 7 of the first 8 Super Bowl players).
  - DraftKings had good prices on top players and poor prices on mid-tier players.
  - Kambi was best on some secondary RB and TE prices.
- **Anytime TD hold by book, same article:** bet365 26.7%, FanDuel 33%, Kambi 35%, DraftKings 38%, BetMGM 40.9%, Caesars 43.1%, Hard Rock 46.7%. SuperBook offered two-way anytime TD at 4.1%. **[B]** — [Action Network](https://www.actionnetwork.com/nfl/best-sportsbooks-first-touchdown-anytime-td-props)
- **Rule differences change payout odds across books for the same event.**
  - FanDuel's first-TD "Team Defense" excludes special teams.
  - DK and BetMGM offer a combined D/ST.
  - Offshore books offer "the field." — [Closing Line](https://closingline.substack.com/p/fanduel-td-markets)
  - Caesars has no field option. — [Action Network](https://www.actionnetwork.com/nfl/best-sportsbooks-first-touchdown-anytime-td-props)
- **Rufus Peabody (Super Bowl props):** play three unanswered scores "Yes" up to −170, because the event happens "far more often than public intuition suggests." He waits for markets to develop and targets "mispriced props, alt lines" later in the week. **[B, paraphrased in VSiN]** — [VSiN](https://vsin.com/the-vsin-daily/inside-the-mind-of-sharpest-of-the-super-bowl-prop-sharps/)
- PFF launched a "First Touchdown Finder" (Jan 23, 2026) that uses first-15-play EPA differentials and red-zone rates to flag value against the best market price. It gives no performance record. **[C]** — [PFF](https://www.pff.com/news/introducing-pff-first-touchdown-finder-a-smarter-way-to-attack-first-td-scorer-markets)
- **Local:** anytime TD was already tested with no pass. Solo tackle unders showed +5.0% ± 2.6% (p = 0.03) on 1,358 bets, flagged as a lead, not a pass. **[A, local]** — `output/research/props_more.md`

### Inferences
- **(a) First-TD-scorer structural model (new angle vs anytime TD).**
  - P(player scores first TD) = P(team scores the first TD of the game) × P(player | team's first TD).
  - Fit P(team first TD) from the sharp spread and total. P(no TD) is about 1-2% of games.
  - Fit P(player | team TD) as a shrunk share of team rushing and receiving TDs plus red-zone opportunities over the trailing 8-16 games, using nflverse play-by-play and leakage-safe `shift(1)`.
  - Rule: bet when the best allowed-book price is ≥ 20% above fair. A 20% overlay is needed because the vig is 20-40%; this is roughly break-even plus margin.
  - Grade on results: binomial with a fixed-odds ROI test, at least 500 bets for p < 0.05 at +15% ROI given about +1500 average odds.
  - Data: Odds API NFL props market for first TD scorer (the key name needs verifying, likely `player_1st_td`) at the Friday and close snapshots, 2023-25 historical.
  - Expect a high failure probability because of the huge hold. The test is cheap since the repo already has the props download harness.
- **(b) "No" or "Yes" sides of novelty props that intuition misprices.** Examples are three unanswered scores, overtime yes/no, a successful 2-point conversion, and a safety.
  - Compute empirical base rates from nflverse play-by-play 2015-2025, conditioned on spread and total.
  - Compare with book prices where they exist. AZ regulated books mostly post these for the Super Bowl and some primetime games.
  - Example: overtime frequency by |spread| bucket, from nflverse pbp. Rule: bet OT "Yes" when price-implied probability is < 0.8 × the empirical rate in that bucket, with at least 30 historical games per bucket.
  - Data need: the Odds API likely doesn't carry these specials. They would need manual logging or book scraping, which is the main obstacle.
- **(c) Kicker props (FG made, longest FG, kicking points).** The repo has a forecast-wind totals track and Open-Meteo forecasts.
  - Hypothesis: retail books price longest-FG and FG-made lines off season averages and ignore forecast wind.
  - Rule: under on "longest FG" and "FG made" when forecast sustained wind is ≥ 15 mph at an open stadium.
  - Data: Odds API kicker markets (keys to verify, likely `player_field_goals` / `player_kicking_points`), historical 2023-25 from the existing props harness, plus nflverse kicking stats and the Open-Meteo archive.
  - Note: the repo says the recorded-wind version of totals leaked. Use forecast wind as of the bet snapshot only.
  - Pass bar: CLV > 0 vs the median of other books plus ROI p < 0.05 over at least 150 bets.
- **(d) Winning-margin bands and race-to-X points.** These can be priced exactly from the repo's key-number margin distribution (`margins.py`, `margin_total.py`). A margin-band shop rule is a natural extension of what the repo does well: price any number from the sharp line and shop it.
  - Rule: EV ≥ 5% at the best AZ book vs the `margin_total` fair, for 13-18-point bands and 1-6-point bands.
  - Data need: margin-band odds are not in the Odds API main feed (verify the alternate or derivative coverage). Without them, this test cannot run.
- **(e) "Highest scoring half/quarter."** These can be priced from 1H/2H scoring splits. Historically the second half scores slightly more in NFL games, a fact books know. Low priority unless odds become available.

### Gaps
- The 2022 hold figures are dated. Current 2025-26 first-TD hold at DK, FD, theScore, MGM and Caesars in AZ was not found.
- No source documents an edge in kicker props, overtime yes/no, race-to-points or winning-margin bands with a real sample.
- Odds API market keys and historical coverage for first TD, kicker props and specials were not verified in this session.

---

## 5. Quarter and second-half markets, halftime live pricing, team to score first

### Takeaway
No sourced evidence was found that NFL quarter or second-half lines are soft at US retail books, and the repo found NFL and college 1H markets priced better than a naive conversion. The plausible remaining angle is halftime and 2H lines, which books set fast (often algorithmically) during a roughly 13-minute window. Testing it needs live halftime odds snapshots, which the repo does not collect.

### Cited Findings
- **Local, NFL:** 1H spread, 1H total and team totals vs the full-game market showed no edge. Team-style features (scripted EPA, 1H pace) never reached |t| ≥ 3. The best was 1H pace for 1H totals, at about 0.3 points per SD and t ≈ 2. 2025 derivative coverage thinned (team totals only at BetOnline, BetRivers, FanDuel and part of Caesars). **[A, local]** — `output/research/derivatives.md`
- **Local, college:** converting full-game lines into 1H and team totals lost 2-4% CLV. "The books price derivatives better than a simple conversion does." **[A, local]** — `output/research/cfb/deriv_screen.md`
- **Team to score first.** PFF's first-TD tool uses first-15-snap EPA differentials as the "early advantage" signal but publishes no track record. **[C]** — [PFF](https://www.pff.com/news/introducing-pff-first-touchdown-finder-a-smarter-way-to-attack-first-td-scorer-markets)
- An OpticOdds blog on how books handle correlation in same-game parlays was found but not fetched. **[C]** — [OpticOdds](https://opticodds.com/blog/correlation-in-same-game-parlays)

### Inferences
- **Team to score first.** This is roughly a coin flip modulated by the spread and the opening-kickoff decision, which is itself 50/50 and unknown pre-game.
  - Test: from nflverse pbp 2012-2025, fit P(home scores first | spread, total). Check whether scripted-play EPA adds |t| > 3.
  - The repo's derivatives work already found scripted EPA useless for 1H, so the prior is "no."
- **Halftime 2H lines (live).**
  - Hypothesis: retail books set 2H lines from the pregame line plus the 1H score and under-use 1H yards per play, success rate and turnover luck.
  - Test design:
    1. From nflverse pbp, build a 2H margin and total model using pregame sharp lines plus 1H EPA, success rate and turnovers.
    2. Check |t| > 3 for 1H EPA beyond 1H score on 2012-2022, then validate on 2023-25.
    3. Only if the backtest passes, collect live halftime 2H odds via the Odds API `h2h_h2`/`spreads_h2`/`totals_h2` markets. These may be in-play only; verify they exist live and that snapshots can be taken in the halftime window.
  - The repo's hourly watch would need a halftime-triggered job (kickoff + about 95 min for 1 pm ET games).
  - Limits on live markets are typically low. Account risk is high if consistently beating halftime lines.
- **Quarter lines (1Q/3Q).** Expect high hold and thin limits. No evidence was found. Low priority.

### Gaps
- No 2023-26 study of NFL halftime or 2H line efficiency was found.
- Historical halftime 2H odds archives were not located. The Odds API historical in-play coverage is unverified.

---

## 6. NFL draft props, preseason games, coach and player "next" markets

### Takeaway
No reliable evidence was gathered on these markets in this pass. They are information-asymmetry markets: draft props are driven by leaked reporting, and next-coach markets by front-office sources. Edges there come from news speed, not modeling, and books cut limits quickly. They don't fit the repo's automated data strengths.

### Cited Findings
- No sources on NFL draft-prop or next-head-coach market efficiency were retrieved in this session.
- **Related local infrastructure:** `news_llm.py` and `news_sources.py` log first-seen times of news from RSS and Bluesky with per-source accuracy, and `news_audit.py` holds a pre-registered promotion rule for QB news. **[local]** — `CLAUDE.md`

### Inferences
- **Draft props (first pick at position, player draft position over/under).** Lines move on reporter tweets. The repo's Bluesky reporter-tracking stack is a natural fit for measuring whether named insiders' posts precede line moves.
  - Test: log draft-prop odds hourly in April (Odds API coverage unverified) plus Bluesky first-seen times, then measure the price move in the 60 minutes after a reporter mention.
  - This is research-only. Arizona allows NFL draft betting as far as known (unverified), and limits are typically small.
- **Preseason games.** Coach-specific starter-play tendencies are the classic angle, but the market has known this for decades. No evidence of a remaining edge was found. Skip.
- **Next head coach.** Retail-heavy and low-limit, resolving about 5-8 times a year. Not testable statistically. Skip.

### Gaps
- No sourced data on draft-prop holds, limits, Arizona availability or documented edges.
- No sourced data on preseason-game ATS biases in the 2020s.

---

## 7. Novelty: player milestones (season-long yards), weekly "leaders" markets, prop parlays and SGPs

### Takeaway
Season-long player props and weekly leader markets are among the least-attended markets by sharp origination, as is all prop pricing to some degree. Same-game parlays are priced with correlation engines and carry high hold. The repo already pre-registered season-long prop unders for 2027 but lacks a data source. The weekly-leader idea can be priced exactly from the repo's existing per-player prop distributions, but its odds are not in the Odds API main feed.

### Cited Findings
- Establish The Run:
  - "There is no sharp book for props, at least in the same way there is for sides and totals."
  - FanDuel originates its own prop numbers. DK and Caesars get numbers "from the same source," and MGM outsources too.
  - Many offshore books use the same prop tool.
  - Props remain "incomparably illiquid" vs sides and totals.
  - Books that can't price the full weekly sheet minimize liability with low limits. **[B]** — [Establish The Run](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/)
- Same source:
  - Caesars typically allows at least $500 on props even for sharp accounts, even on openers, and "sometimes moves too aggressively," so "you can sometimes get very strong prices fading an initial move."
  - FanDuel gives "respectable limits for most accounts, especially closer to game time." **[B]**
- Per ESPN (via Establish The Run), spread, moneyline and total legs were only 11% of same-game-parlay legs in the 2021 NFL season. Props dominate SGP construction. **[B]** — [Establish The Run](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/)
- **Local:** the 2027 preseason season-long prop unders were pre-registered on 2026-10-04 in `season_props_rules.json`, and "none of our feeds carry season props." The receptions shop rule passed and the Tuesday star-unders track is live. **[local]** — `CLAUDE.md`, `output/research/props_full.md`

### Inferences
- **(a) Fade Caesars' initial prop moves (new angle; the receptions shop rule shops levels, not move overshoot).**
  - Hypothesis from Establish The Run: Caesars (`williamhill_us` in the Odds API) over-moves props after early limit bets.
  - Rule: when Caesars' line moves ≥ 1.0 units (yards or receptions) or ≥ 25 cents between two snapshots, and the move exceeds the median move at the other ≥ 3 books by ≥ 0.5, bet the opposite side at Caesars if its price beats the others' median no-vig by ≥ 2%.
  - Grade on CLV vs the median close.
  - Data: hourly props snapshots. The repo currently pulls props at Friday and close only. Add a Tuesday-Wednesday opener snapshot plus 2-3 intraday pulls for about 3 credits per event per pull.
  - Requires opening a Caesars account in AZ.
- **(b) Weekly leader markets (top passer, rusher or receiver of the week).**
  - Fair prices can be simulated from per-player yardage distributions: the repo has prop lines plus the dispersion from `props_full`. Sample correlated player outcomes and count the winner. A Monte Carlo over about 30 QBs is cheap.
  - Retail books plausibly over-price stars, a favorite-longshot pattern in multi-runner markets. This is unverified.
  - Data need: these markets aren't in the Odds API. They require manual or browser capture of DK and FD leader boards. The data barrier is high, so rank this low.
- **(c) SGPs and prop parlays.** Books price correlation explicitly, and the hold is high. The only known exploitable form is a correlated SGP at a book whose engine under-prices correlation; such opportunities get patched fast and lead to limits. It cannot be tested without SGP quote capture. Skip.
- **(d) Season milestones (e.g., 1,000 rushing yards yes/no).** These can be priced by simulating the remaining season with the repo's player-stat model. Data and availability in AZ are unverified. This is the same blocker as the 2027 season-props item.

### Gaps
- No current (2025-26) source quantifies SGP hold for NFL at AZ books. A DeucesCracked hold page was found but rated [D] and not used.
- No source documents weekly-leader market hold or a favorite-longshot bias in them.

---

## 8. Which niches pros say still pay in 2025-2026, and the ranked catalogue

### Takeaway
The consistent practitioner message is that props and late-week alternate lines remain the least sharply originated markets, with no true "sharp prop book." Shopping against stale or vendor-shared prices and fading over-reactive moves is where value persists, with Caesars and FanDuel giving usable limits.

For this repo, the receptions shop rule and college main-line shop already capture the core of that. The remaining untested niches with the best (evidence × testability × AZ availability) are ranked below.

### Cited Findings
- Practitioner views:
  - No sharp book for props; vendor-shared lines across books; low limits as the books' defense — [Establish The Run](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/) **[B]**
  - Peabody targets mispriced props and alt lines later in the week, not openers — [VSiN](https://vsin.com/the-vsin-daily/inside-the-mind-of-sharpest-of-the-super-bowl-prop-sharps/) **[B]**
  - Pros are moving Super Bowl action to Kalshi and Polymarket (headline) — [Bloomberg Law](https://news.bloomberglaw.com/tech-and-telecom-law/gambling-pros-adjust-to-a-super-bowl-on-the-prediction-markets) **[B, not fetched]**
- Documented biases with numbers:
  - Win-total over-performer regression, 64.3% unders on +3 over-performers (2000-18) — [Action Network](https://www.actionnetwork.com/nfl/nfl-season-win-totals-public-betting-biases-bears-saints-2019)
  - Service academy unders, 44-9-1 since 2006 with adjustment underway — [Covers](https://www.covers.com/ncaaf/service-academy-betting-trends-over-under-total)
  - Power-vs-mid-major spreads, +2.94% for 2002-11 — [J. Prediction Markets 2013](https://ircommons.uwf.edu/esploro/outputs/journalArticle/The-power-of-wagering-on-power/99380090342306600)
- Hold on specials is very large, 22-65% for first TD (2022) — [Action Network](https://www.actionnetwork.com/nfl/best-sportsbooks-first-touchdown-anytime-td-props)
- Arizona rules: no collegiate proposition bets and no injury bets — [AZ Dept of Gaming](https://gaming.az.gov/node/1255)

### Inferences

**Ranked catalogue (author's synthesis; rank = plausibility of edge × testability with existing data × AZ availability).**

| # | Niche | Why soft | Evidence | Limits / account risk | Concrete test (pre-register; pass = CLV > 0 in ≥ 3/4 seasons, or ROI p < 0.05 over ≥ 150 bets) | Data status |
|---|---|---|---|---|---|---|
| 1 | Kicker props (FG made, longest FG, kicking points) in forecast wind | Retail lines likely built off season averages; a vendor-priced, low-attention market | C (hypothesis); repo has forecast-wind infrastructure | Low limits ($50-250 typical, unverified); low account risk | Under when forecast wind is ≥ 15 mph at an open stadium, at the bet-time forecast; best allowed-book number; 2023-25 historical Odds API kicker markets plus nflverse kicking | Have weather and props harness; verify Odds API kicker market keys |
| 2 | Fade Caesars' over-reactive prop moves | Caesars "moves too aggressively" (ETR) | B (practitioner) | Caesars ≥ $500 prop limits; reviews larger bets | Move ≥ 1 unit vs ≥ 3-book median; bet the opposite side at ≥ 2% EV; CLV vs median close | Needs 2-3 extra intraday props snapshots per event (credits) |
| 3 | First TD scorer, structural model + rule-difference shopping | Holds of 22-65%, but mid-tier players are inconsistent across books (DK weak mid-tier, MGM strong) | B (hold data, 2022) | Low limits; low risk | P(team first TD) × player TD share from nflverse; bet at ≥ 20% overlay; ≥ 500 bets for a results test | Verify Odds API first-TD market and 2023-25 history |
| 4 | Division and playoff yes/no futures vs season sim plus consensus | Very high hold, but different books lag on news; the repo has a calibrated simulator | B (structure) / C (edge) | Moderate limits, long lockup | EV ≥ 5% vs a 50/50 blend of sim and consensus; CLV a week later; injury-lag sub-rule (≥ 60 min stale after QB-out news) | Futures archive missing; verify Odds API futures keys; start logging daily now |
| 5 | Service academy and triple-option unders | Style extremes; books adjusting but possibly still lagging | B (44-9-1, small n) | Normal college limits | Under in academy and triple-option games at best AZ total vs Pinnacle close, 2014-25 CFBD | Have data (CFBD lines); about 1 hour |
| 6 | Power-vs-G5 non-conference spreads | 2013 academic finding | A (old) / likely arbitraged | Normal | Back the P4 side by spread bucket and week, 2014-25 CFBD closes | Have data; about 1 hour |
| 7 | Halftime 2H lines from 1H EPA and success rate | Algorithmic, fast pricing in a short window | C | Low live limits; higher account risk | Backtest a 2H margin and total model (pregame line + 1H EPA) |t| > 3, then live snapshots | Needs live halftime odds capture (new infrastructure) |
| 8 | Novelty yes/no (OT, three unanswered scores, safety, 2-point) | Public intuition misjudges base rates (Peabody) | B (one practitioner) | Mostly Super Bowl and primetime only | nflverse base rates by spread bucket vs posted prices | Odds mostly not in API; manual capture |
| 9 | Winning-margin bands and race-to-X | Exactly priceable from `margins.py` | C | Low | EV ≥ 5% vs `margin_total` fair | Odds likely not in API |
| 10 | Weekly leaders and season milestones | Multi-runner favorite-longshot pattern likely | C | Low | Monte Carlo from prop distributions | No odds source |
| — | College player props | — | — | **Not legal at AZ regulated books** | Research-only | — |
| — | Draft props, next coach, preseason | Information markets | none gathered | Fast limits | Not suited to automation | — |

Also cheap and worth doing:
- Solo-tackle unders (+5.0%, p = 0.03, local) are worth a pre-declared 2026 re-test, since the 2024-25 line coverage was too thin to clear the multiple-testing bar (`output/research/props_more.md`).
- The win-total over-performer regression should be checked as a single added feature in the existing preseason prior.

### Gaps
- No direct 2025-26 interviews or transcripts from Unabated, Circles Off, Bet the Process or Captain Jack Andrews were retrieved. The podcast search returned only episode listings, so "what pros say in 2025-26" rests mainly on Establish The Run and Peabody via VSiN.
- AZ-specific limits by market and book were not found in any reliable source.
- Odds API coverage (market keys, history) for kicker props, first TD, NFL futures, halftime and in-play markets needs checking in its docs before tests 1, 3, 4 and 7 are scheduled.
