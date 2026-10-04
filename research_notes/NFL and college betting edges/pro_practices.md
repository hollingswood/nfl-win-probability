# How successful professional football bettors (NFL and CFB) actually operate

Research date: 2026-10-04. Scope: practices of known pros/syndicates, niche angles, process/sizing/accounts, bookmaker perspective, contests, court/syndicate history. About 26 tool calls; several primary pages (Action Network, Bloomberg, Review-Journal, podcast transcripts) were paywalled, robots-blocked or only had show notes, so many podcast claims could not be verified from text. Coverage is uneven: Walters, the preseason market, the MA limits hearing, survivor pools and service-academy totals are well sourced; Spanky, Peabody, Boston, Fezzik, Rynning and the rest are mostly gaps.

## 1. What known pros and syndicates say about how they win (verified vs marketing)

### Takeaway
The best-documented pro method (Billy Walters) is power ratings plus point values for individual players. Each player gets a numeric value, weather and situation are adjusted for, and the bet goes in early on favorites and late on dogs, spread over as many books as possible with a maximum of about 3% of bankroll per bet. Most other "pro" names on the list come to us through podcasts whose methods we can't confirm from text. Treat any record claim as unverified unless it comes from a contest leaderboard or a court record.

### Cited Findings
- Walters ("Gambler", 2023, as summarized by Covers) keeps neutral-field power ratings and updates them weekly as roughly "90% of its old rating plus 10%" of the latest game's performance. — [Covers: Billy Walters shares the secrets behind his betting system](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Walters' player values: a QB is worth about a touchdown, top non-QBs are worth "between 2.5-3 points", and "at least 60% of players have a value of basically zero". He calls injuries the "second-most important factor in gaining a handicapping advantage" and adjusts for players who play hurt. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Walters on NFL home field: historically "closer to 2.5 points" (1974-2022), but "over the last four years, it's worth less than one point". — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Walters' situational adjustments: Super Bowl teams are "upgraded for its first four games of the next season"; Monday-night road games get significant downgrades; teams that lost by 19+ points are upgraded the next week. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Walters on execution: "bet favorites early and dogs late", hold "as many accounts as you can with different sportsbooks", cap any single event at 3% of bankroll, size bets in half units from 0.5 to 3 units, and "start with the assumption that you'll lose it all". — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Walters' book also alleges that Phil Mickelson wagered over $1 billion. This is press coverage of the book, not a description of a method. — [NBC News](https://www.nbcnews.com/news/us-news/phil-mickelson-wagered-1-billion-famed-gambler-billy-walters-says-book-rcna99257)
- Westgate SuperContest winners describe their own processes:
  - Damon Graham (2016) shortlisted 6-7 games a week and cut to 5, using divisional and scheduling context and beat writers on Twitter for injury news.
  - James Salinas (2015) re-watched every Sunday game on film, looked for emotional letdown spots, and avoided Thursday games so he had time for practice reports.
  - Robert Burns (2006) built his edges in the offseason from NFL Gamebooks, noting that raw stats mislead without score and time context.
  - — [TheLines: three SuperContest winners](https://www.thelines.com/westgate-supercontest-winner-tips/)
- Rufus Peabody and Captain Jack Andrews (Unabated, 2021 show notes) covered NFL season simulators, how single results move season projections, automating derivative markets, Super Bowl prop arbitrage, NJ exchanges and place/show pricing. These are show-note topics only; no transcript was available. — [Gambling With an Edge podcast, 2021-09-16](https://www.lasvegasadvisor.com/gambling-with-an-edge/podcast-captain-jack-andrews-and-rufus-peabody/)
- Spanky Kyrollos has long-form interviews (Business of Betting E259, 2025-10-10; Gambling With an Edge). I could not retrieve the text. — [Business of Betting E259](https://businessofbetting.podbean.com/e/e259-gadoon-spanky-kyrollos/); [GWAE podcast](https://www.lasvegasadvisor.com/gambling-with-an-edge/podcast-pro-sports-bettor-spanky/)
- Alan Boston is "best known however as a college basketball handicapper". He was profiled in the 2001 book *The Odds: One Season, Three Gamblers and the Death of Their Las Vegas*. Wikipedia has no detail on his methods. — [Wikipedia: Alan Boston](https://en.wikipedia.org/wiki/Alan_Boston)

### Inferences
- Walters' method is, in effect, a hand-tuned rating with player-level injury values and situational adjustments. It is also explicitly structural about timing (favorites early, dogs late) and about distribution across many accounts. The project's backlog already lists "favorites early, dogs late". Walters naming it as a core rule is outside support for testing it from `history/odds_*.json`.
- Walters says home field has shrunk to under a point. That supports checking whether the model's home-field term adapts over time or is fixed from long history.
- The SuperContest winners lean on qualitative information that arrives late (beat writers, practice reports, film). That is consistent with the project's AI-news reader being a reasonable direction. None of these accounts has audited ROI outside the contest.

### Gaps
- No verified text from Spanky, Peabody, Fezzik, Kelly Stewart, Gill Alexander, Joe Peta, Ed Miller, Matthew Davidow, Krackomberger, Professor MJ, Pauly Howard or Erin Rynning on football methods. Their podcasts exist (Be Better Bettors has an Alan Boston interview: [Spotify](https://creators.spotify.com/pod/profile/bebetterbettors/episodes/Alan-Boston-Interview-e1cig6a)), but there are no transcripts. Verified vs. tout status could not be assessed for most names on the list.
- I did not retrieve Walters' specific key-number half-point values from primary text.

## 2. Niche CFB and NFL angles pros mention

### Takeaway
Two niche markets are documented. The NFL preseason is an openly sharp-dominated market where coach and QB-competition information drives lines up to 7 points. Service-academy (triple-option) matchup unders hit at a very high historical rate even after totals were cut sharply. Both are low-limit or low-volume.

### Cited Findings
- Ed Salmons (Westgate oddsmaker, Aug 2023) on the NFL preseason:
  - "you're probably looking at 80 to 90 percent sharp money".
  - Preseason "lines can move seven points".
  - The drivers are coach motivation (especially first-year coaches), QB competitions that give starters extended snaps, and offensive-line availability.
  - Limits are $2,000-3,000, versus about $20,000 in the regular season.
  - Baltimore had a 20-game preseason win streak, and Salmons called backing them "an automatic bet". The Rams under McVay sit starters and get faded.
  - — [Las Vegas Review-Journal (develop mirror), Aug 2023](https://develop.reviewjournal.com/?p=2376396)
- Service-academy head-to-head totals:
  - The under went 44-9-1 since 2006-07 (83%), and 15-2 in the last 17 games.
  - Covers' explanation: triple-option rush rates near 90% (Army) keep the clock running; each academy practices against the option daily; coaching is continuous and the portal is barely used.
  - Totals were cut from 51-56 (2006-2017) to 28-36.5 (2021-2024).
  - The 2023 CFB clock rule (no clock stoppage after first downs outside the last two minutes) is noted as a possible change in scoring.
  - — [Covers, 2024-10-02](https://www.covers.com/ncaaf/service-academy-betting-trends-over-under-total)
- Action Network has repeatedly published the service-academy under system and "service academies as big underdogs" angles. These pages only came up as search titles and were not fetched. — [Action Network: Army vs Air Force under system](https://www.actionnetwork.com/ncaaf/army-vs-air-force-odds-betting-prediction-service-academy-under-system-trend); [UCF-Navy academies as big dogs](https://www.actionnetwork.com/ncaaf/ucf-navy-betting-odds-service-academies-big-underdogs)
- Rufus Peabody has discussed college football futures on a 2024 podcast. Only the title was available. — [Audible listing](https://origin-www.audible.in/podcast/ITEM_NAME/B0DHG23PYB)

### Inferences
- NFL preseason fits the existing stack well. The information that drives the lines is QB snap plans and coach tendencies, which is the same kind of depth-chart and news input that `news.py`/`news_llm.py` already reads. Low limits cap the dollar value, but these are also markets that books expect sharps to bet.
- The service-academy result is a famous public trend with obvious selection bias, and the market has already cut totals by about 20 points. Treat it as a hypothesis for a pre-registered paper track, priced with the existing key-number total distribution (`totals.py`), not as a proven edge.

### Gaps
- I found no sourced material this session on the other angles on the list:
  - Week 0-2 inefficiencies, FCS-vs-FBS pricing, bowl-season motivation and opt-outs.
  - MAC weather, Hawaii travel, coordinator changes, small-conference specialization.
  - Smaller books' line-making in college football.
- Erin Rynning's small-conference CFB approach was not found in text.

## 3. Process: ratings, timing, sizing, record keeping, accounts

### Takeaway
The documented pro process is: a rating with a slow update (about 10% weight on each new game), numeric player values, a hard cap on risk per event (3%), half-unit sizing tied to confidence, and as many accounts as possible. Bettors are now also actively managing how "pro" their accounts look.

### Cited Findings
- Walters' process (see section 1 for the details) is the 90/10 rating update, a 3% cap per event, 0.5-3 units in half-unit steps, as many accounts as possible, and no chasing losses on games without an edge. — [Covers](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
- Account-longevity practice (podcast episode, 2026-01-20): "account health matters more than short-term EV", "betting 'too perfectly' is a red flag", plus bonus farming and hedged strategies. — [The Modern Edge ep3: Drew Tabor / Ungambled](https://themodernedge.podbean.com/e/ep3-drew-tabor-ungambled-%E2%80%94-the-stealth-strategy-sharp-bettors-use-to-avoid-limits/)

### Inferences
- The 3% cap and 0.5-3 unit scale amount to roughly fractional-Kelly sizing with a hard ceiling. The project's flat 1u paper tracks are fine for testing. If any track goes live, the sizing rule should be pre-registered as a new version of the rules file.
- The project's soft-book-vs-Pinnacle track is exactly the "betting too perfectly" profile. Expect short account life at US regulated books (see section 4) unless the bets are deliberately camouflaged. Measured edge should be discounted by expected account lifetime.

### Gaps
- No primary source this session on specific Kelly fractions used by pros, beard or runner networks today, or offshore vs. US legal account practice.
- Nothing found on pros' record-keeping or CLV-tracking methods beyond general statements.

## 4. Bookmaker perspective: what books fear and what gets limited

### Takeaway
US regulated books told the Massachusetts regulator that they limit for these reasons:
- taking mispriced or stale odds
- betting low-liquidity markets
- syndicate (coordinated) betting
- courtsiding / data-feed latency

Few accounts are limited by headcount, and bettors usually aren't told. Circa positions itself as the book that welcomes sharps.

### Cited Findings
- Massachusetts Gaming Commission roundtable, 2024-09-11 (Fanatics, BetMGM, DraftKings, FanDuel):
  - Limit triggers are exploiting mispriced odds, low-liquidity markets, syndicate betting and courtsiding; "latency of third-party data feeds" is also cited.
  - BetMGM said about 1% of MA customers had limits. FanDuel said 0.043% of bets were subject to maximum limits.
  - Books generally do not notify limited bettors.
  - — [Birches Health summary of MGC roundtable](https://bircheshealth.com/resources/limiting-bettors)
- Circa Sports "has made a name for itself in the sports betting world by welcoming the sharp bettors that some books shun". Chris Bennett has been sportsbook director since 2023; this is from a 2025-11-12 brief, and the full interview was paywalled. — [CDC Gaming brief](https://cdcgaming.com/brief/qa-circa-sportsbook-director-talks-about-his-dream-job/)
- Bookmakers deliberately cap preseason NFL at $2-3k because the market is 80-90% sharp. — [Review-Journal mirror, Aug 2023](https://develop.reviewjournal.com/?p=2376396)
- Chris Andrews (veteran Vegas bookmaker) discusses bookmaking on The Power Rank podcast; not transcribed here. — [The Power Rank podcast](https://thepowerrank.libsyn.com/chris-andrews-on-bookmaking-in-las-vegas)

### Inferences
- In the regulators' list, "mispriced odds" plus "syndicate" is effectively a description of soft-vs-sharp price arbitrage. The project's most successful track (soft-book price vs Pinnacle) is therefore the behavior most likely to get limited. Props and other low-liquidity markets (the receptions track) are explicitly called out as a trigger too.
- Books that take sharp action (Circa, exchanges, prediction markets) are where an edge can scale. That argues for measuring edge against fills achievable there, not only at soft books.

### Gaps
- I could not retrieve direct quotes from Jay Kornegay, John Murray, Jeff Benson or Ed Salmons on football-specific limiting. The Action Network article on modern sharp tactics was robots-blocked.

## 5. Contests and pools

### Takeaway
Sharp survivor pools (Circa Survivor) reward balancing three things: a pick's win probability (EV), the future value of saving a team, and pick popularity. Holiday weeks with only 3-4 games are the biggest leverage points. SuperContest-style ATS contests are won with selective picks and late information.

### Cited Findings
- PoolGenius on sharp survivor pools such as Circa:
  - Evaluate EV, future value and popularity together. "Just fade the chalk" does not work, and "fading just to fade is probably a mistake."
  - On Thanksgiving/Christmas slates (3-4 games), a 7-point favorite can draw 50%+ of picks instead of about 20%.
  - When chalk exceeds 50% popularity, underdogs become justifiable.
  - Save strong favorites for the holiday weeks.
  - The Circa field's pick distribution often differs from the general public's.
  - — [PoolGenius: navigating sharp NFL survivor pools](https://poolgenius.teamrankings.com/circa-survivor-picks/articles/navigate-sharp-nfl-survivor-pools-circa-survivor/)
- Further Circa Survivor strategy coverage is in [PoolGenius core concepts](https://poolgenius.teamrankings.com/circa-survivor-picks/articles/circa-survivor-strategy-core-concepts/), [VSiN Circa Survivor 2025](https://vsin.com/nfl/circa-survivor-2025-winning-nfl-survivor-pool-strategies/) and [Action Network Circa Survivor tips](https://www.actionnetwork.com/nfl/circa-survivor-strategy-expert-tips-for-the-largest-nfl-survivor-contest). These were found in search and not fetched.
- SuperContest winners' processes are summarized in section 1. — [TheLines](https://www.thelines.com/westgate-supercontest-winner-tips/)

### Inferences
- The project already produces calibrated weekly win probabilities for every NFL game plus season simulations. That is the core input for survivor EV and future-value math. The missing pieces are a pick-popularity estimate and a path optimizer across weeks.
- Contests are not limited the way accounts are, which makes them a natural way to use a model that does not beat closing lines.

### Gaps
- I found no verified ROI figures for contest entries (Circa Millions or SuperContest entry economics). I also found nothing on office pick'em strategy math.

## 6. Court records and syndicate history (Computer Group, Walters)

### Takeaway
The Computer Group (Michael Kent, from the early 1980s) was among the first to bet college football with statistical models. Its documented wagers for 1983-84 were about $5M in winnings ($10-15M including undocumented bets), with Billy Walters and Dr. Ivan Mindlin among those using the information.

### Cited Findings
- Michael Kent, a former Westinghouse nuclear-reactor engineer, built "the first successful program for handicapping basketball and football games" by entering collected team statistics into computer models aimed at college football.
- Documented 1983-84 winnings were nearly $5M; the estimate including undocumented bets is $10-15M.
- Walters, Glen Walker and Dr. Mindlin had access to the group's numbers.
- It was first reported nationally in Sports Illustrated in March 1986.
- — [Wikipedia: Michael Kent](https://en.wikipedia.org/wiki/Michael_Kent_(computer_specialist)); [Bloomberg Law, 2026-04-10](https://news.bloomberglaw.com/securities-law/the-sports-betting-revolution-started-with-a-nuclear-engineer) (paywalled beyond the intro)

### Inferences
- The Computer Group's historical edge came from statistical models of college football at a time when books priced by hand. That gap has largely closed in big markets, which is consistent with the project's own result that models do not beat closing lines.

### Gaps
- I did not retrieve primary court documents on the 1985 FBI investigation of the Computer Group, its outcome, or the beard/runner mechanics. Walters' 2017 insider-trading conviction (his stock trading, not sports betting) was not verified this session.

## 7. Strategy categories a quantitative bettor might be missing

### Takeaway
Compared with the current system (soft vs Pinnacle, exchanges, props shopping, win-total priors, Tuesday moves, AI news), the documented pro edges not yet covered are:
- player-level injury point values
- timing by side (favorites early, dogs late)
- the NFL preseason
- service-academy and option-team totals
- survivor and contest EV
- treating account longevity as part of EV

### Cited Findings
- Each item above is cited in sections 1-5:
  - Player point values and timing by side: [Covers on Walters](https://www.covers.com/guides/betting-tips-from-pro-sports-bettor-billy-walters)
  - NFL preseason: [Review-Journal mirror](https://develop.reviewjournal.com/?p=2376396)
  - Academy unders: [Covers](https://www.covers.com/ncaaf/service-academy-betting-trends-over-under-total)
  - Survivor EV: [PoolGenius](https://poolgenius.teamrankings.com/circa-survivor-picks/articles/navigate-sharp-nfl-survivor-pools-circa-survivor/)
  - Account health: [Modern Edge podcast](https://themodernedge.podbean.com/e/ep3-drew-tabor-ungambled-%E2%80%94-the-stealth-strategy-sharp-bettors-use-to-avoid-limits/)
  - Limit triggers: [MGC summary](https://bircheshealth.com/resources/limiting-bettors)

### Inferences
- Ranked by fit with the existing code (my judgment, not sourced):
  1. **Survivor/contest EV.** It reuses calibrated weekly probabilities and season sims, and there are no limits.
  2. **Favorites early / dogs late** as a timing rule for existing bets, measured from the saved odds snapshots.
  3. **NFL preseason news-driven bets**, using the existing depth-chart and news stack at low limits.
  4. **Player-value injury adjustments in points** beyond the QB. The project's README says position-group injury features showed no gain in the model, so the value would be in news timing, not model fit.
  5. **Academy and option totals** as a pre-registered paper track.
- Account management (camouflage, bet distribution across books, staying under triggers) is a real category that the current system does not model at all.

### Gaps
- Not covered this session: prediction-market making (Kalshi/Sporttrade liquidity provision) by pros, bowl-season angles, and CFB early-season rating priors.
