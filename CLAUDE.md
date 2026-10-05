# NFL Win Probability — rules for Claude

Predicts home-team win probability for NFL games from nflverse data, explains each prediction,
and benchmarks against Vegas. Runs weekly on GitHub Actions and publishes to GitHub Pages.

## Layout
- `src/nflpred/data.py` — downloads nflverse schedules + play-by-play (release parquet files), cached in `data/raw/`
- `src/nflpred/features.py` — all pre-game features. `FEATURES` is the model's input list.
- `src/nflpred/injuries.py` — injury-report feature (Out/Doubtful × prior snap share)
- `scripts/experiment.py` — harness: `run(name, feat_cfg, model_kind, features)` returns validation (2015-19) and holdout (2020-25) log loss. Pick changes on validation; report holdout.
- `src/nflpred/model.py` — `MarginModel` (production: ridge on point margin → win prob), baselines, backtest, explanations in points (`FACTOR_GROUPS`)
- `src/nflpred/pipeline.py` — CLI: `update`, `backtest`, `gate`
- `src/nflpred/travel.py`, `weather.py`, `odds.py`, `ngs.py` — context sources (travel/tz/body clock, Open-Meteo forecast, The Odds API multi-book lines, Next Gen Stats). Shown on picks; NOT model inputs (tested, no gain).
- `src/nflpred/news.py` — live injury/depth-chart news (Sleeper + ESPN public feeds) applied to upcoming games; tested offline with fixtures in tests
- `src/nflpred/bets.py` + `betting_rules.json` — paper-bet ledger (`history/paper_bets.json`), grading, CLV, pre-registered switch to live recommendations. NEVER edit rules or the validation test in place to fit results: bump `version` instead.
- `buy_costs.json` — measured per-book cost of buying half points (2023-25 alt lines); display only, buys never bet (research: never +EV)
- `src/nflpred/margins.py` — key-number-aware margin distribution (cover/push/win probabilities)
- `src/nflpred/spread_bets.py` + `spread_rules.json` — spread paper-bet track (separate ledger `history/paper_bets_spread.json`); `my_books.json` = books allowed for line shopping
- `src/nflpred/odds_history.py` + workflow `odds_backfill.yml` — one-time resumable download of 2020-25 historical lines (paid Odds API plan) into `data/historical_odds/` (committed), at the same moments the live pipeline runs
- `src/nflpred/ml_v2.py` + `moneyline_v2_rules.json` — moneyline v2 track (soft-book price vs sharp no-vig + model agrees); ledger history/paper_bets_ml_v2.json
- `moneyline_v3_rules.json` — same as v2 but fair price = median of Pinnacle/LowVig/BetOnline (`ml_v2.process_v3`, ledger paper_bets_ml_v3.json). Note: circasports/bookmaker never appear in the Odds API feed
- `src/nflpred/ml_v4.py` + `moneyline_v4_rules.json` + `margin_total.py`/`margin_total_model.json` — moneyline v4: allowed-book ML vs the sharp SPREAD-implied win prob (total-aware key-number model); bets only at Tue/Sun 14:10 and Fri 21:40 UTC windows
- `src/nflpred/totals.py` + `totals_wind_rules.json` + `totals_dist.json` — totals pricing (key-number total distribution) and the forecast-wind under track; ledger history/paper_bets_totals_wind.json
- `src/nflpred/night_west.py` + `night_west_rules.json` — night-game body-clock track (back the more western team in 7pm+ ET games, bet in the last 3 h); graded on cover rate (binomial), CLV informational; ledger history/paper_bets_night_west.json
- `src/nflpred/news_llm.py` — AI news reader (free RSS feeds → Claude Haiku → history/news_llm.jsonl with first-seen time); runs in the hourly watch when ANTHROPIC_API_KEY is set, else probes feed reachability. Logging only, no bets
- `src/nflpred/news_sources.py` — extra news sources + per-source accuracy: Bluesky (public AppView API, no login; seed reporters + weekly searchActors discovery + team-domain handles -> history/bluesky_accounts.json) and official team sites (Google News site: queries, FEEDS team_sites_1..4). Audit rows carry `evidence` (each signal's source); wrong calls are classified once by Claude (misread / stale_or_hedged / source_wrong -> history/news_errors.json); history/news_sources.json = per-source accuracy, weight Beta(4,1), muted at >= 10 judged and weight < 0.6 (muted signals logged but hidden). Labels only; news_rules.json v1 unchanged
- `src/nflpred/roster.py` — roster check of AI signals: full runs save Sleeper's roster to history/roster.json (normalized name → team/position/gsis/depth; nflverse players.parquet fallback, re-checked once Sleeper is in); each signal gets `verify` {roster: match/team_mismatch/not_found/ambiguous} + `team_verified` (team corrected only from a Sleeper roster); AI fields never changed. Dashboard assigns news to games by team_verified (✓ verified / team unclear)
- `src/nflpred/news_audit.py` + `news_rules.json` — outcome audit at full runs (history/news_audit.json): played/started vs nflverse snap counts + schedule QB ids, official final report agreement, lead time vs the report proxy (16:00 ET two days before; Wed for Thu games), 6 h consensus spread move around QB signals. Pre-registered promotion rule (v1, frozen 2026-10-02): AI QB signals (certainty >= 0.8, roster match, game-week relevant) may become QB-availability overrides only after >= 25 audited such units at >= 95% precision. `promotion_check` only reports; nothing is wired into predictions (test guards this). Never edit the rule in place: bump the version
- `pipeline watch` + `.github/workflows/odds_watch.yml` — hourly odds-only refresh for the price-sensitive tracks (no retraining; v1 tracks act only on full runs). Odds snapshots are saved as history/odds_*.json.gz; alerts for validated tracks go to output/alert_watch.md (issue step in the workflow)
- `news.log_first_seen` → history/news_log.jsonl: when we first saw each injury/QB status (to measure news-vs-line timing)
- `src/nflpred/grading.py` — v1 points grades A+..C (GRADING_VERSION 1): now shown only on SPREAD offers (labeled v1); bump the version to change the scheme
- `src/nflpred/grade_v2.py` + `grade_v2_model.json` — MONEYLINE grade v2 (GRADING_VERSION 2): frozen LightGBM predicted CLV of each side's best allowed-book price (research output/research/grade_v2.md; A+ >= 2.5%, A >= 1.5%, B >= 0.5%). Runs in `update` and `watch` (game['grade_v2'], grade_v2/predicted_clv on new ML paper bets). LABEL ONLY: never use it to qualify, veto or size v1-v4 bets; only A+ showed an edge. Never edit/refit grade_v2_model.json (verbatim copy of the frozen research model); a new model = new research + new version. Fail-safe: null grade if lightgbm/model missing
- `src/nflpred/grade_totals.py` + `grade_totals_model.json` — TOTALS grade (version 2): frozen LightGBM predicted CLV of each allowed-book over/under quote (research output/research/grade_granular.md Q2; A+ >= 0.0%, A >= -1%, B >= -2%). Features rebuilt from `live_odds.totals.all_quotes` + first saved odds snapshot within 7 days (parity with the research table checked exactly). LABEL ONLY (game['totals_grade'], totals_grade/predicted_clv on totals bets); only A+ showed an edge (+1.5% CLV 2023-25). Never edit/refit the model file. Fail-safe: null grade
- `src/nflpred/grade_aplus.py` + `grade_aplus_{ml,spread,totals}_rules.json` — A+ grade paper tracks (frozen 2026-10-02): bet EVERY A+ offer of the moneyline grade v2, the spread grade v1 (candidate copied from scripts/research/grade_v2_spread.py v1_eval: any snapshot, best blend-EV offer among research-filtered prices -145..+125) and the totals grade; flat 1u, one bet per game per market locked at the first A+ snapshot (full runs AND hourly watch), 10 min..7 days before kickoff. Graded with bets.grade / spread_bets.grade+closing_margins / totals.grade+closing_totals (side-aware); ledgers history/paper_bets_aplus_{ml,spread,totals}.json; per-game views game['aplus_ml'|'aplus_spread'|'aplus_totals']. The grades stay labels for every other track
- `src/nflpred/totals_early_under.py` + `totals_early_under_rules.json` — early-week under track (Tue 14:10 UTC window, 96 h-7 days out, under EV >= 0 vs the sharp fair total); NEW 2026 hypothesis (post-hoc in the grade study); CLV vs totals.closing_totals; ledger history/paper_bets_totals_early_under.json
- `src/nflpred/props_receptions.py` + `props_receptions_rules.json` + `player_stats.py` — receptions line-shopping props track (research rule rec_shop_early): per-event Odds API call (`odds.fetch_event_props`, ~1 credit per event) once at each game's window (Fri 21:40 UTC / kickoff-24h) and 10-90 min before kickoff for games with open bets; raw responses history/props_*.json.gz; state history/props_receptions_state.json; graded with nflverse weekly player stats (stats_player release) + snap counts (no offensive snap = void); ledger history/paper_bets_props_receptions.json
- `src/nflpred/devig.py` — vig removal (multiplicative, Shin, power). Since 2026-10-04 the moneyline tracks use Shin (moneyline v2/v3 rules v2, exchange_value v2, cfb_ml v2, cfb_shop v2): multiplicative overstated longshots (output/research/devig_methods.md, devig_tracks.json)
- `src/nflpred/kalshi_maker.py` + `kalshi_maker_rules.json` — simulated Kalshi resting (maker) bids at Pinnacle Shin fair − 3¢, NFL + college; fills from Kalshi public trades (only trades printed below the bid count); paper only (Kalshi's Arizona status contested)
- `src/nflpred/futures_value.py` + `futures_shop_rules.json` — Super Bowl price at Arizona books vs the all-book power-devig consensus (weak reference); ledger history/paper_bets_futures_shop.json
- `src/nflpred/picks_log.py` + `picks_rules.json` — published analyst picks (RSS + article text → Haiku) logged with first-seen time to history/picks_log.jsonl; scored vs the close at season end. Logging only
- `src/nflpred/picks_score.py` + `picks_hot_rules.json` — scores logged published picks (history/picks_log.jsonl) at the CLOSING line (NFL nflverse, college DK close via CFBD), line value = stated number vs close, per-picker records, and the pre-registered hot-picker test (v1, frozen 2026-10-05: 60%+ on 8+ picks over the prior 4 weeks -> follow next week; pass >= 200 followed picks, > 52.4% at p < 0.05 and beating the rest). picks_log also reads betting-site RSS (vsin, pickswise, covers, oddsshark; reachability in history/picks_probe.json) and Bluesky pick accounts (history/bluesky_pickers.json, weekly discovery). Not read: Action Network (robots.txt), Reddit (login, blocks cloud IPs). Output history/picks_scores.json -> dashboard 'Published pickers'. Logging only
- `src/nflpred/props_unders.py` + `props_unders_rules.json` — Tuesday star-unders props track (v1, frozen 2026-10-04; found in the 2024-25 holdout, output/research/props_full.md, so 2026 is its real test): at Tue 14:10 UTC +-50 min, one event call per game (rush yds, receptions, rec yds; 3 credits), UNDER on every player-market quoted by >= 2 books, best number then price at allowed books, main line only; graded on RESULTS (ROI p < 0.05 over >= 150 bets), not CLV; generic `player_stats.stat_value`; ledger history/paper_bets_props_unders.json; one summary push (notify.QUIET_TRACKS)
- `src/nflpred/qb_availability.py` — sit-probability blend for upcoming games with a hurt listed QB; `overrides.json` for late news
- `src/cfbpred/` — college football (page at site/cfb/): `fetch` (CFBD -> data/cfb/raw), `pipeline` (ratings, this week's games, odds view), `tracks` + `cfb_ml_rules.json` (college moneyline price-rule paper track, the only college bets), `odds_live`/`odds_history` (hourly odds), `news` (AI news reader, logging only, history/cfb/news_*.jsonl), `weather` (Open-Meteo: live kickoff forecast logged to history/cfb/weather_log.jsonl; 2021-26 forecast backfill via cfb_weather.yml -> data/cfb/weather). `dist` + `cfb_dist.json` (college key-number margin/total distributions, prices any number from the sharp line), `shop` + `cfb_shop_rules.json` (shop-vs-sharp paper track: spreads EV>=4%, totals/ML EV>=2% vs Pinnacle across AZ books, quarter-Kelly stake shown; ledger history/cfb/paper_bets_cfb_shop.json; research output/research/cfb/round2.md: all 12 pre-declared shop rules beat the close, openers-vs-ratings fail, QB news mostly priced by Sunday). Research in scripts/research/cfb (factor screen: all 10 situational factors fail; wind_screen.py pre-declared 2026-10-03 before the weather data existed)
- `scripts/build_dashboard.py` — renders `output/*.json` into `site/`
- `model_baseline.json` — backtest log loss that CI must not regress past

## Commands
```
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q                                   # includes leakage tests
python -m nflpred.pipeline gate             # backtest 2018-2025 + regression gate
python -m nflpred.pipeline update           # refresh data, predict next ~9 days
python scripts/build_dashboard.py
```

## Non-negotiable rules
1. **No leakage.** A feature for a game may only use results from games with an earlier `gameday`.
   Rolling stats must `shift(1)` before aggregating. `test_no_future_leakage` erases all results
   from a cutoff date onward and asserts features on that date don't change; every new feature
   must pass it. Never weaken or skip that test.
2. **Time-based validation only.** No random train/test splits, no k-fold across seasons. Tune on
   2015–2019; 2020–2025 is the holdout reported in the dashboard.
3. **Vegas lines are a benchmark, not a feature** in the main model. A separate "model + market"
   variant is fine if it's clearly labeled and scored separately.
4. **Judge models by log loss and Brier score**, not accuracy. Report next to Vegas.
5. If a change improves the backtest, update `model_baseline.json` in the same PR and say why in the
   PR description. If it's worse by more than 0.002 log loss, CI fails; don't raise the tolerance.
6. Keep explanations exact: per-game factor contributions must sum to the predicted margin
   (`test_explanations_sum_to_predicted_margin`). If you switch to a non-linear model, use
   SHAP and keep an equivalent test.

## Goal and how it's judged
The goal is an edge over the betting market. Closing lines are not beaten on this data (see README),
so judge progress by: (1) backtest log loss (CI gate), (2) `scripts/vs_vegas.py` blend test, which must
beat Vegas alone on 2020-2025 to claim new information, and (3) live closing line value in `clv` of
predictions.json. Never report a betting edge from in-sample or tuned-on-holdout results.

## Ideas backlog (good next PRs)
- Clinch-aware final-week feature (actual seeding scenarios) instead of the simple final-week interaction
- Once `history/odds_*.json` has a season of snapshots, measure CLV per book and per day of week
- Live formats checked 2026-09-29 via web fetch: ESPN summary `injuries` structure matches `parse_espn_summary` exactly; Sleeper field names match `parse_sleeper` (saw injury_status values IR/PUP; practice_participation values not yet observed live, confirm on first game-week run)
- Once paper bets accumulate: use our own line-movement history (history/odds_*.json) as a bet filter
- Already tried without gain (don't repeat without a new angle): garbage-time filter, opponent-adjusted EPA,
  weekly power ratings, recency weighting, QB/Elo parameter grids, gradient boosting,
  travel/time zones/body clock, weather interactions, Next Gen Stats (QB CPOE/TTT, RYOE, separation),
  special teams EPA, starter injuries by position group / OL clusters, QB draft-capital prior, QB time-off fade,
  new head coach, early-season interactions, referee tendencies, turf mismatch, conference game, Thu/Mon,
  home stand/road trip, bye flags, bounce-back/letdown (all 2026-09-29, see README table; code in
  scripts/extra_features.py). ADOPTED from that batch: coming off overtime (ot_diff).
  Also tried 2026-09-30 (README "Research round 2"): Wong teasers, line-movement model, QB-news timing,
  totals/wind (recorded-wind version leaks), Sunday-night openers, state-space/Kalman ratings.
- Spread CLV must use closing PRICES (spread_bets.closing_margins / edge_lab.closing_fair), never spread_line alone.
- Leads to paper-track: moneyline v2 (live), Tuesday forecast-wind unders (needs totals odds + logged forecast), hourly price checks.
- Measure "favorites early, dogs late" from history/odds_*.json once ~6 weeks of snapshots exist
- 2027 preseason: season-long prop unders (season_props_rules.json, pre-registered 2026-10-04; needs a data source, none of our feeds carry season props)
