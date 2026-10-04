# Model Architecture, Timing and Evaluation for Beating NFL Lines

Scope note: about 20 search/fetch calls. Several suggested primary sources could not be reached: Pinnacle Betting Resources pages were blocked by the fetch tool's provenance rule, and Unabated, the Wizard of Odds key-number tables, Glickman/Wilkins papers and the books (Wong, Yao, Buchdahl's "Squares & Sharps") were not fetched. Those are listed under Gaps rather than reconstructed from memory. Anything marked "Inference" is my own arithmetic or reasoning, not a sourced claim.

## 1. Predict the market (close) vs predict the outcome; CLV-to-ROI conversion

### Takeaway
The evidence supports targeting the **closing line** rather than the game result. The bettor-friendly gap is opener to close. Closing prices from sharp books are close to calibrated, but they are not perfect, and positive CLV does **not** guarantee profit unless it exceeds the vig. For NFL sides at -110, CLV has to beat roughly a 4.5% margin, so "a little CLV" can still lose money.

### Cited Findings
- **NFL opener vs close accuracy (2002-2011, 2,560 games):** "no statistically significant difference" in predictive accuracy between opening and closing lines. More than 2,000 of the 2,560 games moved 1 point or less, and only about 20% moved more than 1 point. Closing line minus actual margin had mean -0.009 and SD **13.588 points**, stable from 1980 to 2011. — [Szalkowski & Nelson, ODU, arXiv 1211.4000](https://arxiv.org/pdf/1211.4000)
- **Same study:** blind home-underdog betting went 53.5% ATS in 2002-11 (breakeven 52.38%). The authors note the edge "has been reduced in later years" compared with pre-2002. This is a descriptive single-filter result with no multiple-testing control. — [arXiv 1211.4000](https://arxiv.org/pdf/1211.4000)
- **CLV ≠ guaranteed profit (Karl Whelan, economist):** NBA moneylines, 3,670 games (2022/23–2024/25), 34,944 odds quotes from licensed US books (The-Odds-API). Bets were sorted into deciles by CLV. Only the **top decile** made a meaningful profit (+11.4%). The 9th decile had about 5% CLV but only +0.2% return. Three of the five positive-CLV deciles lost money. The lowest decile lost 25%. Mechanism: book margin of about 4.5% means "you [can] get significant CLV and still lose on average." — [Karl Whelan, "The Truth about Closing Line Value"](https://www.karlwhelan.com/?p=2595)
- **Calibration of sharp-book prices as an EV yardstick (soccer, using Buchdahl's football-data.co.uk data, 22 European leagues):** betting soft books when they beat Pinnacle's *pre-closing* devigged price gave 3.6% actual ROI vs 3.8% expected over **31,247 bets** (2012/13–2025/26), which is near-perfect calibration. Measured against Pinnacle *closing* odds, 2019/20–2025/26 gave 3.5% actual vs 4.3% expected over 23,418 bets. The last two seasons gave **1.9% actual vs 4.3% expected over 6,806 bets**. The author concedes that about 7,000 bets "isn't necessarily enough to distinguish signal from noise when the expected ROI is a mere 4%." — [Networked substack, "A view from the Pinnacle"](https://networked.substack.com/p/a-view-from-the-pinnacle)
- **Industry framing that CLV is the "gold standard"** (affiliate/vendor sources, lower rigor): [SharpAPI CLV docs](https://docs.sharpapi.io/en/api-reference/historical-clv/), [OddsPapi EV/CLV in Python](https://oddspapi.io/blog/?p=2906), [Action Network CLV explainer](https://www.actionnetwork.com/education/closing-line-value-definition-importance-sports-betting). Whelan attributes the popularity of CLV advice partly to affiliate incentives. — [Whelan](https://www.karlwhelan.com/?p=2595)
- **Line movement direction as a signal:** TeamRankings publishes NFL ATS and over/under results split by opening line and line movement. These are descriptive trend tables, not validated strategies. — [BetIQ ATS by line movement](https://betiq.teamrankings.com/nfl/betting-trends/ats-results-by-opening-line-and-line-movement/), [BetIQ O/U by line movement](https://betiq.teamrankings.com/nfl/betting-trends/over-under-results-by-line-movement/)

### Inferences
- In a market-forecasting design, the target is `close_spread - open_spread`, or the devigged close probability, given information available at bet time. A bet on the opener is warranted only when **E[move in your favour] × (cover-prob per point) > vig cost**. Near a pick'em, normal SD ≈ 13.5 means 1 point ≈ 2.9–3.0 pp of cover probability (density φ(0)/13.5 ≈ 0.0295). Breaking even at -110 needs 52.38%, which is +2.38 pp over 50%. So you need an **expected favourable move of about 0.8 points off-key, or about 0.5 points if it crosses 3 or 7**, before any adverse selection or limit effects. This lines up with the project's finding: a correlation of 0.1–0.3 with line moves implies an expected move given the signal of only about 0.1–0.3 × SD(move). With most NFL moves ≤1 point (per Szalkowski & Nelson), that is usually well under 0.8 points.
- Whelan's decile result implies a thresholding rule: only the extreme tail of predicted CLV should be bet. Average CLV across all bets is a poor guide when the vig is around 4.5%.
- The fact that opener and close are about equally accurate for outcomes (Szalkowski & Nelson, 2002-11) does **not** mean the close is uninformative relative to the opener. The close can still be the better *price-forecast benchmark*. Outcome noise (SD 13.6) swamps a 0.5-point difference at n=2,560.

### Gaps
- Could not fetch Buchdahl's Pinnacle articles on CLV vs. ROI, or "Squares & Sharps" (the regression of yield on CLV). These are widely cited but unverified here.
- No sourced NFL-specific estimate of how often lines move *through* 3 or 7 from open to close. Szalkowski & Nelson only show that ~80% of games move ≤1 point (2002-11). Modern data is likely different (more efficient openers, earlier lookahead lines), and this should be computed from the project's own opener/close data.
- No sourced "pros require X points of predicted move" rule. Unabated's articles on this were not retrieved.

## 2. Residual modeling and Bayesian updating of the market prior

### Takeaway
The best-documented public NFL approach is to **regress the model heavily toward the market**. nfelo's optimum is about 65% market / 35% model, with the weight set dynamically from trailing relative error. In practice that means modeling only the residual. Academic work (Hubáček et al.) argues that what matters is making the model's errors **decorrelated** from the market, not raw accuracy.

### Cited Findings
- **nfelo market regression:** the optimal blend weight is **65% market / 35% model**, chosen by maximizing accuracy. The regressed model beats both the standalone model and the market on Brier score. Standalone, the model "only slightly underperforms the market in predicting margin" and beats 538. A **dynamic** weight, scaled by trailing exponentially-weighted squared error of model vs. market over the previous N games, improves on a flat weight. Published Nov 2020; N and the exact years are not disclosed. — [nfelo, "Using market regression to improve prediction accuracy"](https://www.nfeloapp.com/analysis/using-market-regression-to-improve-prediction-accuracy-in-the-nfl/)
- **nfelo self-reported track record 2009–2026:** 56.81% ATS vs. opener and 54.34% vs. close, model MAE 10.1, average CLV 6.11%, +169.4 units. Season ATS ranges from 46.9% to 69.2%. **Caveat:** these are self-reported, and much of 2009–2020 is presumably back-fitted (the model and regression weights were designed with that data visible), so this is not clean out-of-sample evidence. The gap between opener and closer results is itself consistent with the edge being in early-week prices. — [nfelo model performance](https://nfeloapp.com/games/nfl-model-performance/)
- **Decorrelation principle:** Hubáček & Šír argue a bettor "need not possess superior price prediction accuracy." Profit comes from building models whose estimation errors are decorrelated from the bookmaker's, because the market taker chooses which side to take. They propose optimizing for low correlation with market prices rather than pure accuracy. — [Hubáček & Šír, "Beating the market with a bad predictive model," arXiv 2010.12508](https://ar5iv.labs.arxiv.org/html/2010.12508)
- Companion peer-reviewed paper: Hubáček, Šourek & Železný, "Exploiting sports-betting market using machine learning," *International Journal of Forecasting* 35(2), 2019, pp. 783-796. It decorrelates model outputs from bookmaker odds; the results summary was not retrieved here. — [IDEAS/RePEc](https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html), [CTU page](https://ida.fel.cvut.cz/papers/hubacek2019exploiting.html)

### Inferences
- The project's result (a 2015-19 fitted logit blend had log loss 0.609 OOS vs. 0.607 for Vegas alone) matches what you would expect when model information is mostly already in the line. A static blend fitted in one regime can overweight the model. Two better designs: (a) fit **residual = margin − close** (or close − open) only on features plausibly unpriced at bet time, and (b) use a time-varying weight like nfelo's, shrunk hard toward 100% market.
- For an "open-time" model, use the **opener** (or the lookahead line) as the prior and the residual target. Using the close as a feature for an early-week bet is leakage.
- To apply decorrelation, check the partial correlation of model errors with market errors. A model that is a noisy copy of the market (high correlation) cannot produce edges, however good its MAE.

### Gaps
- No sourced quantitative schedule of how much weight the market deserves by day of week (Tuesday opener vs. Sunday close).
- Glickman/Stern-style Bayesian state-space NFL rating papers were not retrieved, so no sourced comparison of their accuracy vs. markets.

## 3. Required accuracy: MAE/RMSE vs. lines

### Takeaway
Closing-line errors are about 13.6 SD in margin (2002–11), and the market MAE is around 10–10.5. Public models that report MAE of about 10.1 are claiming near-market accuracy. Because outcome noise dominates, small MAE differences are statistically hard to detect, and MAE parity is neither necessary nor sufficient for profit (see decorrelation).

### Cited Findings
- Closing-line error SD 13.588 points, mean ≈ 0, 2002–2011, stable back to 1980. — [arXiv 1211.4000](https://arxiv.org/pdf/1211.4000)
- nfelo reports model MAE 10.1 (2009–2026) and says it "only slightly underperforms the market in predicting margin" before regression. — [nfelo performance](https://nfeloapp.com/games/nfl-model-performance/); [nfelo market regression](https://www.nfeloapp.com/analysis/using-market-regression-to-improve-prediction-accuracy-in-the-nfl/)
- A CMU capstone poster also compares model predictions to Vegas (content not fetched). — [CMU 495 Peterson poster](https://www.stat.cmu.edu/capstoneresearch/spring2020/495-Peterson-Poster.pdf)

### Inferences
- With a residual SD of about 13.5, the standard error of a MAE difference between two highly correlated predictors over ~270 games/season is small relative to their *difference*, but tiny MAE gaps (0.05–0.1 point) still need several seasons to resolve. A paired test on per-game absolute-error differences is the right tool, not comparing two MAEs.
- Matching the close's MAE is not the bar. What matters is whether the model's **disagreement** with the line predicts the residual (margin − line) with a slope well above 0. In practice, regress (margin − line) on (model − line). Profit requires a slope large enough that the predicted residual, converted to cover probability, exceeds 2.38 pp on the bets taken.

### Gaps
- The "typical NFL closing MAE ≈ 10.3–10.5" figure and the opener MAE in recent seasons were not found in a fetched source. Compute them from the project's own data.
- Wilkins/Glickman and inpredictable "how good do you need to be" posts were not retrieved.

## 4. Key numbers and margin distributions

### Takeaway
NFL margins are lumpy. 3 is about 14–15% of games, and 7 is second. 6, 10, 14, 4 and (since the 2015 PAT change) 5, 6 and 8 matter more. Converting a point edge to cover probability needs a **discrete margin distribution conditional on the spread**, not a normal curve. Totals have "key ranges" rather than sharp key numbers.

### Cited Findings
- 2003–2020: "three is the most common margin, followed by seven, six and 10". 3-point margins are about **14.8%** of games historically. In 2015–2019, margins of 5 and 6 rose while 4 and 10 fell. Half-point positioning at 3 (-2.5/+3.5) is called out as especially valuable. — [Action Network, NFL key numbers](https://www.actionnetwork.com/nfl/nfl-key-betting-numbers-spread-margins-of-victory-line-value)
- Since the 2015 PAT move to the 15-yard line, PAT attempts fell from 95.2% to 90.7% of TDs. Six- and eight-point games rose a combined **2.4%** over the nine seasons after vs. the nine before, and five-point games rose 1.4% (more two-point attempts). "Three and seven are still king." Totals have "key ranges" of about 3–4% each: 36–37, 39–41, 43–51, 54–55. 2025-26 scoring averaged 44.7 points. — [Covers, NFL key numbers 2025-26](https://www.covers.com/nfl/key-numbers)
- **nfelo spread-to-margin method:** a baseline distribution centred on the spread, plus key-number distributions weighted by proximity to the spread, a "super gaussian" weighting, and a binary-outcome multiplier. The distribution is forced to put 50% on each side of the spread. The home average margin used is 2.1. — [nfelo, margin probabilities from NFL spreads](https://www.nfeloapp.com/analysis/margin-probabilities-from-nfl-spreads/)
- Other key-number tables and half-point calculators (secondary sources): [WalterFootball margins](https://walterfootball.com/nflmargins.php), [OddsPapi "what a half point actually costs"](https://oddspapi.io/blog/?p=3165)

### Inferences
- Build P(margin = k | spread) empirically, smoothed by spread bucket, era-adjusted for post-2015 and post-2023 scoring. Price every bet as P(win) − P(lose) with an explicit P(push). A 0.5-point edge through 3 is worth several times one through 4–5 or 11–12.
- In a market-forecasting design, the payoff from predicting a move from 2.5 to 3.5 is far larger than from 4.5 to 5.5. Key-number crossings should get their own target or weighting.

### Gaps
- Exact current push probabilities by spread (for example P(push | -3) ≈ ?) and frequencies at 41/43/44/37/47/51 for totals were not retrieved from a primary table (Wizard of Odds was not fetched).

## 5. Model selection, overfitting and sample sizes

### Takeaway
Outcome-based ATS testing needs thousands of bets to detect a 1–3 pp edge. CLV-based evaluation is far more statistically efficient, but it only works if the closing benchmark is itself well calibrated, and that calibration appears to have degraded recently even at Pinnacle. Testing many models against closing lines demands multiple-comparison control.

### Cited Findings
- About 7,000 bets at 4% expected ROI "isn't necessarily enough to distinguish signal from noise." Calibration of Pinnacle closing odds has drifted: 1.9% actual vs. 4.3% expected recently. — [Networked substack](https://networked.substack.com/p/a-view-from-the-pinnacle)
- Average CLV can be positive while returns are negative in most CLV buckets. Evaluate the **distribution** of CLV, not the mean. — [Whelan](https://www.karlwhelan.com/?p=2595)
- Self-reported long-run records (nfelo, 2009–2026) mix back-fit and live periods, and season-level ATS ranges from 46.9% to 69.2%, which shows how large single-season variance is. — [nfelo performance](https://nfeloapp.com/games/nfl-model-performance/)
- A systematic review of ML in sports betting covers methods and evaluation pitfalls (not read in detail). — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)

### Inferences (my arithmetic)
- ATS test, one-sided α = 0.05, power 0.8. Detecting a true 54.5% vs. 52.38% breakeven (2.1 pp) needs n ≈ (1.645+0.84)² × 0.25 / 0.021² ≈ **3,500 bets**. Detecting 53.5% vs. 52.38% needs about 12,000. One NFL season has about 270 games.
- CLV test: if per-bet CLV in points has SD of about 1 point (most moves ≤1 point per Szalkowski & Nelson), then detecting a mean CLV of +0.3 points needs only about 70–100 bets. That is why CLV is used as the evaluation metric. But (a) a positive mean is not enough (Whelan), and (b) the convertibility of CLV to ROI should be checked on your own NFL data.
- Multiple testing: the project tested "many team/player models." With k models, use Bonferroni (α/k) or Benjamini–Hochberg FDR on per-model p-values from walk-forward OOS results. Better still, freeze a single pre-registered model plus a final holdout season, or test sequentially (SPRT on CLV) going forward.
- Walk-forward discipline: refit blend weights only on data before each season (or week). The 2015-19 → 2020-25 blend failure is the expected signature of regime shift plus overfitting a static weight.
- Kelly sizing: with an estimated edge that has large uncertainty, fractional Kelly (for example ¼–½) or shrinking the edge toward 0 before sizing is standard practice. (No source retrieved; see Gaps.)

### Gaps
- No retrieved source on Kelly sizing under parameter uncertainty (for example Baker & McHale 2013 on "optimal betting under parameter uncertainty"; not verified here).
- No retrieved Unabated/Pinnacle guidance on the number of bets needed to validate CLV.

## 6. Ensembling, priors, time decay, deep learning

### Takeaway
The only sourced NFL evidence here is that **market-regressed** ensembles beat standalone models, with a dynamic weight beating a static one. Decorrelated, independent information is what adds value. No reliable evidence was found that deep learning on play-by-play beats NFL closing lines.

### Cited Findings
- Market-regressed model beats both the model and the market on Brier score. Dynamic weighting by trailing error improves on a fixed weight. — [nfelo market regression](https://www.nfeloapp.com/analysis/using-market-regression-to-improve-prediction-accuracy-in-the-nfl/)
- Optimizing for decorrelation from bookmaker odds, not accuracy, is the academically argued route to profit. — [Hubáček & Šír, arXiv 2010.12508](https://ar5iv.labs.arxiv.org/html/2010.12508); [Hubáček, Šourek & Železný, IJF 2019](https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html)
- A systematic review of ML in sports betting exists (arXiv 2410.21484), but its NFL-specific market-beating evidence was not extracted. — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)

### Inferences
- Combining several independent ratings (Elo/EPA-based/QB-adjusted) helps only to the extent their errors are uncorrelated with each other **and with the line**. Ratings-of-ratings mostly re-derive the market.
- Preseason priors (win totals, lookahead lines) with in-season Bayesian updating are a natural fit for the "market as prior" architecture, because season win totals are themselves market prices.

### Gaps
- Massey composite performance vs. NFL lines, Glickman–Stern state-space results, and neural-net/PBP studies against closing lines were not retrieved. No reliable evidence was found either way.

## 7. Live, in-game and derivative markets

### Takeaway
Practitioners widely argue that player props and other derivative markets are softer than sides and totals: fewer pricing resources, slower news reaction, and low limits as a tell. That argument comes with low limits, and the sourced evidence is mostly opinion, not rigorous ROI studies.

### Cited Findings
- Sides and totals are "incredibly efficient," and props are less so because books cannot monitor hundreds of props. Props are a marketing tool. Props react to injury news more slowly than main markets. Winning prop bettors get limited to "$200-300 (at most)… some books… $10 or fewer," which the author reads as books knowing props are beatable. This is opinion from a prop-betting content site. — [Jack Miller, Establish The Run](https://establishtherun.com/miller-why-prop-betting-is-profitable/); see also [ETR on the NFL prop ecosystem](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/)
- Alternate-line pricing and buying points around key numbers (secondary sources): [DeucesCracked alt-lines guide](https://www.deucescracked.com/blog/alternate-lines-betting-guide-buying-selling-points), [TheLines alt lines](https://www.thelines.com/alternate-lines/)

### Inferences
- For this project, the most plausible model-based edges are: (a) early-week openers or lookahead lines, where nfelo's results vs. opener are better than vs. close; (b) props, team totals and halves, priced off the main line with a model-driven margin and scoring distribution; and (c) alt lines and SGPs where the book's correlation or key-number pricing is crude. All of these trade edge for low limits.

### Gaps
- No rigorous (peer-reviewed or large-sample) evidence was found on NFL prop, live or SGP market efficiency. Live-betting model evidence was not retrieved.
- CFB differences: not researched. Expect lower efficiency and limits, wider margin distributions and weaker key numbers, but this is unsourced.
