# College situational factor screen (2014-2025)

9085 FBS-vs-FBS games with a closing line (Prediction Tracker archive). Each factor was declared before looking, bet against the closing spread at -110. Pass bar: Bonferroni p < 0.005 over 10 rules, a positive ROI in most seasons, and holding up in 2022-25.

| Factor | Bets | Cover | ROI at -110 | p | Seasons > 52.4% | 2022-25 cover | Verdict |
|---|---|---|---|---|---|---|---|
| F1 long trip >= 1500 mi -> home | 536 | 50.6% | -3.5% | 0.801 | 5/12 | 51.4% | fail |
| F2 body clock: away >= 2 h west, kickoff < 1pm local -> home | 77 | 54.5% | +4.1% | 0.352 | 7/12 | 56.7% | fail |
| F3 altitude >= 1500 m vs away < 600 m -> home | 129 | 54.3% | +3.6% | 0.334 | 6/12 | 52.3% | fail |
| F4 rest edge >= 4 days -> rested team | 1623 | 50.5% | -3.5% | 0.933 | 6/12 | 50.7% | fail |
| F5 letdown after 10+ pt upset win -> fade | 416 | 50.5% | -3.6% | 0.781 | 6/12 | 54.6% | fail |
| F6 in-state conference game -> underdog | 699 | 48.5% | -7.4% | 0.980 | 4/12 | 45.4% | fail |
| F7 favorite by 21+ -> underdog | 1347 | 51.3% | -2.1% | 0.786 | 6/12 | 51.2% | fail |
| F8 home underdog | 3448 | 49.7% | -5.1% | 0.999 | 0/12 | 51.4% | fail |
| F9 late-season dog 1 h west of home -> that dog | 342 | 53.5% | +2.1% | 0.338 | 7/12 | 58.6% | fail |
| F10 preseason prior wk2-6 prior vs close gap >= 7 | 1118 | 48.2% | -8.0% | 0.997 | 2/11 | 47.8% | fail |

**Result: all 10 fail.** Travel, time zones, altitude, rest, letdowns, rivalries and big favorites are already priced into college closing lines. None is a model input. The few above 53% (body clock, altitude, late-season dog travelling west) are too small to separate from luck. The body-clock rule (F2) is paper-tracked anyway at Tyler's request (body_clock_rules.json), labeled unvalidated.

## Spread and total price rules (P5, P6)

Same idea as the moneyline price rules that passed (soft-book price better than the sharp no-vig fair price), applied to spreads and totals at the same number.

| Rule | Bets 2021-25 | Price CLV | ROI | Verdict |
|---|---|---|---|---|
| P5 spreads | 23 | +3.6% | -12.3% | too few bets to judge |
| P6 totals | 12 | +1.0% | -50.0% | too few bets to judge |

Books rarely hang a college spread or total at the same number as the sharp books with a better price, so these rules almost never fire. These two rules are not tracked; the college paper tracks are shop-vs-sharp, the moneyline track (P1/P2) and the unvalidated body-clock rule.

Not financial advice.
