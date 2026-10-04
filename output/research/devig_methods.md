# Which vig-removal method fits Pinnacle closing moneylines best?

Log loss of the home team's win probability from Pinnacle's last pre-kickoff moneyline, by de-vig method (output/research/devig_methods.csv).

| Set | Games | Multiplicative (ours) | Additive | Power | Shin |
|---|---|---|---|---|---|
| NFL 2022-25 | 1,141 | 0.60701 | 0.60652 | **0.60630** | 0.60652 |
| CFB 2021-26 | 4,194 | 0.52469 | **0.52418** | 0.52445 | **0.52418** |

Multiplicative, the method every track uses today, is the worst of the four in both sports, as the literature says, but only by about 0.0005. The difference is concentrated in lopsided games: multiplicative gives the underdog too much. Example: Pinnacle −350/+290 gives the dog 24.8% multiplicative vs about 24.1% with power. On a +300 soft-book price that is about 2.6 points of fake EV. Spreads and totals (near 50/50) are unaffected. The college shop track caps moneylines at +200, which limits the exposure. ML v2 (up to +1000) and the college ML track (up to +300) are more exposed. Frozen rules are not edited; a fix means new versions of those tracks with Shin or power devig.
