# College long-shot moneylines vs Pinnacle (2021-25, post-hoc check)

Question: are +1000 / +2500 college dogs that beat Pinnacle's no-vig (Shin) price good bets? The shop track only
bets -200 to +200, so long shots were never part of its validation. Script: `scripts/research/cfb/longshots.py`.

Arizona-book dog prices +300 or longer, 2%+ above Pinnacle's Shin fair price (first such quote per game and side):

| Price | Bets | EV at bet | Fair win % | Actual win % | ROI ± SE | CLV vs Pinnacle close |
|---|---|---|---|---|---|---|
| +300 to +500 | 142 | +4.6% | 20.9% | 17.6% | −17.7% ± 15.1 | +3.8% ± 1.0 |
| +500 to +1000 | 177 | +5.7% | 12.9% | 9.0% | −25.3% ± 18.3 | +2.9% ± 0.8 |
| +1000 to +2000 | 163 | +8.3% | 7.4% | 6.1% | −11.7% ± 27.5 | +4.1% ± 1.0 |
| +2000 and up | 44 | +19.7% | 3.9% | 2.3% | −18.2% ± 81.8 | +16.5% ± 4.4 |

They beat the close, but they won less often than the fair price said in every bucket: 27 wins against 37 expected for
+500 and longer, about 1.7 standard errors short. That isn't proof they lose, but it is not an edge either. Pinnacle's
own calibration for 3-6% dogs (24-72 h out) was also off (4.6% fair, 2.9% actual, 175 games). Verdict: no long-shot
track; the college board now shows prices outside -200..+200 as ungraded and NO.
