# College football price screen (pre-declared 2026-10-03), NCAAF odds 2021-2025 (+2026 to date)

Your books: DraftKings, FanDuel, theScore Bet. Fair price is Pinnacle no-vig. One bet per game per rule. Pass bar: p < 0.0125.

| Rule | Bets | CLV vs Pinnacle close | ROI | Seasons with CLV > 0 | 2026 so far | Verdict |
|---|---|---|---|---|---|---|
| P1 soft books vs Pinnacle moneyline (EV ≥ 2%) | 472 | +2.2% (p < 0.0001) | +1.5% ± 6.6% | 5/5 | 8 bets, +2.7% | **pass** |
| P2 moneyline vs Pinnacle spread (EV ≥ 2%, ≥ 0 vs ML) | 855 | +1.8% (p < 0.0001) | +6.8% ± 4.9% | 5/5 | 42 bets, +2.2% | **pass** |
| P3 early unders (96-200 h) | 16 | +3.9% | +11% ± 25% | 3/4 | 3 | too few bets |
| P4 follow Pinnacle early (≥ 1 pt off US median) | 182 | +0.60 pts (p 0.0002) | −0.6% ± 7.0% | 5/5 | 22 bets, −17% ROI | fail: CLV did not turn into profit |

## Why P1 and P2 are only candidates
- CLV shrank: P1 +3.7/2.7/2.8% in 2021-23, then +0.2/0.9% in 2024-25; P2 +2.6/2.5/2.2% in 2021-23, then +1.3/0.6%.
- Research snapshots were 2-6 per day, so the "close" is the last snapshot before kickoff, not the true close.
- Frozen as one paper track, `cfb_ml` (cfb_ml_rules.json). It runs on every live NCAAF snapshot.
