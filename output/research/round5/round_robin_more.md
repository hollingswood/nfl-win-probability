# Round robins on every remaining pool (pre-registered 2026-10-16): no reliable pass

Rules: `round_robin_more_prereg.json` (same method as `round_robin_grid_prereg.json`). Code: `round_robin_more.py`,
`round_robin_matched.py`. Legs rebuilt from the historical odds with the frozen grade models (NFL A+ ML / spread / totals),
the college moneyline track rule (P1/P2), and the college shop legs; cached in `legs/`.

Correction made after the first run: 'all picks' round robins only exist in weeks with at least k picks, but were
compared with singles over every week. The pre-registered comparison is 'the same picks as singles', so the verdicts
below use singles on exactly the same weeks (`round_robin_matched.json`). First-run verdicts are kept in `round_robin_more.json`.

| pool | legs 2021-25 | picks/week (median) | singles ROI dev / holdout | singles CLV dev / holdout | formats passing (of 45) |
|---|---|---|---|---|---|
| 1 NFL A+ (ML + spread + totals) | 620 | 6 | +13.5% / +13.0% | +2.6% / +1.9% | 0  |
| 2 College A+ only | 170 | 2 | +6.8% / +6.3% | +8.7% / +3.7% | 4 (N=5 by 2, N=6 by 2, N=6 by 3, N=6 by 2-6) |
| 3 College moneyline track | 937 | 9 | +7.1% / +2.2% | +2.5% / +1.3% | 0  |
| 4 All college tracks | 1389 | 13 | +4.3% / +0.8% | +3.4% / +1.3% | 0  |
| 5 All A+ (college + NFL) | 790 | 7 | +11.6% / +12.2% | +4.4% / +2.2% | 0  |
| 6 Everything (all college + NFL A+) | 2009 | 19 | +7.2% / +4.6% | +3.1% / +1.4% | 0  |
| 0 College shop vs sharp (first test) | | | | | 0 |
| 7 Ordinary NFL legs (closing spreads, -110) | | | | | 0 (of 22) |

## Reading it
- The only passes are college A+ round robins of the top 5-6 picks, which exist in 4 and 6 weeks (top 5) and 3 and 3 weeks (top 6) of the two periods. With 315 formats tried, 4 passes resting on 3-6 weeks each are what chance produces; not evidence.
- With A+ legs, parlays DO raise the expected return (CLV-implied): e.g. all A+ legs (college + NFL), 2023-25: singles +2.1%, by 2s +4.3%, by 3s +6.7%, by 2-6 +8.0%. Two +2% edges compound to about +4%. The cost is variance: largest losing stretch 2.3x singles (by 2s) to 5x (by 2-6), and 47-75% of weeks lose. That is why nothing passes the pre-registered bar (better results AND no more than 1.5x the drawdown).
- Realized results flatter the A+ pools: NFL A+ singles made +13% per bet (+18% to +21% averaged by week) on +1.9% to +2.7% CLV, far above what the closing line implies (luck in a 2-3 season sample); parlays multiply that luck.
- Pools with thinner edges (college moneyline track, all college tracks) turn negative as legs are added; ordinary legs lose more with every leg.
- NFL legs in 2021-22 are in-sample for the grade models (declared in advance); 2023-25 is clean.
