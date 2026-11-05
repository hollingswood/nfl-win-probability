# Published-picker consensus (round 7, pre-registered) — no edge

Pre-registration: consensus_prereg.json (committed before any result). Data: every pick the scorer counts, NFL + college,
graded games to 2026-11-05. One row per game x class (side = spread + moneyline pooled by team; total = over/under);
majority direction graded at the closing line, pushes dropped. Break-even 52.4% at -110.

| | games | majority covered | ROI at -110 |
|---|---|---|---|
| NFL sides | 57 | 47.4% | -9.6% |
| NFL totals | 47 | 53.2% | +1.6% |
| College sides | 170 | 51.8% | -1.2% |
| College totals | 98 | 48.0% | -8.4% |
| **All** | **372** | **50.3%** | **-4.0%** |

T1 (primary): cover rate vs net consensus (pickers for minus against): slope -1.2 points per extra net picker,
p = 0.067 (two-sided) -> FAIL. If anything, bigger consensus did slightly WORSE.
T2 grid: 4 cells with 20+ games; none significant after Bonferroni (alpha 0.0125) and none replicated dev -> holdout. No leads.

Descriptive views (not tests): unanimous 3+ pickers 62% (n 37, p 0.14); unanimous 5+ 50% (n 12); net >= 5: 43% (n 91);
lopsided (80%+ share, 4+ pickers): 34% (n 50, p 0.02 - one of 8 views, not significant after correction);
contested (2+ against): 47% (n 114).

Reading: published pickers agreeing tells you nothing the closing line doesn't; heavy agreement may be a mild fade
signal (consistent with the public piling onto popular sides) but this sample can't separate that from noise.
