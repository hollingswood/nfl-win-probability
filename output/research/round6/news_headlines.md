# Headline tone vs the closing spread (round 6, pre-registered) — INVALID: leakage

Pre-registration: news_headlines_prereg.json (Google News RSS per team-game, 2023 dev / 2024-25 holdout, VADER tone, items published >= 2 h before kickoff).

| Check (holdout 2024-25, n 568) | slope | t |
|---|---|---|
| As pre-registered (>= 2 h before kick) | 12.6 | 2.81 |
| minus titles holding this game's final score | 9.9 | 2.45 |
| minus any score/recap-like title | 4.9 | 1.23 |
| only items >= 30 h before kick | 0.6 | 0.21 |
| >= 30 h, no score/recap titles | 0.2 | 0.06 |

Why: Google News gives most items a day-level timestamp (95.5% exactly on the hour). Recaps of the game itself
("Lions beat Bears 31-17") then look like they were published hours before kickoff: 2,761 kept titles contain this
game's final score, 2,712 of them dated 2-24 h pre-kick. The "signal" is the result leaking in.

Verdict: headline tone carries nothing beyond the closing spread once the leak is removed. T4 (betting rule) not run.
Lesson for any future news-timing study: never trust a feed's publish time at day resolution; use our own first-seen
time (history/news_*.jsonl seen_at), which is what news_features.py and the blind news candidates already do.
