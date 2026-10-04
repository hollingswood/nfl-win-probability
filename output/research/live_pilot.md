# NFL live (in-game) betting pilot, 2022-25

Rules were committed before running (scripts/research/live_pilot.py). Data: in-game prices caught by the hourly odds snapshots (only quotes the book had updated within 3 minutes), matched to the nflverse play-by-play game state and nflfastR's win probability. 3,444 in-game snapshots across 1,112 games.

## Result: no edge; the apparent one is a timing artifact

On the pre-declared alignment (game state at the snapshot time), betting the side nflfastR liked by 6+ points looked spectacular: 783 bets, ROI +21.1%; at 10+ points, ROI +34.3%. By its own pre-declared bar it would pass (the JSON's pass flag reads false only because of a bug, since fixed, that treated p = 0 as missing). But it is the signature of stale quotes, not a betting edge: a play happens, the book suspends or hasn't repriced, and its last price sits in the snapshot while the play-by-play already knows the result.

Robustness check (added after seeing the result, so it can only kill the finding, not create one): line the game state up with each book's own last update instead.

| Game state used | Bets (gap ≥ 6 pts) | ROI | Bets (gap ≥ 10 pts) | ROI |
|---|---|---|---|---|
| At the snapshot time (pre-declared) | 999 | +9.4% ± 3.4% | 697 | +21.0% ± 4.4% |
| At the book's last update | 834 | +3.9% ± 3.4% | 399 | +5.9% ± 5.4% |
| 60 seconds before the book's last update | 866 | −0.1% ± 3.5% | 493 | −2.5% ± 4.9% |

Once the model only knows what the book knew, the edge is gone. Live prices are about as well calibrated as nflfastR (log loss market 0.4926 vs nflfastR 0.4754, both including the stale-quote moments). Fading early moves (L3: 367 bets, ROI +16.1% ± 11.2%) and following late moves (L4: 410 bets, ROI +5.1% ± 4.3%) were not significant either.

The only "live edge" in this data is being faster than the book after a play, which is latency betting: books delay and void such bets and limit accounts that do it. Not pursued. Not financial advice.
