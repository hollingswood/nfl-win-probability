# Blind evaluation (set up 2026-10-30)

New models are judged on data nobody has looked at, once. Code: `src/nflpred/blind.py`.

* **Sealed vault, NFL 2002-2011** (2001 warm-up). The data loaders refuse these seasons outside `blind.open_vault`.
  A model is registered first (`blind.register`, its pre-registration file hashed into `registry.json`), then scored
  once, walk-forward 2005-2011, next to the production model rebuilt the same way and the closing line. The result is
  permanent; a changed model needs a new name.
* **Prospective log.** Every full run appends each registered live candidate's pre-kickoff probabilities to
  `prospective/<name>.jsonl` (the production model is logged as `production`). Scores are revealed only at the
  dates in `checkpoints.json`, for games before each date (`revealed.json`).
