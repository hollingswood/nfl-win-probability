"""Bet grades (A+ ... C), version 1, defined 2026-09-29 before any paper results.

A grade summarizes how much to trust an edge, not just how big it is. The backtest showed that the
biggest model-vs-market disagreements were NOT the best bets, so a large edge alone can't earn an A.

Points (each with a stated reason; nothing here was tuned to results):
  +1  edge at best price >= 3%          +2 if >= 5% (capped: bigger isn't more reliable)
  -1  model far from the market          spread >= 7 pts apart / moneyline >= 20 pct pts apart
                                         (historically those gaps were usually model errors)
  +1  line moved our way since first seen (the market is agreeing with us)
  -1  line moved against us since first seen
  +1  spread on the right side of 3 or 7  (underdog +3/+3.5/+7/+7.5, favorite -3/-2.5/-7/-6.5)
  -1  spread on the wrong side of 3 or 7  (underdog +2.5/+6.5, favorite -3.5/-7.5)
  -1  either starting QB differs from his team's usual starter (the model's weakest spot)

Score -> grade: >=4 A+, 3 A, 2 B+, 1 B, 0 C+, <=-1 C.
Whether grades mean anything is measured on the paper record (performance by grade) and shown
on the dashboard; changing this scheme means bumping GRADING_VERSION.
"""
from __future__ import annotations

GRADING_VERSION = 1
LETTERS = [(4, "A+"), (3, "A"), (2, "B+"), (1, "B"), (0, "C+")]
QB_CHANGE_FLAG = 0.05  # EPA/dropback gap between today's starter and the team's usual QBs


def letter(score: int) -> str:
    for cut, g in LETTERS:
        if score >= cut:
            return g
    return "C"


def key_number_side(point: float) -> int:
    """+1 right side of 3/7, -1 wrong side, 0 neither. `point` is the bettor's own spread."""
    if point in (3, 3.5, 7, 7.5, -3, -2.5, -7, -6.5):
        return 1
    if point in (2.5, 6.5, -3.5, -7.5):
        return -1
    return 0


def grade(market: str, edge: float, disagreement: float, moved: float | None,
          point: float | None = None, qb_flag: bool = False) -> dict:
    """market: 'spread' (disagreement/moved in points) or 'moneyline' (in probability points, 0-100).
    moved: + = market moved toward our side since first seen, None = no earlier snapshot."""
    pts, why = 0, []
    if edge >= 0.05:
        pts += 2; why.append(f"+2 edge {edge:+.1%}")
    elif edge >= 0.03:
        pts += 1; why.append(f"+1 edge {edge:+.1%}")
    big = 7 if market == "spread" else 20
    if abs(disagreement) >= big:
        pts -= 1; why.append(f"-1 model {abs(disagreement):.0f} {'pts' if market == 'spread' else 'pct pts'} from market")
    thresh = 0.5 if market == "spread" else 1.5
    if moved is not None and moved >= thresh:
        pts += 1; why.append("+1 line moved our way")
    elif moved is not None and moved <= -thresh:
        pts -= 1; why.append("-1 line moved against us")
    if market == "spread" and point is not None:
        k = key_number_side(point)
        if k:
            pts += k; why.append(f"{k:+d} {'right' if k > 0 else 'wrong'} side of 3/7")
    if qb_flag:
        pts -= 1; why.append("-1 QB change")
    return {"grade": letter(pts), "score": pts, "why": why, "grading_version": GRADING_VERSION}


def by_grade(ledger: list[dict]) -> list[dict]:
    """Paper-record performance per grade (the validation of the grades themselves)."""
    order = ["A+", "A", "B+", "B", "C+", "C"]
    rows = []
    for g in order:
        bs = [b for b in ledger if b.get("status") == "graded" and b.get("grade") == g]
        if not bs:
            continue
        staked = sum(b["units"] for b in bs if b["result"] != "push")
        profit = sum(b["profit_units"] for b in bs)
        clv = [b["clv"] for b in bs if "clv" in b]
        rows.append({"grade": g, "bets": len(bs), "wins": sum(b["result"] == "win" for b in bs),
                     "losses": sum(b["result"] == "loss" for b in bs), "profit_units": round(profit, 2),
                     "roi": round(profit / staked, 4) if staked else 0.0,
                     "avg_clv": round(sum(clv) / len(clv), 4) if clv else None})
    return rows
