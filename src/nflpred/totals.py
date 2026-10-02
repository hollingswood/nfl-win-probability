"""Totals (over/under): key-number-aware distribution of total points, live totals summary, and the
pre-registered "forecast wind -> under" paper track (totals_wind_rules.json).

Distribution: normal around the expected total x empirical weights for each integer total
(totals pile up on 37, 41, 44, 47, 51 ...), fit once on 2012-2019 closing totals vs results and
saved in totals_dist.json. Used to price half-points/pushes, turn a market line + no-vig prices
into a fair expected total (implied_mu), and value bets at the close (CLV).

Track idea (research round 2, output/research/totals.md): strong wind suppresses scoring and
early-week totals under-adjust. The historical test used RECORDED wind (too optimistic), so this
track uses the real Open-Meteo kickoff forecast at the time of the bet — the honest version.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import bets as ML

ROOT = Path(__file__).resolve().parents[2]
DIST_PATH = ROOT / "totals_dist.json"
RULES_PATH = ROOT / "totals_wind_rules.json"
KT = np.arange(0, 121)
MU_GRID = np.round(np.arange(20.0, 75.0001, 0.05), 2)


class TotalDist:
    def __init__(self, sigma: float, weights: np.ndarray):
        self.sigma, self.w = float(sigma), np.asarray(weights, float)
        P = self.pmf(MU_GRID)
        self.mean_at = (P * KT[None, :]).sum(axis=1)
        self.cdf = np.cumsum(P, axis=1)

    @classmethod
    def fit(cls, games: pd.DataFrame, smooth=10.0, cap=(0.2, 3.0)) -> "TotalDist":
        tr = games[games.season.between(2012, 2019) & games.home_score.notna() & games.total_line.notna()]
        tot = (tr.home_score + tr.away_score).values
        sigma = float((tot - tr.total_line.values).std())
        base = cls._normal(tr.total_line.values, sigma).sum(axis=0)
        act = np.bincount(np.clip(tot.astype(int), 0, KT[-1]), minlength=len(KT))
        return cls(sigma, np.clip((act + smooth) / (base + smooth), *cap))

    @staticmethod
    def _normal(mu, sigma):
        mu = np.atleast_1d(np.asarray(mu, float))[:, None]
        z = (KT[None, :] - mu) / sigma
        p = np.exp(-0.5 * z * z)
        return p / p.sum(axis=1, keepdims=True)

    def pmf(self, loc):
        p = self._normal(loc, self.sigma) * self.w[None, :]
        return p / p.sum(axis=1, keepdims=True)

    def _row(self, mu: float) -> int:
        loc = float(np.interp(mu, self.mean_at, MU_GRID))      # expected total -> location parameter
        return int(np.clip(round((loc - MU_GRID[0]) / 0.05), 0, len(MU_GRID) - 1))

    def probs(self, mu: float, point: float, side: str) -> tuple[float, float]:
        """(P(win), P(push)) of an over/under at `point` when the expected total is mu."""
        c = self.cdf[self._row(mu)]
        p_over = 1 - c[int(math.floor(point))]
        p_under = c[int(math.ceil(point)) - 1] if point == int(point) else c[int(math.floor(point))]
        push = max(0.0, 1 - p_over - p_under)
        return (p_over if side == "over" else p_under), push

    def ev(self, mu, point, price, side) -> float:
        w, pu = self.probs(mu, point, side)
        return w * ML.decimal(price) + pu - 1

    def implied_mu(self, point: float, nv_over: float) -> float:
        lo, hi = 20.0, 75.0
        for _ in range(40):  # P(over | no push) increases with mu
            mid = (lo + hi) / 2
            w, pu = self.probs(mid, point, "over")
            q = w / max(1 - pu, 1e-12)
            lo, hi = (mid, hi) if q < nv_over else (lo, mid)
        return (lo + hi) / 2

    def save(self, path: Path = DIST_PATH):
        path.write_text(json.dumps({"sigma": self.sigma, "weights": [round(float(x), 5) for x in self.w],
                                    "fit_on": "2012-2019 nflverse closing totals vs final totals"}))

    @classmethod
    def load(cls, path: Path = DIST_PATH) -> "TotalDist":
        d = json.loads(path.read_text())
        return cls(d["sigma"], d["weights"])


_DEFAULT: "TotalDist | None" = None


def default_dist() -> "TotalDist":
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TotalDist.load()
    return _DEFAULT


def _nv(over, under):
    io = -over / (-over + 100) if over < 0 else 100 / (over + 100)
    iu = -under / (-under + 100) if under < 0 else 100 / (under + 100)
    return io / (io + iu)


def market_total(rows, dist: TotalDist) -> float | None:
    """Median fair expected total over books. rows: (point, over_price, under_price)."""
    vals = [dist.implied_mu(pt, _nv(o, u)) for pt, o, u in rows if None not in (pt, o, u)]
    return round(float(statistics.median(vals)), 2) if vals else None


def summarize_event(ev: dict, allowed: set | None, sharp: set, dist: TotalDist) -> dict | None:
    rows, sharp_rows, by_book, all_q = [], [], [], []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "totals":
                continue
            t = {o["name"]: o for o in m.get("outcomes", [])}
            ov, un = t.get("Over"), t.get("Under")
            if not ov or not un or ov.get("point") is None:
                continue
            r = (ov["point"], ov["price"], un["price"])
            rows.append(r)
            # every book's raw quote (totals grade / early-under track apply the research filters themselves)
            all_q.append([bk.get("key"), bk.get("title", bk.get("key")), ov["point"], ov["price"], un["price"]])
            if bk.get("key") in sharp:
                sharp_rows.append(r)
            if allowed is None or bk.get("key") in allowed:
                by_book.append({"book": bk.get("title", bk.get("key")), "key": bk.get("key"), "point": ov["point"],
                                "over_price": ov["price"], "under_price": un["price"]})
    if not rows:
        return None
    return {"consensus_total": market_total(rows, dist), "sharp_total": market_total(sharp_rows, dist),
            "median_point": statistics.median(r[0] for r in rows), "totals_by_book": by_book, "all_quotes": all_q}


# ---------------------------------------------------------------- forecast-wind under track
def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def evaluate(game: dict, r: dict, dist: TotalDist, now: datetime | None = None) -> dict | None:
    q = r["qualify"]
    reasons = game["wind_check"] = []
    tot = ((game.get("context") or {}).get("live_odds") or {}).get("totals") or {}
    wx = (game.get("context") or {}).get("weather") or {}
    if not tot.get("totals_by_book") or tot.get("consensus_total") is None:
        reasons.append("no live totals")
        return None
    if wx.get("indoors", True) or wx.get("wind_mph") is None:
        reasons.append("indoors or no forecast")
        return None
    mu = tot["consensus_total"]
    best = None
    for b in tot["totals_by_book"]:
        ev = dist.ev(mu, b["point"], b["under_price"], "under")
        if best is None or ev > best["ev"]:
            best = {"book": b["book"], "book_key": b.get("key"), "point": b["point"], "price": b["under_price"],
                    "ev": round(ev, 4)}
    game["totals_view"] = {"forecast_wind_mph": wx["wind_mph"], "consensus_total": mu, "best_under": best}
    kick = pd.Timestamp(game["kickoff_utc"]) if game.get("kickoff_utc") else None
    now = now or datetime.now(timezone.utc)
    if wx["wind_mph"] < q["min_forecast_wind_mph"]:
        reasons.append(f"forecast wind {wx['wind_mph']:.0f} mph < {q['min_forecast_wind_mph']}")
    if kick is not None and (kick - pd.Timestamp(now)).total_seconds() / 86400 > q["max_days_before_kickoff"]:
        reasons.append("more than 6 days out (forecast too uncertain)")
    if not (q["min_american_odds"] <= best["price"] <= q["max_american_odds"]):
        reasons.append("price outside allowed range")
    if reasons:
        return None
    return {"id": f"{game['game_id']}:under", "track": "totals_wind", "rules_version": r["version"],
            "placed_at": now.isoformat(timespec="minutes"), "game_id": game["game_id"], "season": game["season"],
            "week": game["week"], "gameday": game["gameday"], "side": "under",
            "team": f"{game['away_team']}@{game['home_team']} UNDER", "opponent": "",
            "point": best["point"], "price": best["price"], "book": best["book"], "edge": best["ev"],
            "forecast_wind_mph": wx["wind_mph"], "consensus_total": mu, "units": r["sizing"]["units"], "status": "open",
            **_totals_grade_label(game, best)}  # totals grade: label only, never qualifies


def _totals_grade_label(game: dict, best: dict) -> dict:
    try:
        from . import grade_totals
        return grade_totals.bet_fields(game, "under", best.get("book_key"), best["point"], best["price"])
    except Exception:
        return {"totals_grade": None, "predicted_clv": None}


def _snapshots(history_dir: Path):
    for f in sorted(list(history_dir.glob("odds_*.json")) + list(history_dir.glob("odds_*.json.gz"))):
        stem = f.name.split(".")[0]
        try:
            ts = datetime.strptime(stem[5:], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)
            raw = gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text()
            yield ts, json.loads(raw)
        except Exception:
            continue


def closing_totals(history_dir: Path, dist: TotalDist) -> dict[tuple[str, str], list[dict]]:
    """Fair closing expected total from our last odds snapshot before each kickoff (prices included).
    `mu` = all-book median (the CLV reference of the totals tracks); `mu_sharp` = LowVig/BetOnline/... median
    (informational, None if no sharp book quoted)."""
    from . import odds as O
    best = {}
    for ts, events in _snapshots(history_dir):
        for ev in events:
            home, away = O.TEAM_ABBR.get(ev.get("home_team")), O.TEAM_ABBR.get(ev.get("away_team"))
            if not home or not ev.get("commence_time"):
                continue
            kick = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            s = summarize_event(ev, None, O.SHARP_BOOKS, dist) if ts < kick else None
            if not s or s["consensus_total"] is None:
                continue
            k = (home, away, ev["commence_time"])
            if k not in best or ts > best[k]["ts"]:
                best[k] = {"ts": ts, "mu": s["consensus_total"], "mu_sharp": s.get("sharp_total")}
    out = {}
    for (h, a, c), v in best.items():
        out.setdefault((h, a), []).append({"commence": c, "mu": v["mu"], "mu_sharp": v["mu_sharp"],
                                           "ts": v["ts"].isoformat(timespec="minutes")})
    return out


def grade(bet: dict, games: pd.DataFrame, dist: TotalDist, closes: dict) -> dict:
    g = games[games["game_id"] == bet["game_id"]]
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    x = g.iloc[0]
    total = x["home_score"] + x["away_score"]
    side = bet.get("side", "under")  # the A+ totals track (grade_aplus.py) also bets overs
    won = total < bet["point"] if side == "under" else total > bet["point"]
    res = "win" if won else "push" if total == bet["point"] else "loss"
    out = dict(bet, status="graded", result=res, final=f"{int(total)} pts")
    out["profit_units"] = round(bet["units"] * (ML.decimal(bet["price"]) - 1), 3) if res == "win" else (0.0 if res == "push" else -bet["units"])
    for c in closes.get((x.get("home_team"), x.get("away_team")), []):
        if abs((pd.Timestamp(c["commence"]).tz_convert(None).normalize() - pd.Timestamp(x["gameday"])).days) <= 1:
            out["clv"] = round(dist.ev(c["mu"], bet["point"], bet["price"], side), 4)
            out["closing_total"] = c["mu"]
            out["clv_source"] = f"own closing snapshot {c['ts']}, prices included"
            if c.get("mu_sharp") is not None:  # informational: vs the sharp books' close (research CLV reference)
                out["clv_sharp_close"] = round(dist.ev(c["mu_sharp"], bet["point"], bet["price"], side), 4)
    return out


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None) -> dict:
    r = r or load_rules()
    dist = default_dist()
    path = history_dir / "paper_bets_totals_wind.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    if any(b.get("status") == "open" for b in ledger):
        closes = closing_totals(history_dir, dist)
        ledger = [grade(b, games, dist, closes) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    new = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r, dist)
        if bet and g["game_id"] not in have:
            new.append(bet)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ML.record(ledger, r)
    for g in pred.get("upcoming", []):
        v = g.get("totals_view")
        if v:
            logged = any(b["game_id"] == g["game_id"] and b.get("status") != "void" for b in ledger)
            v["verdict"] = "bet" if logged else "pass"
            v["reasons"] = [] if logged else list(g.get("wind_check") or [])
    try:
        from . import grade_totals
        btg = grade_totals.by_grade(ledger)
    except Exception:
        btg = []
    return {"track": "totals_wind", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "by_grade": [], "by_totals_grade": btg, "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}
