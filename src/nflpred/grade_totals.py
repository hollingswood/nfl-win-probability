"""Totals grade (TOTALS GRADING_VERSION 2): predicted closing-line value of a live over/under offer. LABEL ONLY.

Research: scripts/research/grade_granular.py (Q2) -> output/research/grade_granular.md. The frozen LightGBM model
(fit on 2020-22, one-shot holdout 2023-25) is copied verbatim to `grade_totals_model.json` at the repo root (never
edit/refit it; a new model = new research + new version). Letters on predicted CLV: A+ >= 0.0%, A >= -1%, B >= -2%,
else C (a -110 quote at the consensus number is about -4.5%). Holdout 2023-25: A+ +1.5% ± 0.7 CLV (p 0.012), A -1.3%,
B -2.0%, C -2.6%. Only A+ has shown an edge, so the grade never qualifies, vetoes or sizes a bet.

Features (tfeaturize, 'small' set) are rebuilt at bet time from live data exactly as in the research:
  * every book's raw totals quote at this snapshot (`live_odds.totals.all_quotes`), research filters of
    scripts/research/totals.load_totals: both prices -250..+200 with |price| >= 100, point 25..80, overround
    1.00..1.12, point within 5 of the snapshot median point. Pinnacle and the exchanges are left out (the research
    feed, regions=us, had neither; same convention as grade_v2).
  * each book's implied E[T] = the grid inversion of the production key-number distribution (research
    VDist.implied_mu, identical to TotalDist.implied_mu within 0.03 points); consensus = median over books, sharp =
    median over lowvig/betonlineag/circasports/bookmaker, dispersion = SD across books.
  * move_mu / move_pts: vs the FIRST of our saved odds snapshots (history/odds_*.json[.gz]) within 7 days of
    kickoff (research: first snapshot <= 7 days). Live snapshots are hourly, the research's were sparser.
  * hours before kickoff from the game's kickoff_utc; candidates: allowed-book (my_books.json) quotes 10 min ..
    7 days before kickoff. Outside that range a game is not graded.
Live approximations: the us2 books (ESPN BET, Fanatics, Hard Rock ...) count in the consensus (the research feed had
only regions=us books); everything else matches. Fail-safe: missing lightgbm/model or any error -> grade None.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "grade_totals_model.json"
GRADING_VERSION = 2
ORDER = ["A+", "A", "B", "C"]
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
EXCLUDE = {"pinnacle", "kalshi", "prophetx", "polymarket", "novig", "betopenly"}
KEYT = (37.0, 41.0, 44.0, 47.0, 51.0)
MAX_HOURS, MIN_HOURS = 7 * 24, 10 / 60
NOTE = "Totals grade = predicted closing-line value; only A+ has shown an edge (2023-25: +1.5% CLV)"
_CACHE: dict = {}


# ------------------------------------------------------------------------------------------------ model
def load_model(path: Path = MODEL_PATH) -> dict | None:
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    m = None
    try:
        fz = json.loads(Path(path).read_text())
        md = fz["model"]
        if md.get("kind") != "lgbm":
            raise ValueError(f"unsupported model kind {md.get('kind')}")
        import lightgbm as lgb
        m = {"booster": lgb.Booster(model_str=md["model_string"]), "feature_order": list(md["feature_order"]),
             "thresholds": fz["thresholds_pred_clv"], "ev_clip": tuple(fz.get("ev_clip", (-0.15, 0.25))),
             "frozen_at": fz.get("frozen_at")}
        if m["booster"].num_feature() != len(m["feature_order"]):
            raise ValueError("feature count mismatch")
    except Exception as e:
        print(f"totals grade: unavailable ({e})")
        m = None
    _CACHE[key] = m
    return m


def letter(pred: float | None, th: dict) -> str | None:
    if pred is None or not math.isfinite(pred):
        return None
    return "A+" if pred >= th["A+"] else "A" if pred >= th["A"] else "B" if pred >= th["B"] else "C"


# ------------------------------------------------------------------------------------------------ market refs
def _imp(a: float) -> float:
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def _dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def _dist():
    from . import totals as T
    return T.default_dist()


def implied_mu(dist, point: float, nv_over: float) -> float:
    """Research VDist.implied_mu (grid version of TotalDist.implied_mu)."""
    fl = int(math.floor(point))
    po = 1 - dist.cdf[:, fl]
    pu = dist.cdf[:, int(math.ceil(point)) - 1] if point == int(point) else dist.cdf[:, fl]
    q = np.maximum.accumulate(po / np.maximum(po + pu, 1e-12))
    return float(np.interp(nv_over, q, dist.mean_at))


def research_rows(quotes) -> list[dict]:
    """[book_key, title, point, over, under] -> filtered rows (research load_totals filters), feed books only."""
    rows = []
    for q in quotes or []:
        try:
            k, title, pt, o, u = q[0], q[1], float(q[2]), float(q[3]), float(q[4])
        except (TypeError, ValueError, IndexError):
            continue
        if k in EXCLUDE or not all(map(math.isfinite, (pt, o, u))):
            continue
        if not (-250 <= o <= 200 and -250 <= u <= 200 and abs(o) >= 100 and abs(u) >= 100 and 25 <= pt <= 80):
            continue
        ovr = _imp(o) + _imp(u)
        if not (1.0 <= ovr <= 1.12):
            continue
        rows.append({"key": k, "book": title or k, "point": pt, "over": o, "under": u, "nv_over": _imp(o) / ovr})
    if rows:
        med = statistics.median(r["point"] for r in rows)
        rows = [r for r in rows if abs(r["point"] - med) <= 5]
    return rows


def refs(rows: list[dict], dist=None) -> dict | None:
    """Snapshot aggregates of the research table: mu_cons, pt_cons, disp, mu_sharp, pt_sharp (None if missing)."""
    if not rows:
        return None
    dist = dist or _dist()
    for r in rows:
        r["mu"] = implied_mu(dist, r["point"], r["nv_over"])
    mus = [r["mu"] for r in rows]
    sh = [r for r in rows if r["key"] in SHARP]
    return {"mu_cons": float(statistics.median(mus)), "pt_cons": float(statistics.median(r["point"] for r in rows)),
            "disp": float(statistics.stdev(mus)) if len(mus) >= 2 else None, "n_books": len(rows),
            "mu_sharp": float(statistics.median(r["mu"] for r in sh)) if sh else None,
            "pt_sharp": float(statistics.median(r["point"] for r in sh)) if sh else None}


# ------------------------------------------------------------------------------------------------ first seen
def _snapshot_files(history_dir: Path) -> list[tuple[datetime, Path]]:
    out = []
    for f in list(history_dir.glob("odds_*.json")) + list(history_dir.glob("odds_*.json.gz")):
        try:
            ts = datetime.strptime(f.name.split(".")[0][5:], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append((ts, f))
    return sorted(out)


def _load(f: Path, cache: dict):
    if f not in cache:
        try:
            cache[f] = json.loads(gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text())
        except Exception:
            cache[f] = []
    return cache[f]


def _find_event(events, home: str, away: str, kick: datetime):
    from . import odds as O
    for ev in events or []:
        if O.TEAM_ABBR.get(ev.get("home_team")) != home or O.TEAM_ABBR.get(ev.get("away_team")) != away:
            continue
        try:
            c = datetime.fromisoformat(str(ev.get("commence_time")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if abs((c - kick).total_seconds()) <= 36 * 3600:
            return ev
    return None


def event_quotes(ev: dict) -> list:
    """Raw [book_key, title, point, over, under] from one Odds API event (same parse as totals.summarize_event)."""
    out = []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "totals":
                continue
            t = {o.get("name"): o for o in m.get("outcomes", [])}
            ov, un = t.get("Over"), t.get("Under")
            if ov and un and ov.get("point") is not None:
                out.append([bk.get("key"), bk.get("title", bk.get("key")), ov["point"], ov.get("price"), un.get("price")])
    return out


def first_seen(history_dir: Path, games: list[dict], dist=None) -> dict[str, dict]:
    """game_id -> refs at our first saved snapshot within 7 days of kickoff (only the files that can matter
    are opened)."""
    files = _snapshot_files(history_dir)
    cache: dict = {}
    out = {}
    for g in games:
        try:
            kick = datetime.fromisoformat(g["kickoff_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        for ts, f in files:
            if ts < kick - timedelta(hours=MAX_HOURS) or ts >= kick:
                continue
            ev = _find_event(_load(f, cache), g["home_team"], g["away_team"], kick)
            r = refs(research_rows(event_quotes(ev)), dist) if ev else None
            if r:
                out[g["game_id"]] = dict(r, ts=ts.isoformat(timespec="minutes"))
                break
    return out


# ------------------------------------------------------------------------------------------------ features
def key_feats(point: float, side: str, cons: float) -> tuple[float, float, float]:
    over = side == "over"
    kr = onk = cross = 0.0
    adv = (cons - point) if over else (point - cons)
    lo, hi = min(point, cons), max(point, cons)
    for k in KEYT:
        kr += (float(point == k - 0.5) - float(point == k + 0.5)) if over else (float(point == k + 0.5) - float(point == k - 0.5))
        onk += float(point == k)
        cross += float(lo <= k <= hi and abs(point - cons) > 1e-9)
    return kr, onk, float(np.sign(adv)) * cross


def offer_features(side: str, point: float, price: float, R: dict, hours: float, first: dict | None,
                   best_ev_cons: float | None, dist=None, clip=(-0.15, 0.25)) -> tuple[dict, dict]:
    """(features, raw) for one allowed-book over/under quote; identical to research tfeaturize ('small' set)."""
    dist = dist or _dist()
    c = (lambda x: min(max(x, clip[0]), clip[1]))
    sg = 1.0 if side == "over" else -1.0
    ev_s = float(dist.ev(R["mu_sharp"], point, price, side)) if R.get("mu_sharp") is not None else None
    ev_c = float(dist.ev(R["mu_cons"], point, price, side))
    pt_sh = R["pt_sharp"] if R.get("pt_sharp") is not None else R["pt_cons"]
    kr, onk, cr = key_feats(point, side, R["pt_cons"])
    mf = (first or {}).get("mu_cons")
    pf = (first or {}).get("pt_cons")
    F = {
        "ev_sharp": c(ev_s if ev_s is not None else ev_c),
        "ev_cons": c(ev_c),
        "sharp_missing": float(R.get("mu_sharp") is None),
        "pt_adv_cons": min(max(sg * (R["pt_cons"] - point), -2.5), 2.5),
        "pt_adv_sharp": min(max(sg * (pt_sh - point), -2.5), 2.5),
        "key_right": kr, "on_key": onk, "key_cross": cr,
        "p_imp": 1 / _dec(price),
        "move_mu": min(max(sg * (R["mu_cons"] - mf), -6), 6) if mf is not None else 0.0,
        "move_pts": min(max(sg * (R["pt_cons"] - pf), -6), 6) if pf is not None else 0.0,
        "disp": min(max(R["disp"], 0.0), 3.0) if R.get("disp") is not None else 0.5,
        "log_hours": math.log1p(max(hours, 0.0)),
        "is_over": float(side == "over"),
        "tot_level": R["mu_cons"] - 44.0,
        "best_gap": min(max((best_ev_cons - ev_c) if best_ev_cons is not None else 0.0, 0.0), 0.1),
    }
    return F, {"ev_sharp_raw": ev_s, "ev_cons_raw": ev_c}


def predict(model: dict, feats: list[dict]) -> list[float]:
    x = np.array([[float(f[k]) for k in model["feature_order"]] for f in feats])
    return [float(v) for v in model["booster"].predict(x, num_threads=1)]


def grade_game(game: dict, model: dict, now: datetime, first: dict | None, allowed: set | None, dist=None) -> dict | None:
    """Grade every allowed-book over/under quote; game['totals_grade'] = best predicted CLV offer,
    game['totals_grade_sides'] = best per side, game['totals_grade_offers'] = [side, book_key, point, price, pred]."""
    game["totals_grade"], game["totals_grade_sides"], game["totals_grade_offers"] = None, {}, []
    game.pop("totals_grade_check", None)
    tot = ((game.get("context") or {}).get("live_odds") or {}).get("totals") or {}
    if not tot.get("all_quotes"):
        game["totals_grade_check"] = ["no live totals"]
        return None
    try:
        hours = (datetime.fromisoformat(game["kickoff_utc"]) - now).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return None
    if not (MIN_HOURS <= hours <= MAX_HOURS):
        game["totals_grade_check"] = ["graded only within 7 days of kickoff"]
        return None
    dist = dist or _dist()
    rows = research_rows(tot["all_quotes"])
    R = refs(rows, dist)
    if R is None:
        game["totals_grade_check"] = ["no usable totals quotes"]
        return None
    offers = [(side, r) for r in rows if allowed is None or r["key"] in allowed for side in ("over", "under")]
    if not offers:
        game["totals_grade_check"] = ["no allowed-book totals"]
        return None
    ev_c = {(side, id(r)): dist.ev(R["mu_cons"], r["point"], r[side], side) for side, r in offers}
    best_c = {s: max(v for (ss, _), v in ev_c.items() if ss == s) for s in ("over", "under")
              if any(ss == s for ss, _ in ev_c)}
    feats = [offer_features(side, r["point"], r[side], R, hours, first, best_c.get(side), dist,
                            model.get("ev_clip", (-0.15, 0.25)))[0] for side, r in offers]
    preds = predict(model, feats)
    best = None
    for (side, r), p in zip(offers, preds):
        o = {"side": side, "point": r["point"], "price": int(r[side]) if float(r[side]).is_integer() else r[side],
             "book": r["book"], "book_key": r["key"], "predicted_clv": round(p, 4),
             "grade": letter(p, model["thresholds"]), "grading_version": GRADING_VERSION}
        game["totals_grade_offers"].append([side, r["key"], r["point"], o["price"], o["predicted_clv"]])
        cur = game["totals_grade_sides"].get(side)
        if cur is None or o["predicted_clv"] > cur["predicted_clv"]:
            game["totals_grade_sides"][side] = o
        if best is None or o["predicted_clv"] > best["predicted_clv"]:
            best = o
    best = dict(best, hours_before=round(hours, 1), consensus_total=round(R["mu_cons"], 2),
                sharp_total=None if R["mu_sharp"] is None else round(R["mu_sharp"], 2))
    game["totals_grade"] = best
    return best


def attach(pred: dict, history_dir: Path, now: datetime | None = None, model: dict | None | bool = None) -> dict:
    """Grade every upcoming game in place. Never raises; on any failure grades are None."""
    now = now or datetime.now(timezone.utc)
    ups = pred.get("upcoming", [])
    for g in ups:
        g["totals_grade"], g["totals_grade_sides"], g["totals_grade_offers"] = None, {}, []
    try:
        m = load_model() if model is None else (model or None)
    except Exception:
        m = None
    if not m:
        return {"version": GRADING_VERSION, "available": False, "note": NOTE}
    from . import odds as O
    allowed = O.load_allowed_books()
    try:
        first = first_seen(history_dir, ups)
    except Exception as e:
        print(f"totals grade: first-seen failed ({e})")
        first = {}
    n = errs = 0
    for g in ups:
        try:
            if grade_game(g, m, now, first.get(g.get("game_id")), allowed):
                n += 1
        except Exception as e:
            errs += 1
            g["totals_grade"], g["totals_grade_sides"], g["totals_grade_offers"] = None, {}, []
            print(f"totals grade: {g.get('game_id')} failed ({e})")
    return {"version": GRADING_VERSION, "available": True, "graded_games": n, "errors": errs,
            "thresholds": m["thresholds"], "frozen_at": m.get("frozen_at"), "note": NOTE}


def bet_fields(game: dict, side: str, book_key: str | None, point, price) -> dict:
    """totals_grade / predicted_clv of exactly this offer (label only; never used to qualify)."""
    for s, k, pt, pr, p in game.get("totals_grade_offers") or []:
        if s == side and k == book_key and pt == point and pr == price:
            return {"totals_grade": letter(p, (load_model() or {}).get("thresholds") or
                                           {"A+": 0.0, "A": -0.01, "B": -0.02}), "predicted_clv": p}
    return {"totals_grade": None, "predicted_clv": None}


# ------------------------------------------------------------------------------------------------ record
def by_grade(ledger: list[dict], track: str | None = None) -> list[dict]:
    rows = []
    for gl in ORDER:
        bs = [b for b in ledger if b.get("status") == "graded" and b.get("totals_grade") == gl]
        if not bs:
            continue
        staked = sum(b["units"] for b in bs if b.get("result") != "push")
        profit = sum(b.get("profit_units", 0) for b in bs)
        clv = [b["clv"] for b in bs if b.get("clv") is not None]
        pc = [b["predicted_clv"] for b in bs if b.get("predicted_clv") is not None]
        rows.append({"grade": gl, **({"track": track} if track else {}), "bets": len(bs),
                     "wins": sum(b.get("result") == "win" for b in bs),
                     "losses": sum(b.get("result") == "loss" for b in bs), "profit_units": round(profit, 2),
                     "roi": round(profit / staked, 4) if staked else 0.0,
                     "avg_clv": round(sum(clv) / len(clv), 4) if clv else None,
                     "avg_predicted_clv": round(sum(pc) / len(pc), 4) if pc else None})
    return rows


TOTALS_LEDGERS = {"Totals wind": "paper_bets_totals_wind.json", "Totals early under": "paper_bets_totals_early_under.json"}


def record(history_dir: Path) -> dict:
    """Paper record by totals grade across the totals tracks (all together, then per track)."""
    allb, per = [], []
    for name, f in TOTALS_LEDGERS.items():
        p = history_dir / f
        try:
            led = json.loads(p.read_text()) if p.exists() else []
        except Exception:
            led = []
        led = [b for b in led if b.get("totals_grade")]
        allb += led
        per += by_grade(led, name)
    return {"all": by_grade(allb), "by_track": per, "open_bets_with_grade": sum(b.get("status") == "open" for b in allb)}
