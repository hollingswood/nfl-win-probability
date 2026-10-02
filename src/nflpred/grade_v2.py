"""Grade v2 (GRADING_VERSION 2): predicted closing-line value of a live MONEYLINE offer.

Research: scripts/research/grade_v2.py -> output/research/grade_v2.md. The frozen LightGBM model (fit on 2020-22,
one-shot holdout 2023-25) is copied verbatim to `grade_v2_model.json` at the repo root. Letters on predicted CLV:
A+ >= 2.5%, A >= 1.5%, B >= 0.5%, else C. Holdout: realized CLV rose with the letter and only A+ showed an edge
(+2.3% ± 1.0). So the grade is a LABEL: it never vetoes, qualifies or sizes a v1-v4 bet. Spread offers are not graded.

Each upcoming game is graded at the best allowed-book price for each side (what the tracks would bet). Features are
computed as in the research (`featurize`), from predictions.json game data (context.live_odds). Live approximations:
  * consensus / sharp / dispersion / tie prob use `live_odds.by_book` (per-book raw prices, same filters as the
    research: |ML| 100..2500, overround -1%..15%; spread prices -145..+125, half points, within 2.5 of the median).
    Pinnacle and exchanges are left out of the consensus (the research feed had neither). Older snapshots without
    `by_book` fall back to consensus_home_prob / sharp_home_prob (which include Pinnacle/exchanges), tie 0.3%,
    dispersion 0.01 and -consensus_home_margin.
  * ev_sp_sharp: ml_v4.spread_implied_home on the LowVig/BetOnline spreads that pass the research filters, total =
    live median total point; fallback the all-book spread-implied median, then ev_cons.
  * move_p / move_pts: vs the first predictions_*.json in history/ that had live odds for the game (weekly full runs,
    not the research's daily 14:10 snapshots), both sides measured with the production consensus fields so the
    Pinnacle/exchange difference cancels. Not seen before -> 0 (as the research's first snapshot).
  * model_elig / ev_model: eligible = official injury report out and both QBs confirmed (the v2 rule's live test);
    p_model = published home_win_prob (QB-availability blend) instead of the research's walk-forward p_model.
  * best_gap = 0 (we grade the best allowed price per side); is_last = 1 within 90 min of kickoff (research: the last
    snapshot, ~75 min pre-kick); v4 window via ml_v4.in_window (+-50 min instead of exact snapshot times).
  * is_last, n_signals, sharp_missing, longshot, big_fav and the book dummies have zero split gain in the frozen model.
Fail-safe: if lightgbm or the model file is missing, or anything errors, the grade is None and nothing else changes.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "grade_v2_model.json"
GRADING_VERSION = 2
ORDER = ["A+", "A", "B", "C"]
EXCLUDE_FROM_CONSENSUS = {"pinnacle", "kalshi", "prophetx", "polymarket", "novig", "betopenly"}
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
TITLE_TO_KEY = {"draftkings": "draftkings", "fanduel": "fanduel", "betmgm": "betmgm", "caesars": "williamhill_us",
                "williamhill_us": "williamhill_us", "betrivers": "betrivers", "thescore bet": "espnbet",
                "espn bet": "espnbet", "espnbet": "espnbet", "fanatics": "fanatics", "hard rock bet": "hardrockbet",
                "hardrockbet": "hardrockbet"}
TIE_FALLBACK = 0.003
DISP_FALLBACK = 0.01
NOTE = "Grade = predicted closing-line value; only A+ has shown an edge (2023-25: +2.3% CLV)"

_CACHE: dict = {}


# ------------------------------------------------------------------------------------------------ model
def load_model(path: Path = MODEL_PATH) -> dict | None:
    """{'booster', 'feature_order', 'thresholds', 'ev_clip'} or None (lightgbm missing / bad file)."""
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    m = None
    try:
        fz = json.loads(Path(path).read_text())
        import lightgbm as lgb
        md = fz["model"]
        if md.get("kind") != "lgbm":
            raise ValueError(f"unsupported model kind {md.get('kind')}")
        m = {"booster": lgb.Booster(model_str=md["model_string"]),
             "feature_order": list(md.get("feature_order") or fz["feature_order"]),
             "thresholds": fz["thresholds_pred_clv"], "ev_clip": tuple(fz.get("ev_clip", (-0.15, 0.25))),
             "frozen_at": fz.get("frozen_at")}
        if m["booster"].num_feature() != len(m["feature_order"]):
            raise ValueError("feature count mismatch")
    except Exception as e:
        print(f"grade v2: unavailable ({e})")
        m = None
    _CACHE[key] = m
    return m


def letter(pred: float | None, th: dict) -> str | None:
    if pred is None or not math.isfinite(pred):
        return None
    return "A+" if pred >= th["A+"] else "A" if pred >= th["A"] else "B" if pred >= th["B"] else "C"


# ------------------------------------------------------------------------------------------------ helpers
def decimal(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def _imp(a: float) -> float:
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def _nv_ml(h, a) -> float | None:
    """No-vig home prob with the research's ml_ok filter."""
    try:
        h, a = float(h), float(a)
    except (TypeError, ValueError):
        return None
    if not (100 <= abs(h) <= 2500 and 100 <= abs(a) <= 2500):
        return None
    ih, ia = _imp(h), _imp(a)
    if not (-0.01 <= ih + ia - 1 <= 0.15):
        return None
    return ih / (ih + ia)


def _median(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def _book_key(b: dict | None) -> str | None:
    if not b:
        return None
    if b.get("key"):
        return b["key"]
    return TITLE_TO_KEY.get(str(b.get("book", "")).strip().lower())


def _margin_model():
    from . import ml_v4
    return ml_v4._model()


def _spread_rows(by_book: dict, books=None) -> list:
    """[(book, home_point, home_price, away_price)] passing the research sp_ok filter (prices, half points,
    overround, within 2.5 of the median point of the valid rows)."""
    rows = []
    for k, d in by_book.items():
        if k in EXCLUDE_FROM_CONSENSUS or (books is not None and k not in books):
            continue
        sp = (d or {}).get("sp")
        if not sp:
            continue
        try:
            hp, hpr, apr = float(sp[0]), float(sp[1]), float(sp[2])
        except (TypeError, ValueError):
            continue
        if not (-145 <= hpr <= 125 and -145 <= apr <= 125) or (hp * 2) % 1 != 0:
            continue
        if not (-0.01 <= _imp(hpr) + _imp(apr) - 1 <= 0.12):
            continue
        rows.append((k, hp, hpr, apr))
    if rows:
        med = statistics.median(r[1] for r in rows)
        rows = [r for r in rows if abs(r[1] - med) <= 2.5]
    return rows


def _spread_probs(rows: list, tot: float) -> tuple[list, list]:
    """Per-row (P(home win | no tie), P(tie)) from the total-aware margin model."""
    if not rows:
        return [], []
    import numpy as np
    M = _margin_model()
    hp = np.array([r[1] for r in rows])
    q = np.array([_imp(r[2]) / (_imp(r[2]) + _imp(r[3])) for r in rows])
    mu = M.implied_mu(hp, q, np.full(len(rows), tot))
    pw, pt = M.win_tie(mu, np.full(len(rows), tot))
    return [float(w / (1 - t)) for w, t in zip(pw, pt)], [float(t) for t in pt]


def market_refs(lo: dict) -> dict:
    """Per-game references (home-side probabilities) the features need, from one live_odds snapshot."""
    from . import ml_v4
    bb = lo.get("by_book") or {}
    tot = (lo.get("totals") or {}).get("median_point") or (lo.get("totals") or {}).get("consensus_total") or 44.0
    ref = {"approx": []}
    ml = {k: _nv_ml(*(d.get("ml") or [None, None])) for k, d in bb.items() if d}
    cons = [p for k, p in ml.items() if p is not None and k not in EXCLUDE_FROM_CONSENSUS]
    sharp = [p for k, p in ml.items() if p is not None and k in SHARP]
    ref["p_cons"] = _median(cons) if cons else lo.get("consensus_home_prob")
    ref["p_sharp"] = _median(sharp) if bb else lo.get("sharp_home_prob")
    if not cons:
        ref["approx"].append("consensus from summary (incl. Pinnacle/exchanges)")
    ref["disp"] = statistics.stdev(cons) if len(cons) >= 2 else None
    ref["p_v3"] = lo.get("pin_sharp_home_prob")
    pts = [float(d["sp"][0]) for k, d in bb.items()
           if d and d.get("sp") and d["sp"][0] is not None and k not in EXCLUDE_FROM_CONSENSUS]
    if pts:
        ref["pt_cons"] = statistics.median(pts)
    elif lo.get("consensus_home_margin") is not None:
        ref["pt_cons"] = -float(lo["consensus_home_margin"])
    else:
        ref["pt_cons"] = None
    rows = _spread_rows(bb)
    p_rows, t_rows = _spread_probs(rows, tot)
    ref["p_tie"] = _median(t_rows)
    ref["p_sp_cons"] = _median(p_rows)
    # sharp spread-implied P(win) through ml_v4.spread_implied_home, on the sharp rows that pass the filters
    r4 = ml_v4.load_rules()
    if bb:
        sharp_sp = [[hp, hpr, apr] for (k, hp, hpr, apr) in rows if k in set(r4["qualify"]["sharp_spread_books"])]
    else:
        sharp_sp = lo.get("sharp_spreads") or []
    ref["p_sp_sharp"] = ml_v4.spread_implied_home(dict(lo, sharp_spreads=sharp_sp), r4) if sharp_sp else None
    if ref["p_tie"] is None:
        ref["approx"].append("tie prob fallback 0.3%")
    return ref


def first_seen(history_dir: Path) -> tuple[dict, dict]:
    """(game_id -> first consensus home prob, game_id -> first consensus home margin) from predictions history."""
    from . import bets, spread_bets
    try:
        return bets.first_seen_market(history_dir), spread_bets.first_seen_margin(history_dir)
    except Exception:
        return {}, {}


# ------------------------------------------------------------------------------------------------ features
def offer_features(game: dict, side: str, price: float, book: str | None, ref: dict, now: datetime,
                   first_p: float | None = None, first_m: float | None = None, clip=(-0.15, 0.25)) -> dict:
    """The research feature vector (featurize) for one moneyline offer, plus the candidate-filter verdict."""
    from . import ml_v4
    lo = (game.get("context") or {}).get("live_odds") or {}
    h = side == "home"
    s = (lambda p: p if h else 1 - p)
    c = (lambda x: min(max(x, clip[0]), clip[1]))
    dec = decimal(price)
    pt = ref.get("p_tie") if ref.get("p_tie") is not None else TIE_FALLBACK

    def ev(p):
        return None if p is None else (s(p) * dec - 1) * (1 - pt)
    ev_cons = ev(ref.get("p_cons"))
    ev_sharp = ev(ref.get("p_sharp"))
    ev_sp_sharp = ev(ref.get("p_sp_sharp"))
    ev_sp_cons = ev(ref.get("p_sp_cons"))
    ev_v3 = ev(ref.get("p_v3"))
    qbs = game.get("qb_status") or {}
    eligible = bool(game.get("injury_report")) and all((qbs.get(x) or {}).get("play_prob", 1) >= 1
                                                       for x in ("home", "away"))
    pm = game.get("home_win_prob")
    ev_model = dec * s(pm) - 1 if eligible and pm is not None else None
    kick = game.get("kickoff_utc")
    hours = max(0.0, (datetime.fromisoformat(kick) - now).total_seconds() / 3600) if kick else 72.0
    sg = 1 if h else -1
    p_now, m_now = lo.get("consensus_home_prob"), lo.get("consensus_home_margin")
    move_p = sg * (p_now - first_p) if p_now is not None and first_p is not None else 0.0
    move_pts = sg * (m_now - first_m) if m_now is not None and first_m is not None else 0.0
    pt_side = ref.get("pt_cons")
    pt_side = 0.0 if pt_side is None else (pt_side if h else -pt_side)
    pts = round(pt_side * 2) / 2
    from . import grading as G1
    try:
        win_ok = ml_v4.in_window(now, ml_v4.load_rules())
    except Exception:
        win_ok = False
    v2 = ev_sharp is not None and ev_sharp >= 0.02 and ev_model is not None and ev_model >= 0
    v3 = ev_v3 is not None and ev_v3 >= 0.02 and ev_model is not None and ev_model >= 0
    v4 = win_ok and ev_sp_sharp is not None and ev_sp_sharp >= 0.02 and ev_sharp is not None and ev_sharp >= 0 \
        and price <= 400
    F = {
        "ev_ml_sharp": c(ev_sharp if ev_sharp is not None else (ev_cons or 0.0)),
        "ev_sp_sharp": c(next((x for x in (ev_sp_sharp, ev_sp_cons, ev_cons) if x is not None), 0.0)),
        "ev_cons": c(ev_cons or 0.0),
        "ev_model": c(ev_model or 0.0),
        "model_elig": float(ev_model is not None),
        "sharp_missing": float(ev_sharp is None),
        "n_signals": float(v2) + float(v3) + float(v4),
        "move_p": min(max(move_p, -0.2), 0.2),
        "move_pts": min(max(move_pts, -7.0), 7.0),
        "log_hours": math.log1p(hours),
        "is_last": float(10 / 60 <= hours <= 1.5),
        "p_imp": 1 / dec,
        "is_dog": float(price > 0),
        "longshot": float(price >= 250),
        "big_fav": float(price <= -200),
        "disp": min(max(ref["disp"], 0.0), 0.1) if ref.get("disp") is not None else DISP_FALLBACK,
        "best_gap": 0.0,
        "key_pos": float(G1.key_number_side(float(pts))),
        "on3": float(2.5 <= abs(pts) <= 3.5),
    }
    for b in ("fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet"):
        F[f"bk_{b}"] = float(book == b)
    gap = abs(1 / dec - s(ref["p_cons"])) if ref.get("p_cons") is not None else None
    anypos = any(x is not None and x >= 0 for x in (ev_sharp, ev_v3, ev_sp_sharp, ev_cons, ev_model))
    cand = -1000 <= price <= 1000 and gap is not None and gap <= 0.12 and anypos
    return {"features": F, "candidate": cand, "hours": hours}


def predict(model: dict, feats: dict) -> float:
    import numpy as np
    x = np.array([[float(feats[f]) for f in model["feature_order"]]])
    return float(model["booster"].predict(x, num_threads=1)[0])


def grade_offer(game: dict, side: str, price: float, book: str | None, model: dict, ref: dict, now: datetime,
                first_p=None, first_m=None) -> dict | None:
    o = offer_features(game, side, price, book, ref, now, first_p, first_m, model.get("ev_clip", (-0.15, 0.25)))
    if not o["candidate"]:
        return None
    pc = predict(model, o["features"])
    return {"predicted_clv": round(pc, 4), "grade": letter(pc, model["thresholds"])}


def grade_game(game: dict, model: dict, now: datetime, first_p=None, first_m=None) -> dict | None:
    """Grades both sides at the best allowed price; sets game['grade_v2'] (best predicted CLV) and
    game['grade_v2_sides']. Returns the best or None."""
    lo = (game.get("context") or {}).get("live_odds") or {}
    game["grade_v2"], game["grade_v2_sides"] = None, {}
    game.pop("grade_v2_check", None)
    if not lo.get("best_home_ml") or not lo.get("best_away_ml"):
        return None
    ref = market_refs(lo)
    best = None
    for side in ("home", "away"):
        bk = lo[f"best_{side}_ml"]
        g = grade_offer(game, side, bk["price"], _book_key(bk), model, ref, now, first_p, first_m)
        if g is None:
            continue
        g = {"side": side, "team": game[f"{side}_team"], "price": bk["price"], "book": bk.get("book"), **g,
             "grading_version": GRADING_VERSION}
        game["grade_v2_sides"][side] = g
        if best is None or g["predicted_clv"] > best["predicted_clv"]:
            best = g
    if best and ref["approx"]:
        best["approx"] = ref["approx"]
    if best is None:  # outside the research candidate set: no price is >= 0 EV vs any fair reference
        game["grade_v2_check"] = ["no moneyline price is +EV vs any fair reference (ungraded; below C)"]
    game["grade_v2"] = best
    return best


def attach(pred: dict, history_dir: Path, now: datetime | None = None, model: dict | None | bool = None) -> dict:
    """Grade every upcoming game in place. Never raises; on any failure grades are None."""
    now = now or datetime.now(timezone.utc)
    ups = pred.get("upcoming", [])
    try:
        m = load_model() if model is None else (model or None)
    except Exception:
        m = None
    for g in ups:
        g["grade_v2"], g["grade_v2_sides"] = None, {}
    if not m:
        return {"version": GRADING_VERSION, "available": False, "note": NOTE}
    fp, fm = first_seen(history_dir)
    n, errs = 0, 0
    for g in ups:
        try:
            if grade_game(g, m, now, fp.get(g["game_id"]), fm.get(g["game_id"])):
                n += 1
        except Exception as e:
            errs += 1
            g["grade_v2"], g["grade_v2_sides"] = None, {}
            print(f"grade v2: {g.get('game_id')} failed ({e})")
    return {"version": GRADING_VERSION, "available": True, "graded_games": n, "errors": errs,
            "thresholds": m["thresholds"], "frozen_at": m.get("frozen_at"), "note": NOTE}


def bet_fields(game: dict, side: str, price) -> dict:
    """grade_v2 / predicted_clv for a new paper bet on `side` at `price` (label only; never used to qualify)."""
    g = (game.get("grade_v2_sides") or {}).get(side)
    if not g or g.get("price") != price:
        return {"grade_v2": None, "predicted_clv": None}
    return {"grade_v2": g["grade"], "predicted_clv": g["predicted_clv"]}


# ------------------------------------------------------------------------------------------------ record
def by_grade(ledger: list[dict], track: str | None = None) -> list[dict]:
    rows = []
    for gl in ORDER:
        bs = [b for b in ledger if b.get("status") == "graded" and b.get("grade_v2") == gl]
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


ML_LEDGERS = {"Moneyline v1": "paper_bets.json", "Moneyline v2": "paper_bets_ml_v2.json",
              "Moneyline v3": "paper_bets_ml_v3.json", "Moneyline v4": "paper_bets_ml_v4.json"}


def record(history_dir: Path) -> dict:
    """Paper record by grade v2 across the moneyline tracks (all tracks together, then per track)."""
    allb, per = [], []
    for name, f in ML_LEDGERS.items():
        p = history_dir / f
        try:
            led = json.loads(p.read_text()) if p.exists() else []
        except Exception:
            led = []
        led = [b for b in led if b.get("grade_v2")]
        allb += led
        per += by_grade(led, name)
    open_n = sum(b.get("status") == "open" for b in allb)
    return {"all": by_grade(allb), "by_track": per, "open_bets_with_grade": open_n}
