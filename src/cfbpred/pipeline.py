"""College football page pipeline: refresh data, rate teams, price this week's games, write
output/cfb_predictions.json (read by scripts/build_dashboard.py -> site/cfb/index.html).

Called from the hourly odds watch:
  * CFBD refresh (current season games, lines, advanced stats: 6 calls) at most every 12 h;
  * ratings + model refit whenever the data was refreshed (or no cached features exist);
  * live odds from the newest history/cfb snapshot every run.
The model is display-only: none of the college rules has passed its holdout test
(output/research/cfb/holdout.md), so nothing here is a bet.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import fetch as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
CACHE = OUT / "cfb_features.parquet"
STATE = ROOT / "history" / "cfb_state.json"
SIGMA = 16.0          # sd of college margins around the line (for win probabilities)
FIRST = 2019          # rating chain start (priors converge within a season)


def season_now(now: datetime) -> int:
    return now.year if now.month >= 7 else now.year - 1


# ------------------------------------------------------------------------------------- features
def build_features(season: int) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "scripts" / "research" / "cfb"))
    import alt_models as AM
    import fast
    AM.SEASONS = range(FIRST, season + 1)
    g, tal, ret = fast.prep()
    g = g[g.season.between(FIRST, season)].copy()
    Fe = AM.adjusted(g, AM.adv_long())
    base = fast.run(g, tal, ret, last_seasons=(FIRST, season)).rename(columns={"m_pred": "margin_ridge"})
    G = g.merge(Fe, on="game_id", how="left").merge(base, on="game_id", how="left")
    tz = tal.copy(); tz["tz"] = tz.groupby("season").talent.transform(lambda x: (x - x.mean()) / x.std())
    for side in ("home", "away"):
        G = G.merge(tz[["season", "team", "tz"]].rename(columns={"team": side, "tz": f"tz_{side[0]}"}), on=["season", side], how="left")
        G = G.merge(ret.rename(columns={"team": side, "percentPPA": f"ret_{side[0]}"}), on=["season", side], how="left")
    G["nh"] = (~G.neutral).astype(float)
    for s in AM.STATS:
        G[f"{s}_d"] = G[f"{s}_h"] - G[f"{s}_a"]; G[f"{s}_s"] = G[f"{s}_h"] + G[f"{s}_a"]
    G["tz_d"] = G.tz_h.fillna(0) - G.tz_a.fillna(0); G["ret_d"] = G.ret_h.fillna(0.5) - G.ret_a.fillna(0.5)
    return G


MCOLS = ["margin_ridge", "nh", "pts_d", "ppa_d", "sr_d", "expl_d", "rush_ppa_d", "pass_ppa_d", "line_yds_d", "stuff_d",
         "power_d", "second_lvl_d", "open_field_d", "sd_ppa_d", "pd_ppa_d", "tz_d", "ret_d", "week"]
TCOLS = ["pts_s", "plays_s", "ppa_s", "sr_s", "expl_s"]


def fit_models(G: pd.DataFrame, season: int) -> dict:
    fb = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.margin.notna() & G.season.between(FIRST + 1, season)]
    fm = fb.dropna(subset=MCOLS)
    X = np.column_stack([np.ones(len(fm))] + [fm[c] for c in MCOLS]); A = X.T @ X + np.eye(X.shape[1]); A[0, 0] -= 1
    bm = np.linalg.solve(A, X.T @ fm.margin.to_numpy())
    ft = fb.dropna(subset=TCOLS)
    bt = np.linalg.lstsq(np.column_stack([np.ones(len(ft))] + [ft[c] for c in TCOLS]), ft.total, rcond=None)[0]
    return {"margin": dict(zip(["const"] + MCOLS, bm.round(5).tolist())), "total": dict(zip(["const"] + TCOLS, bt.round(5).tolist())),
            "n_margin": len(fm), "n_total": len(ft)}


def backtest(G: pd.DataFrame, season: int, first: int = 2022) -> list[dict]:
    """Each past season scored with a model fit only on earlier seasons, vs the CFBD consensus closing line."""
    L = D.lines(range(first, season))
    out = []
    for s in range(first, season):
        c = fit_models(G, s - 1)
        g = G[(G.season == s) & G.margin.notna() & ((G.home_div == "fbs") | (G.away_div == "fbs"))].copy()
        g["mm"], g["mt"] = apply(g, c["margin"]), apply(g, c["total"])
        g = g.merge(L, on="game_id", how="inner").dropna(subset=["spread_close"])
        out.append(score_games(g.assign(vm=-g.spread_close, vt=g.total_close), s))
    return out


def score_games(g: pd.DataFrame, label) -> dict:
    """g: margin, total, mm (model home margin), mt (model total), vm (Vegas home margin), vt (Vegas total)."""
    hw = g.margin > 0
    v = g[g.vm != 0]
    ats_side, ats_res = (g.mm - g.vm).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0), (g.margin - g.vm).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    a = (ats_side * ats_res)[(ats_side != 0) & (ats_res != 0)]
    t = g.dropna(subset=["vt", "mt"])
    ou_side, ou_res = (t.mt - t.vt).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0), (t.total - t.vt).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    o = (ou_side * ou_res)[(ou_side != 0) & (ou_res != 0)]
    ph = 0.5 * (1 + (g.mm / (SIGMA * math.sqrt(2))).apply(math.erf))
    pv = 0.5 * (1 + (g.vm / (SIGMA * math.sqrt(2))).apply(math.erf))
    ll = lambda p: float(-(hw * p.clip(1e-4, 1 - 1e-4).apply(math.log) + (~hw) * (1 - p.clip(1e-4, 1 - 1e-4)).apply(math.log)).mean())
    return {"season": label, "games": int(len(g)),
            "model_right": int(((g.mm > 0) == hw).sum()), "vegas_right": int(((v.vm > 0) == (v.margin > 0)).sum()), "vegas_games": int(len(v)),
            "model_ats_w": int((a > 0).sum()), "model_ats_l": int((a < 0).sum()),
            "model_ou_w": int((o > 0).sum()), "model_ou_l": int((o < 0).sum()),
            "model_mae": round(float((g.mm - g.margin).abs().mean()), 2), "vegas_mae": round(float((g.vm - g.margin).abs().mean()), 2),
            "model_total_mae": round(float((t.mt - t.total).abs().mean()), 2) if len(t) else None,
            "vegas_total_mae": round(float((t.vt - t.total).abs().mean()), 2) if len(t) else None,
            "model_log_loss": round(ll(ph), 4), "vegas_log_loss": round(ll(pv), 4)}


def apply(df: pd.DataFrame, coef: dict) -> pd.Series:
    out = pd.Series(coef["const"], index=df.index, dtype=float)
    for k, v in coef.items():
        if k != "const":
            out = out + v * df[k].astype(float).fillna(0.0)
    return out


# ------------------------------------------------------------------------------------- odds
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


ALIAS = {"southernmississippi": "southernmiss", "louisianaragincajuns": "louisiana", "umass": "massachusetts",
         "louisianamonroe": "ulmonroe", "appalachianstate": "appstate", "miamiohio": "miamioh"}


def team_matcher(teams: list[str]):
    keys = sorted(((_norm(t), t) for t in teams), key=lambda x: -len(x[0]))

    def match(odds_name: str) -> str | None:
        n = _norm(odds_name)
        for a, b in ALIAS.items():
            if n.startswith(a):
                n = b + n[len(a):]
        for k, t in keys:
            if n.startswith(k):
                return t
        return None
    return match


def implied(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def latest_snapshot() -> tuple[str | None, list]:
    files = sorted((ROOT / "history" / "cfb").glob("odds_*.json.gz"))
    if not files:
        return None, []
    with gzip.open(files[-1], "rt") as f:
        return files[-1].name[5:20], json.load(f)


def odds_view(ev: dict, allowed: set) -> dict:
    home, away = ev["home_team"], ev["away_team"]
    pts, tots, pin, ml_best, sp_best, tot_best, ml_fair = [], [], {}, {}, {}, {}, []
    for bk in ev.get("bookmakers", []):
        key = bk["key"]
        mk = {m["key"]: {o["name"]: o for o in m["outcomes"]} for m in bk.get("markets", [])}
        sp, h2, tt = mk.get("spreads", {}), mk.get("h2h", {}), mk.get("totals", {})
        if key in ("kalshi", "polymarket", "novig", "prophetx", "betopenly"):
            continue
        if home in sp and sp[home].get("point") is not None:
            pts.append(sp[home]["point"])
        if "Over" in tt and tt["Over"].get("point") is not None:
            tots.append(tt["Over"]["point"])
        if key == "pinnacle":
            if home in sp:
                pin["spread"] = sp[home].get("point")
            if home in h2 and away in h2:
                ph, pa = implied(h2[home]["price"]), implied(h2[away]["price"])
                pin["home_prob"] = round(ph / (ph + pa), 4)
            if "Over" in tt:
                pin["total"] = tt["Over"].get("point")
        if home in h2 and away in h2 and key not in ("pinnacle",):
            ph, pa = implied(h2[home]["price"]), implied(h2[away]["price"])
            ml_fair.append(ph / (ph + pa))
        if key in allowed:
            for side, nm in (("home", home), ("away", away)):
                if nm in h2 and (side not in ml_best or h2[nm]["price"] > ml_best[side]["price"]):
                    ml_best[side] = {"price": h2[nm]["price"], "book": bk.get("title", key)}
                o = sp.get(nm)
                if o and o.get("point") is not None:
                    cur = sp_best.get(side)
                    if cur is None or (o["point"], o["price"]) > (cur["point"], cur["price"]):
                        sp_best[side] = {"point": o["point"], "price": o["price"], "book": bk.get("title", key)}
            for side in ("Over", "Under"):
                o = tt.get(side)
                if o and o.get("point") is not None:
                    cur = tot_best.get(side)
                    better = cur is None or ((o["point"] < cur["point"]) if side == "Over" else (o["point"] > cur["point"])) \
                        or (o["point"] == cur["point"] and o["price"] > cur["price"])
                    if better:
                        tot_best[side] = {"point": o["point"], "price": o["price"], "book": bk.get("title", key)}
    return {"spread": statistics.median(pts) if pts else None, "total": statistics.median(tots) if tots else None,
            "home_prob": round(statistics.median(ml_fair), 4) if ml_fair else None, "n_books": len(pts),
            "pinnacle": pin, "best_ml": ml_best, "best_spread": sp_best, "best_total": tot_best}


# ------------------------------------------------------------------------------------- run
def refresh(now: datetime, force: bool = False) -> bool:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    last = datetime.fromisoformat(state["cfbd"]) if state.get("cfbd") else None
    key = os.environ.get("CFBD_API_KEY")
    if not key or (not force and last and now - last < timedelta(hours=12)):
        return False
    s = season_now(now)
    res = F.pull([s], key)
    state["cfbd"] = now.isoformat(timespec="minutes"); state["cfbd_calls"] = res["calls"]
    STATE.parent.mkdir(parents=True, exist_ok=True); STATE.write_text(json.dumps(state, indent=1))
    print("cfb: CFBD refreshed", res)
    return True


def run(now: datetime | None = None, force: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    season = season_now(now)
    refreshed = refresh(now, force)
    model_path = OUT / "cfb_model.json"
    if refreshed or force or not CACHE.exists() or not model_path.exists():
        G = build_features(season)
        coef = fit_models(G, season)
        coef["prev"] = fit_models(G, season - 1)       # fit only on earlier seasons: honest scorecard for this season
        coef["backtest"] = backtest(G, season)
        model_path.write_text(json.dumps(coef, indent=1))
        G[G.season == season].to_parquet(CACHE)
    coef = json.loads(model_path.read_text())
    G = pd.read_parquet(CACHE)
    G["model_margin"] = apply(G, coef["margin"]).round(1)
    G["model_total"] = apply(G, coef["total"]).round(1)
    allowed = set(json.loads((ROOT / "my_books.json").read_text()).get("allowed_books", []))
    snap_at, events = latest_snapshot()
    match = team_matcher(sorted(set(G.home) | set(G.away)))
    by_pair = {}
    for ev in events:
        h, a = match(ev["home_team"]), match(ev["away_team"])
        if h and a:
            by_pair[(h, a)] = ev
            by_pair[(a, h)] = ev
    up = G[(G.start > now - timedelta(hours=4)) & (G.start < now + timedelta(days=8))
           & ((G.home_div == "fbs") | (G.away_div == "fbs"))].sort_values("start")
    abbr = {}
    for t in D._load(f"teams_fbs_{season}.json.gz"):
        if t.get("abbreviation"):
            abbr[t["school"]] = t["abbreviation"]
    games = []
    for r in up.itertuples():
        ev = by_pair.get((r.home, r.away))
        o = None
        snap_t = datetime.strptime(snap_at, "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc) if snap_at else None
        if ev and snap_t and datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) <= snap_t:
            ev = None   # already kicked off at the snapshot: the feed's prices are in-game, not pre-game
        if ev:
            o = odds_view(ev, allowed)
            o["event_id"] = ev.get("id")
            try:
                from . import shop as SH
                S = SH.sharp(ev)
                az = json.loads((ROOT / "cfb_shop_rules.json").read_text())["books"]
                o["mine"] = SH.offers(ev, sorted(allowed), S)
                o["az"] = SH.offers(ev, az, S)
                o["sharp"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in S.items()}
            except Exception as e:
                print("cfb offers failed:", e)
            if match(ev["home_team"]) != r.home:   # feed lists the teams the other way round (neutral site)
                o = None
        mm = float(r.model_margin) if not pd.isna(r.model_margin) else None
        g = {"game_id": int(r.game_id), "start": r.start.isoformat(), "week": int(r.week), "home": r.home, "away": r.away,
             "home_abbr": abbr.get(r.home), "away_abbr": abbr.get(r.away),
             "home_conf": r.home_conf, "away_conf": r.away_conf, "neutral": bool(r.neutral), "conf_game": bool(r.conf_game),
             "completed": bool(r.completed), "home_pts": None if pd.isna(r.home_pts) else int(r.home_pts),
             "away_pts": None if pd.isna(r.away_pts) else int(r.away_pts),
             "model_margin": mm, "model_total": None if pd.isna(r.model_total) else float(r.model_total),
             "model_home_prob": round(0.5 * (1 + math.erf(mm / (SIGMA * math.sqrt(2)))), 3) if mm is not None else None,
             "odds": o}
        if o and o.get("spread") is not None and mm is not None:
            g["gap"] = round(mm - (-o["spread"]), 1)              # model home margin minus market home margin
        if o and o.get("total") is not None and g["model_total"] is not None:
            g["total_gap"] = round(g["model_total"] - o["total"], 1)
        games.append(g)
    try:   # AI-read college news per team (logging only), shown on the cards
        from . import news as cnews
        recent = cnews.recent_by_team(7, now)
        for g in games:
            g["news"] = {"home": recent.get(g["home"], []), "away": recent.get(g["away"], [])}
    except Exception as e:
        print("cfb news attach failed:", e)
    try:   # kickoff-window forecast for open-air games (display + logged for a later honest test)
        from . import weather as cw
        wx = cw.forecast(games, now)
        for g in games:
            if g["game_id"] in wx:
                g["weather"] = wx[g["game_id"]]
    except Exception as e:
        print("cfb weather failed:", e)
    rdir = ROOT / "output" / "research" / "cfb"
    research_md = "\n\n".join(f.read_text() for f in (rdir / "round2.md", rdir / "wind_screen.md", rdir / "holdout.md", rdir / "factor_screen.md") if f.exists())
    res = {"generated_at": now.isoformat(timespec="minutes"), "season": season,
           "odds_checked_at": (datetime.strptime(snap_at, "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc).isoformat(timespec="minutes") if snap_at else None),
           "model": {"n_games_fit": coef.get("n_margin"), "note": "Model spreads are display only (no model rule passed its holdout). College bets come only from price rules: shop-vs-sharp (spreads, totals, moneylines vs Pinnacle) and the moneyline price track."},
           "games": games, "research_md": research_md,
           "matched_odds": sum(1 for g in games if g["odds"]), "n_games": len(games)}
    try:   # college paper track (P1/P2 moneyline price rules) + phone alerts for new bets
        from . import tracks
        res["cfb_ml_bets"] = tracks.process(now)
        if res["cfb_ml_bets"]["new"]:
            from nflpred import notify
            prev = set()
            notify.new_bets({"upcoming": [], "cfb_ml_bets": res["cfb_ml_bets"]}, prev)
    except Exception as e:
        res["cfb_ml_bets"] = {"error": str(e)}
        print("cfb track failed:", e)
    try:   # shop-vs-sharp paper track (spreads, totals, moneylines vs Pinnacle) + phone alerts
        from . import shop
        res["cfb_shop_bets"] = shop.process(now)
        if res["cfb_shop_bets"]["new"]:
            from nflpred import notify
            notify.new_bets({"upcoming": [], "cfb_shop_bets": res["cfb_shop_bets"]}, set())
    except Exception as e:
        res["cfb_shop_bets"] = {"error": str(e)}
        print("cfb shop track failed:", e)
    try:   # this season so far: model (fit on earlier seasons only) vs Vegas closing line, game by game
        prev = coef.get("prev") or coef
        done = G[G.completed & G.margin.notna() & ((G.home_div == "fbs") | (G.away_div == "fbs"))].copy()
        done["mm"], done["mt"] = apply(done, prev["margin"]), apply(done, prev["total"])
        L = D.lines([season])
        done = done.merge(L, on="game_id", how="inner").dropna(subset=["spread_close"])
        done["vm"], done["vt"] = -done.spread_close, done.total_close
        res["season_to_date"] = {"summary": score_games(done, season) if len(done) else None,
                                 "games": [{"game_id": int(r.game_id), "week": int(r.week), "start": r.start.isoformat(), "home": r.home, "away": r.away,
                                            "home_abbr": abbr.get(r.home), "away_abbr": abbr.get(r.away),
                                            "home_pts": int(r.home_pts), "away_pts": int(r.away_pts),
                                            "model_margin": round(float(r.mm), 1), "model_total": round(float(r.mt), 1),
                                            "vegas_margin": float(r.vm), "vegas_total": None if pd.isna(r.vt) else float(r.vt)}
                                           for r in done.sort_values("start").itertuples()]}
        res["backtest"] = coef.get("backtest") or []
    except Exception as e:
        print("cfb season scorecard failed:", e)
    try:   # body-clock paper track (rule F2; failed its screen, tracked on request)
        from . import bodyclock
        res["cfb_body_clock_bets"] = bodyclock.process(now)
        if res["cfb_body_clock_bets"]["new"]:
            from nflpred import notify
            notify.new_bets({"upcoming": [], "cfb_body_clock_bets": res["cfb_body_clock_bets"]}, set())
    except Exception as e:
        res["cfb_body_clock_bets"] = {"error": str(e)}
        print("cfb body clock failed:", e)
    (OUT / "cfb_predictions.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"cfb: {len(games)} games this week, {res['matched_odds']} with live odds")
    return res


if __name__ == "__main__":
    run(force="--force" in sys.argv)
