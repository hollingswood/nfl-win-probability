"""Replay the frozen betting rules (spread v1, moneyline v1, grading v1) on 2020-2025 using the
lines that were actually on the board when the live pipeline would have run.

Data: data/historical_odds/ (odds_history.py). Model: walk-forward (trained only on seasons before
the one being predicted), exactly like the backtest. Rules and blend weights are read unchanged
from spread_rules.json / betting_rules.json (all fit on 2015-2019 or earlier).

How each game is replayed (mirrors the live pipeline):
  * snapshots are the live run times (daily 7:10am AZ, Friday report run, 75 min pre-kickoff)
  * "injury report published": a game is only eligible from the first daily run after the
    first official practice report of that game week (3 days before a Sun/Mon game, 2 before
    Thu/Sat and others)
  * "starting QBs confirmed": if any QB on either team was listed Questionable/Doubtful on the
    final report, only the pre-kickoff snapshot (after inactives) is eligible; if one was Out,
    eligibility starts a day later (final report)
  * consensus = median over all books; best price / line shopping only over my_books.json books
  * "line moved against us" uses the first snapshot seen for the game (up to 9 days out)
  * a bet locks at the first qualifying snapshot; weekly 8-unit cap applied in time order
  * results graded on final scores; CLV against the nflverse closing line (same as live)

Known optimism, measured separately in the robustness section: the model's QB/injury inputs are
the final-report versions, a day or two newer than an early-week snapshot would have had.

    PYTHONPATH=src python scripts/replay_early_lines.py      # writes output/replay_early_lines.json
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nflpred import bets as ML, spread_bets as SB, grading as G, model as M, odds as O
from nflpred.weather import _kickoff_utc

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "replay_predictions.parquet"


# ---------------------------------------------------------------- inputs
def walk_forward() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import datetime as dt
    from nflpred.pipeline import build
    df = build(refresh=False, today=dt.date(2026, 9, 1), live_news=False)
    parts = []
    for s in range(2020, 2026):
        te = df[(df.season == s) & df.home_win.notna()].copy()
        mdl = M.fit(df, before_season=s)
        te["mu_model"] = mdl.predict_margin(te)
        te["p_model"] = M.predict(mdl, te)
        parts.append(te)
    keep = ["game_id", "season", "week", "game_type", "gameday", "gametime", "weekday", "home_team", "away_team",
            "home_score", "away_score", "spread_line", "vegas_home_prob", "home_qb_id", "away_qb_id",
            "home_qb_change", "away_qb_change", "mu_model", "p_model"]
    out = pd.concat(parts)[[c for c in keep if c in parts[0].columns]].reset_index(drop=True)
    CACHE.parent.mkdir(exist_ok=True)
    out.to_parquet(CACHE)
    return out


def load_odds() -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / "data" / "historical_odds" / "nfl_odds_*.csv.gz")))
    o = pd.concat([pd.read_csv(f).assign(season=int(Path(f).name[9:13])) for f in fs], ignore_index=True)
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    return o


def match_games(o: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    ev = o.drop_duplicates("event_id")[["event_id", "home", "away", "commence", "season"]]
    m = ev.merge(g[["game_id", "home_team", "away_team", "kick", "season"]],
                 left_on=["home", "away", "season"], right_on=["home_team", "away_team", "season"])
    m["dt"] = (m.commence - m.kick).abs().dt.total_seconds()
    m = m[m.dt < 48 * 3600]
    return o.merge(m[["event_id", "game_id"]], on="event_id")


def _nv(ml_h, ml_a):
    ih = -ml_h / (-ml_h + 100) if ml_h < 0 else 100 / (ml_h + 100)
    ia = -ml_a / (-ml_a + 100) if ml_a < 0 else 100 / (ml_a + 100)
    return ih / (ih + ia)


def snapshots(o: pd.DataFrame, allowed: set) -> pd.DataFrame:
    """One row per (game, snapshot): consensus + the allowed books' prices."""
    o = o.copy()
    ok = o.ml_home.notna() & o.ml_away.notna()
    o.loc[ok, "nv_home"] = [_nv(a, b) for a, b in zip(o.loc[ok, "ml_home"], o.loc[ok, "ml_away"])]
    o["home_margin"] = -o.sp_home_point
    grp = o.groupby(["game_id", "requested_ts"])
    cons = grp.agg(p_cons=("nv_home", "median"), m_cons=("home_margin", "median"), n_books=("book", "nunique"))
    al = o[o.book.isin(allowed)]
    books = al.groupby(["game_id", "requested_ts"]).apply(
        lambda d: d[["book", "ml_home", "ml_away", "sp_home_point", "sp_home_price",
                     "sp_away_point", "sp_away_price"]].to_dict("records"), include_groups=False).rename("books")
    return cons.join(books).reset_index()


def eligible_from(kick_utc: pd.Timestamp, weekday: str) -> pd.Timestamp:
    days = 3 if weekday in ("Sunday", "Monday") else 2
    et_day = kick_utc.tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Timestamp(et_day - pd.Timedelta(days=days)).tz_localize("UTC") + pd.Timedelta(hours=14, minutes=10)


def qb_flags(g: pd.DataFrame) -> tuple[set, set]:
    """Games where ANY quarterback on either team was on the final report (audit fix: the actual
    starter is only known after the fact). Questionable/Doubtful -> not confirmed until inactives,
    so only the pre-kickoff snapshot is eligible. Out -> known from the final report (a day later)."""
    frames = []
    for s in range(2020, 2026):
        p = ROOT / "data" / "raw" / f"injuries_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["season", "week", "team", "position", "report_status"]))
    inj = pd.concat(frames)
    inj = inj[inj.position == "QB"]
    qd = set(zip(*[inj[inj.report_status.isin(["Questionable", "Doubtful"])][c] for c in ("season", "week", "team")]))
    out = set(zip(*[inj[inj.report_status == "Out"][c] for c in ("season", "week", "team")]))
    q_games, out_games = set(), set()
    for r in g.itertuples():
        keys = {(r.season, r.week, r.home_team), (r.season, r.week, r.away_team)}
        if keys & qd:
            q_games.add(r.game_id)
        elif keys & out:
            out_games.add(r.game_id)
    return q_games, out_games


# ---------------------------------------------------------------- per-snapshot decisions
def spread_decision(row, snap, first_margin, rules):
    q = rules["qualify"]
    mu = SB.expected_margin(row.mu_model, snap.m_cons, rules)
    best = None
    for b in snap.books or []:
        if any(pd.isna(b[k]) for k in ("sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price")):
            continue
        for side in ("home", "away"):
            point, price = float(b[f"sp_{side}_point"]), int(b[f"sp_{side}_price"])
            ev, pw, pu = SB.side_ev(mu, point, price, side, rules)
            if best is None or ev > best["ev"]:
                best = dict(side=side, point=point, price=price, book=b["book"], ev=ev, p_cover=pw, p_push=pu)
    if best is None:
        return None
    moved = (snap.m_cons - first_margin) if best["side"] == "away" else (first_margin - snap.m_cons)
    qbf = any(abs(x or 0) > G.QB_CHANGE_FLAG for x in (row.home_qb_change, row.away_qb_change) if not pd.isna(x))
    gr = G.grade("spread", best["ev"], row.mu_model - snap.m_cons, -moved, best["point"], qbf)
    ok = (best["ev"] >= q["min_edge_at_best_price"] and q["min_american_odds"] <= best["price"] <= q["max_american_odds"]
          and moved < q["require_line_not_moved_away_since_first_seen_points"])
    p_np = best["p_cover"] / max(1 - best["p_push"], 1e-9)
    best.update(gr, units=ML.kelly_units(p_np, best["price"], rules["sizing"]), qualifies=ok, mu=mu)
    return best


def ml_decision(row, snap, first_p, rules):
    q = rules["qualify"]
    p_home = ML.blend_prob(row.p_model, snap.p_cons, rules["probability"])
    best = None
    for b in snap.books or []:
        for side, p in (("home", p_home), ("away", 1 - p_home)):
            price = b[f"ml_{side}"]
            if pd.isna(price):
                continue
            ev = p * ML.decimal(price) - 1
            if best is None or ev > best["ev"]:
                best = dict(side=side, price=int(price), book=b["book"], ev=ev, p=p)
    if best is None:
        return None
    mkt = snap.p_cons if best["side"] == "home" else 1 - snap.p_cons
    first_side = first_p if best["side"] == "home" else 1 - first_p
    pm = row.p_model if best["side"] == "home" else 1 - row.p_model
    qbf = any(abs(x or 0) > G.QB_CHANGE_FLAG for x in (row.home_qb_change, row.away_qb_change) if not pd.isna(x))
    gr = G.grade("moneyline", best["ev"], 100 * (pm - mkt), 100 * (mkt - first_side), None, qbf)
    ok = (best["ev"] >= q["min_edge_at_best_price"] and q["min_american_odds"] <= best["price"] <= q["max_american_odds"]
          and mkt >= first_side - 0.02)
    best.update(gr, units=ML.kelly_units(best["p"], best["price"], rules["sizing"]), qualifies=ok, market=mkt)
    return best


# ---------------------------------------------------------------- replay
def replay(g, snaps, track, rules, qbf, min_start=None, exclude_qb_change=False):
    questionable, qb_out = qbf
    by_game = {k: v.sort_values("requested_ts") for k, v in snaps.groupby("game_id")}
    events = []  # (time, game_id, decision)
    for row in g.itertuples():
        s = by_game.get(row.game_id)
        if s is None or s.empty:
            continue
        s = s[s.requested_ts < row.kick]
        col = "m_cons" if track == "spread" else "p_cons"
        # "first seen" = first live run with a line, which is at most 9 days out (audit fix)
        seen = s[(s.requested_ts >= row.kick - pd.Timedelta(days=9)) & s[col].notna()]
        if seen.empty:
            continue
        first = seen.iloc[0]
        start = eligible_from(row.kick, row.weekday)
        if min_start is not None:
            start = max(start, min_start(row))
        if row.game_id in questionable:
            start = max(start, row.kick - pd.Timedelta(minutes=90))
        elif row.game_id in qb_out:
            start = start + pd.Timedelta(days=1)
        if exclude_qb_change and any(abs(x or 0) > G.QB_CHANGE_FLAG for x in (row.home_qb_change, row.away_qb_change)
                                     if not pd.isna(x)):
            continue
        for snap in s[s.requested_ts >= start].itertuples():
            if pd.isna(snap.m_cons) or pd.isna(snap.p_cons):
                continue
            d = (spread_decision(row, snap, first.m_cons, rules) if track == "spread"
                 else ml_decision(row, snap, first.p_cons, rules))
            if d and d["qualifies"]:
                events.append((snap.requested_ts, row.game_id, d))
    events.sort(key=lambda e: (e[0], -e[2]["ev"]))
    placed, used = {}, {}
    rows = g.set_index("game_id")
    cap = rules["sizing"]["max_units_per_week"]
    for t, gid, d in events:
        if gid in placed:
            continue
        r = rows.loc[gid]
        wk = (r.season, r.week)
        if used.get(wk, 0) + d["units"] > cap + 1e-9:
            continue
        used[wk] = used.get(wk, 0) + d["units"]
        placed[gid] = dict(d, placed_at=t, game_id=gid, season=int(r.season), week=int(r.week))
    return [settle(b, rows.loc[b["game_id"]], track, rules) for b in placed.values()]


def settle(b, r, track, rules):
    margin = r.home_score - r.away_score
    dec = ML.decimal(b["price"])
    if track == "spread":
        adj = (margin + b["point"]) if b["side"] == "home" else (-margin + b["point"])
        res = "win" if adj > 0 else "push" if adj == 0 else "loss"
        clv = SB.side_ev(float(r.spread_line), b["point"], b["price"], b["side"], rules)[0] \
            if not pd.isna(r.spread_line) else np.nan
        close_pt = (-float(r.spread_line)) if b["side"] == "home" else float(r.spread_line)
        b["clv_points"] = b["point"] - close_pt
    else:
        won = margin > 0 if b["side"] == "home" else margin < 0
        res = "push" if margin == 0 else "win" if won else "loss"
        cp = r.vegas_home_prob if b["side"] == "home" else 1 - r.vegas_home_prob
        clv = dec * cp - 1 if not pd.isna(cp) else np.nan
    b["result"] = res
    b["profit"] = b["units"] * (dec - 1) if res == "win" else 0.0 if res == "push" else -b["units"]
    b["profit_flat"] = (dec - 1) if res == "win" else 0.0 if res == "push" else -1.0
    b["clv"] = clv
    b["hours_before"] = (r.kick - b["placed_at"]).total_seconds() / 3600
    b["favorite"] = (b["price"] < 0) if track == "moneyline" else (b["point"] < 0)
    return b


# ---------------------------------------------------------------- summaries
def summarize(bets: list[dict]) -> dict:
    if not bets:
        return {"bets": 0}
    d = pd.DataFrame(bets)
    staked = d.loc[d.result != "push", "units"].sum()
    flat = d.profit_flat.values
    clv = d.clv.dropna().values
    rng = np.random.default_rng(0)
    boot = [flat[rng.integers(0, len(flat), len(flat))].mean() for _ in range(4000)]
    z = clv.mean() / (clv.std(ddof=1) / math.sqrt(len(clv))) if len(clv) > 1 else 0
    return {"bets": len(d), "wins": int((d.result == "win").sum()), "losses": int((d.result == "loss").sum()),
            "pushes": int((d.result == "push").sum()),
            "units_staked": round(float(staked), 1), "profit_units": round(float(d.profit.sum()), 1),
            "roi_kelly": round(float(d.profit.sum() / staked), 4) if staked else 0,
            "roi_flat": round(float(flat.mean()), 4),
            "roi_flat_95ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "avg_clv": round(float(clv.mean()), 4), "clv_p_value": round(0.5 * math.erfc(z / math.sqrt(2)), 4),
            "pct_beat_close": round(float((clv > 0).mean()), 3),
            "avg_edge_claimed": round(float(d.ev.mean()), 4)}


def breakdown(bets, key):
    d = pd.DataFrame(bets)
    return {str(k): summarize(v.to_dict("records")) for k, v in d.groupby(key)} if len(d) else {}


def main():
    wf = walk_forward()
    wf["gameday"] = pd.to_datetime(wf.gameday)
    wf["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in wf.itertuples()], utc=True)
    wf = wf[wf.home_score.notna()].reset_index(drop=True)
    o = match_games(load_odds(), wf)
    allowed = O.load_allowed_books() or set(o.book.unique())
    snaps = snapshots(o, allowed)
    ques = qb_flags(wf)
    sr, mr = SB.load_rules(), ML.load_rules()
    out = {"games": int(len(wf)), "snapshots": int(snaps.requested_ts.nunique()), "allowed_books": sorted(allowed),
           "games_qb_questionable": len(ques[0]), "games_qb_out": len(ques[1])}

    fri = lambda row: eligible_from(row.kick, row.weekday) + pd.Timedelta(days=1)
    for track, rules in (("spread", sr), ("moneyline", mr)):
        bets = replay(wf, snaps, track, rules, ques)
        res = {"main": summarize(bets),
               "by_season": breakdown(bets, "season"),
               "by_grade": breakdown(bets, "grade"),
               "by_edge": breakdown([dict(b, eb="3-5%" if b["ev"] < .05 else "5-8%" if b["ev"] < .08 else "8%+") for b in bets], "eb"),
               "by_timing": breakdown([dict(b, tb="game day (<6h)" if b["hours_before"] < 6 else "1-2 days" if b["hours_before"] < 60 else "3+ days") for b in bets], "tb"),
               "favorite_vs_dog_by_timing": breakdown([dict(b, k=("fav" if b["favorite"] else "dog") + (" early" if b["hours_before"] >= 60 else " late")) for b in bets], "k"),
               "robust_one_day_later_start": summarize(replay(wf, snaps, track, rules, ques, min_start=fri)),
               "robust_no_qb_change_games": summarize(replay(wf, snaps, track, rules, ques, exclude_qb_change=True)),
               }
        out[track] = res
        pd.DataFrame(bets).to_csv(ROOT / "output" / f"replay_{track}_bets.csv", index=False)
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "replay_early_lines.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
