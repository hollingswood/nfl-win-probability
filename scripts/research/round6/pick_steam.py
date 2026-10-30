"""Round 6: do lines move toward published picks AFTER we first see them (the part a follower could capture)?

For every pick the scorer counted (picks_score.run: matched to its real game, first seen before kickoff), take the
US-book consensus for the pick's side in our first odds snapshot at or after first sight (within 3 h) and in the last
snapshot before kickoff. value = points (spreads/totals) or no-vig probability (moneylines) the market moved toward
the pick between those two moments. Stage `dev` reports picks first seen up to DEV_END (exploration allowed);
the frozen rule (output/research/round6/pick_steam_frozen.json) is judged only on picks first seen after it.
"""
from __future__ import annotations

import bisect
import gzip
import json
import math
import statistics as S
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
US = {"draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet", "ballybet",
      "betparx", "fliff", "bovada", "betonlineag", "lowvig", "mybookieag", "betus"}
DEV_END = "2026-10-26T00:00+00:00"


def _t(s):
    t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


class Snaps:
    def __init__(self, folder: Path):
        self.files = sorted(folder.glob("odds_*.json.gz"))
        self.times = [datetime.strptime(f.name[5:20], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc) for f in self.files]
        self.cache = {}

    def load(self, i):
        if i not in self.cache:
            if len(self.cache) > 40:
                self.cache.clear()
            d = json.load(gzip.open(self.files[i], "rt"))
            self.cache[i] = d if isinstance(d, list) else d.get("events", [])
        return self.cache[i]

    def first_after(self, t, within=timedelta(hours=3)):
        i = bisect.bisect_left(self.times, t)
        return i if i < len(self.times) and self.times[i] - t <= within else None

    def last_before(self, t):
        i = bisect.bisect_left(self.times, t) - 1
        return i if i >= 0 else None


def consensus(ev, market, side_name, home, away):
    """Median line (spreads/totals, for side_name) or median no-vig probability (h2h) across US books."""
    pts, probs = [], []
    for b in ev.get("bookmakers", []):
        if b["key"] not in US:
            continue
        for m in b.get("markets", []):
            if m["key"] != market:
                continue
            oc = {o["name"]: o for o in m["outcomes"]}
            if market == "h2h":
                if home in oc and away in oc:
                    dec = lambda a: 1 + a / 100 if a > 0 else 1 + 100 / -a
                    ph, pa = 1 / dec(oc[home]["price"]), 1 / dec(oc[away]["price"])
                    probs.append((ph if side_name == home else pa) / (ph + pa))
            elif side_name in oc and oc[side_name].get("point") is not None:
                pts.append(oc[side_name]["point"])
    if market == "h2h":
        return S.median(probs) if probs else None
    return S.median(pts) if pts else None


def picks_scored():
    """Re-run the scorer in a temp copy and return its graded + pending rows (ids, game, side, seen_at, kick)."""
    import shutil, tempfile
    from nflpred import picks_score as PS
    tmp = Path(tempfile.mkdtemp())
    for f in ("picks_log.jsonl", "picks_reads.json", "picks_reasons.json"):
        if (ROOT / "history" / f).exists():
            shutil.copy(ROOT / "history" / f, tmp / f)
    cap = {}
    orig = PS.audit
    PS.audit = lambda out, g, p, now: (cap.update(g=g, p=p), orig(out, g, p, now))[1]
    try:
        PS.run(tmp, notify=False)
    finally:
        PS.audit = orig
    return cap["g"], cap["p"]


def measure(rows):
    from nflpred.picks_score import cfb_matcher, nfl_team
    nfl, cfb = Snaps(ROOT / "history"), Snaps(ROOT / "history" / "cfb")
    out = []
    for x in rows:
        snaps = nfl if x["sport"] == "nfl" else cfb
        seen, kick = _t(x["seen_at"]), _t(x["kick"])
        i0, i1 = snaps.first_after(seen), snaps.last_before(kick - timedelta(minutes=5))
        if i0 is None or i1 is None or i1 < i0:
            continue
        def find(i):
            for ev in snaps.load(i):
                if x["sport"] == "nfl":
                    h, a = nfl_team(ev["home_team"]), nfl_team(ev["away_team"])
                    ok = {h, a} == {x["home"], x["away"]}
                else:
                    m = cfb_matcher([x["home"], x["away"]])
                    ok = {m(ev["home_team"]), m(ev["away_team"])} == {x["home"], x["away"]}
                if ok and abs((_t(ev["commence_time"]) - kick).total_seconds()) < 36 * 3600:
                    return ev
            return None
        e0, e1 = find(i0), find(i1)
        if not e0 or not e1:
            continue
        mk = {"spread": "spreads", "total": "totals", "moneyline": "h2h"}[x["market"]]
        if mk == "totals":
            name = x["side"].capitalize()
        else:
            nm = (lambda t: nfl_team(t)) if x["sport"] == "nfl" else cfb_matcher([x["side"]])
            name = next((t for t in (e0["home_team"], e0["away_team"]) if nm(t) == x["side"]), None)
            if name is None:
                continue
        c0 = consensus(e0, mk, name, e0["home_team"], e0["away_team"])
        c1 = consensus(e1, mk, name, e1["home_team"], e1["away_team"])
        if c0 is None or c1 is None:
            continue
        if mk == "h2h":
            v = c1 - c0                                   # probability moved toward the pick
        elif mk == "spreads":
            v = c0 - c1                                   # e.g. +3 -> +2 or -3 -> -4: +1 for the pick
        else:
            v = (c1 - c0) if x["side"] == "over" else (c0 - c1)
        out.append({**{k: x.get(k) for k in ("picker", "sport", "market", "side", "seen_at", "kick", "result", "game_id", "line")},
                    "value": round(v, 4), "hours_seen_to_kick": round((kick - seen).total_seconds() / 3600, 1),
                    "seen_line": c0, "close_line": c1})
    return out


def summarize(rows, label):
    res = {}
    for mk in ("spread", "total", "moneyline"):
        v = [r["value"] for r in rows if r["market"] == mk]
        if len(v) > 2:
            m, se = S.mean(v), S.stdev(v) / math.sqrt(len(v))
            res[mk] = {"n": len(v), "mean": round(m, 4), "se": round(se, 4), "t": round(m / se, 2) if se else None,
                       "share_toward": round(sum(x > 0 for x in v) / len(v), 3), "share_away": round(sum(x < 0 for x in v) / len(v), 3)}
    print(label, json.dumps(res))
    return res


if __name__ == "__main__":
    g, p = picks_scored()
    rows = measure([x for x in g + p if x.get("kick")])
    dev = [r for r in rows if r["seen_at"] < DEV_END]
    out = {"all_dev": summarize(dev, "dev all")}
    by = defaultdict(list)
    for r in dev:
        by[(r["picker"], r["sport"])].append(r)
    pk = []
    for k, v in by.items():
        sp = [r["value"] for r in v if r["market"] in ("spread", "total")]
        if len(sp) >= 8:
            m, se = S.mean(sp), S.stdev(sp) / math.sqrt(len(sp))
            pk.append({"picker": k[0], "sport": k[1], "n": len(sp), "mean_pts": round(m, 3), "t": round(m / se, 2) if se else None})
    pk.sort(key=lambda r: -(r["t"] or 0))
    out["pickers_dev"] = pk
    for lab, f in (("<24h", lambda r: r["hours_seen_to_kick"] < 24), ("24-72h", lambda r: 24 <= r["hours_seen_to_kick"] < 72),
                   (">=72h", lambda r: r["hours_seen_to_kick"] >= 72)):
        out[f"dev_{lab}"] = summarize([r for r in dev if f(r)], f"dev {lab}")
    for sp in ("nfl", "cfb"):
        out[f"dev_{sp}"] = summarize([r for r in dev if r["sport"] == sp], f"dev {sp}")
    (ROOT / "output" / "research" / "round6").mkdir(parents=True, exist_ok=True)
    (ROOT / "output" / "research" / "round6" / "pick_steam_dev.json").write_text(json.dumps(out, indent=1))
    import pandas as pd
    pd.DataFrame(rows).to_parquet(ROOT / "output" / "research" / "round6" / "pick_steam_rows.parquet")
    print("pickers (dev, spreads+totals, n>=8):")
    for r in pk[:12]:
        print("  ", r)
