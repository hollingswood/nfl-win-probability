"""Round 7b: do several HOT pickers (frozen rule v1) agreeing beat the close? Pre-registered in
output/research/round7/hot_consensus_prereg.json."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "round6"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "round7"))
import consensus as C  # noqa: E402


def main():
    from pick_steam import picks_scored
    from nflpred import picks_score as PS
    graded, _ = picks_scored()
    hot = {}                                    # (sport, week) -> hot pickers that week (rule v1)
    for sport in ("nfl", "cfb"):
        rows = [x for x in graded if x["sport"] == sport]
        for wk in sorted({x["week"] for x in rows}):
            rec = defaultdict(lambda: [0, 0])
            for x in rows:
                if wk - 4 <= x["week"] < wk and x["result"] != "push":
                    rec[x["picker"]][0 if x["result"] == "win" else 1] += 1
            hot[(sport, wk)] = {k for k, (w, l) in rec.items() if PS.is_named(k) and w + l >= 8 and w / (w + l) >= 0.60}
    by = defaultdict(lambda: defaultdict(set))
    wk_of = {}
    for g in graded:
        cls = "total" if g["market"] == "total" else "side"
        k = (g["sport"], str(g["game_id"]), cls)
        by[k][g["side"]].add(g["picker"]); wk_of[k] = g["week"]; by[k]["_kick"] = g["kick"]
    rows = []
    for k, dirs in by.items():
        kick = dirs.pop("_kick")
        H = hot.get((k[0], wk_of[k]), set())
        ds = list(dirs)
        for d in ds:
            other = [o for o in ds if o != d]
            hf = len(dirs[d] & H); ha = sum(len(dirs[o] & H) for o in other); na = sum(len(dirs[o]) for o in other)
            if hf >= 1 and ha == 0:
                rows.append({"sport": k[0], "game_id": k[1], "cls": k[2], "dir": d, "F": hf, "A": na, "kick": kick, "hot_for": hf,
                             "all_against": na, "t_F": "", "t_first": "", "pickers_for": []})
    import pandas as pd
    df = C.grade(pd.DataFrame(rows), C.closes())
    groups = {"H2": df.hot_for >= 2, "H2_all": (df.hot_for >= 2) & (df.all_against == 0), "H3": df.hot_for >= 3, "H1": df.hot_for == 1}
    def one_sided(w, n, p0=0.5238):
        if not n:
            return None
        z = (w - n * p0) / math.sqrt(n * p0 * (1 - p0))
        return round(0.5 * math.erfc(z / math.sqrt(2)), 4)
    out = {"hot_picker_weeks": {f"{a} wk{b}": len(v) for (a, b), v in sorted(hot.items()) if v}}
    for name, f in groups.items():
        s = df[f]
        out[name] = {sp: C.cell_stats(s[s.sport == sp]) for sp in ("nfl", "cfb")} | {"all": C.cell_stats(s)}
        out[name]["one_sided_p_vs_52.4"] = one_sided(int(s.cover.sum()), len(s))
    out["H2_pass"] = bool(out["H2"]["all"]["n"] and out["H2"]["one_sided_p_vs_52.4"] < 0.05)
    (ROOT / "output" / "research" / "round7" / "hot_consensus.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
