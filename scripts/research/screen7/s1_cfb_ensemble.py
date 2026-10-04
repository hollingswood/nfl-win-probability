"""S1: college football ensemble of public rating systems, bet at the OPENING line (Coleman 2025).
Pre-declared 2026-10-03: ensemble = median predicted home margin across the individual systems in The Prediction
Tracker archive (market lines and Prediction Tracker aggregates excluded), >= 10 systems required.
Bet the side the ensemble favors vs the opening line when |ensemble - open| >= 3 (primary; 2 and 4 reported),
weeks 2-9 (primary) and all weeks (reported). Graded ATS at the opening number (-110 assumed: the archive has no
prices) and by CLV in points (close - open in our direction). Seasons 2010-2025; Coleman's sample was 2016-24, so
2010-15 and 2025 are outside it. Pass: one-sided p < 0.007 on cover > 52.38% AND mean CLV > 0, most seasons positive."""
import glob, json, math, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
MARKET = {"line", "lineopen", "linemidweek"}
AGG = {"lineavg", "linemedian", "linestd", "linecons", "linecong", "linepimean", "linepibias", "linepig", "linesaggm"}

def main():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/cfb/ptrack/ncaa*.csv"))):
        y = int(re.findall(r"(\d{4})", f)[-1]); d = pd.read_csv(f)
        sys_cols = [c for c in d.columns if c.startswith("line") and c not in MARKET and c not in AGG]
        S = d[sys_cols].apply(pd.to_numeric, errors="coerce")
        d["ens"] = S.median(axis=1); d["n_sys"] = S.notna().sum(axis=1); d["season"] = y
        if "actual" not in d:
            d["actual"] = pd.to_numeric(d.hscore, errors="coerce") - pd.to_numeric(d.vscore, errors="coerce")
        for c in ("linemidweek", "lineopen"):
            if c not in d:
                d[c] = np.nan
        rows.append(d[["season", "week", "Home", "Road", "line", "lineopen", "linemidweek", "ens", "n_sys", "actual"]])
    D = pd.concat(rows, ignore_index=True)
    for c in ("line", "lineopen", "linemidweek", "actual", "week"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D[D.n_sys >= 10]
    D = D[D.lineopen.notna() & D.line.notna() & D.actual.notna()]
    out = {"games": int(len(D)), "seasons": sorted(D.season.unique().tolist()),
           "mae": {"ensemble": round(float((D.actual - D.ens).abs().mean()), 2), "open": round(float((D.actual - D.lineopen).abs().mean()), 2),
                   "close": round(float((D.actual - D.line).abs().mean()), 2)}, "rules": {}}
    def ev(sub, ref, k):
        g = sub.ens - sub[ref]; m = g.abs() >= k; s = np.sign(g[m])
        r = s * (sub.actual[m] - sub[ref][m]); c, l = int((r > 0).sum()), int((r < 0).sum())
        n = c + l; z = (c - n * 0.5238) / math.sqrt(n * 0.5238 * 0.4762) if n else 0
        clv = s * (sub.line[m] - sub.lineopen[m]) if ref == "lineopen" else None
        per = pd.DataFrame({"season": sub.season[m], "r": r}).groupby("season").r.apply(lambda x: (x > 0).sum() / max(1, (x != 0).sum()))
        return {"bets": n, "cover": round(c / max(1, n), 3), "p_vs_52.38": round(0.5 * math.erfc(z / math.sqrt(2)), 4),
                "roi_at_-110": round((c * 0.9091 - l) / max(1, n), 4),
                "mean_clv_pts": round(float(clv.mean()), 3) if clv is not None else None,
                "clv_our_way_vs_against": f"{float((clv > 0).mean()):.0%} / {float((clv < 0).mean()):.0%}" if clv is not None else None,
                "seasons_above_52.4": f"{int((per > 0.524).sum())}/{len(per)}", "by_season": {int(k2): round(float(v), 3) for k2, v in per.items()}}
    W29 = D[D.week.between(2, 9)]
    for k in (2, 3, 4):
        out["rules"][f"open wk2-9 |ens-open|>={k}"] = ev(W29, "lineopen", k)
    out["rules"]["open all weeks |ens-open|>=3"] = ev(D, "lineopen", 3)
    out["rules"]["close wk2-9 |ens-close|>=3 (bet at close)"] = ev(W29, "line", 3)
    out["rules"]["2025 only (post-sample) open wk2-9 >=3"] = ev(W29[W29.season == 2025], "lineopen", 3)
    out["rules"]["2010-15 (pre-sample) open wk2-9 >=3"] = ev(W29[W29.season <= 2015], "lineopen", 3)
    (ROOT / "output/research/screen7/s1_cfb_ensemble.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
