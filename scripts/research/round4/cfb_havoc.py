"""Round 4, college: havoc (pre-registered in output/research/round4/cfb_havoc_prereg.json, 2026-10-12).

Run: PYTHONPATH=src python scripts/research/round4/cfb_havoc.py -> output/research/round4/cfb_havoc.json
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "research")]
from cfbpred import data as D, pipeline as P  # noqa: E402
import matchups_refs as MR  # noqa: E402

OUT = ROOT / "output" / "research" / "round4"
HL, K = 6, 150


def havoc_long() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data" / "cfb" / "raw").glob("stats_game_havoc_*.json.gz")):
        for r in json.load(gzip.open(f)):
            o, d = r.get("offense") or {}, r.get("defense") or {}
            rows.append({"game_id": r["gameId"], "team": r["team"], "o_ev": o.get("totalHavocEvents") or 0, "o_pl": o.get("totalPlays") or 0,
                         "d_ev": d.get("totalHavocEvents") or 0, "d_pl": d.get("totalPlays") or 0})
    return pd.DataFrame(rows)


def ratings(G: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat([G[["game_id", "start", "home"]].rename(columns={"home": "team"}),
                      G[["game_id", "start", "away"]].rename(columns={"away": "team"})])
    long = long.merge(H, on=["game_id", "team"], how="left").fillna({"o_ev": 0, "o_pl": 0, "d_ev": 0, "d_pl": 0})
    long = long.sort_values(["team", "start", "game_id"]).reset_index(drop=True)
    # league rate from all strictly earlier team-games (by kickoff time)
    by = long.groupby("start")[["o_ev", "o_pl", "d_ev", "d_pl"]].sum().sort_index()
    prior = by.cumsum().shift(1).fillna(0)                 # totals over strictly earlier kickoff times
    pr = prior.reindex(long.start.values).reset_index(drop=True)
    lg_rate_o = pd.Series((pr.o_ev / pr.o_pl.replace(0, np.nan)).values, index=long.index)
    lg_rate_d = pd.Series((pr.d_ev / pr.d_pl.replace(0, np.nan)).values, index=long.index)
    out = long[["game_id", "team"]].copy()
    for side, lr in (("o", lg_rate_o), ("d", lg_rate_d)):
        Sn = MR._decayed_prior_sum(long, "team", f"{side}_ev", HL)
        Sd = MR._decayed_prior_sum(long, "team", f"{side}_pl", HL)
        out[side] = (Sn - lr.fillna(0.15) * Sd) / (Sd + K)
    out["net"] = out.d - out.o            # created minus allowed (good-oriented)
    return out


def main():
    G = P.build_features(2025)
    H = havoc_long()
    R = ratings(G, H)
    for side in ("home", "away"):
        G = G.merge(R.rename(columns={"team": side, "net": f"hv_{side[0]}"})[["game_id", side, f"hv_{side[0]}"]], on=["game_id", side], how="left")
    G["havoc_d"] = G.hv_h.fillna(0) - G.hv_a.fillna(0)
    L = D.lines(range(2019, 2026), prefer=D.PREFERRED_PROVIDERS)
    G = G.merge(L[["game_id", "spread_close"]], on="game_id", how="left")
    G["res_m"] = G.margin + G.spread_close
    out = {"prereg": "output/research/round4/cfb_havoc_prereg.json", "havoc_games": int(H.game_id.nunique())}
    # persistence: pre-game rating vs that game's net havoc
    gh = H.assign(net_g=H.d_ev / H.d_pl.replace(0, np.nan) - H.o_ev / H.o_pl.replace(0, np.nan))
    pr = R.merge(gh[["game_id", "team", "net_g"]], on=["game_id", "team"]).dropna()
    out["persistence_corr"] = round(float(pr.net.corr(pr.net_g)), 3)
    # model test
    res, orig = {}, list(P.MCOLS)
    for name, cols in (("baseline", P.MCOLS), ("with_havoc", P.MCOLS + ["havoc_d"])):
        P.MCOLS = cols
        per = {}
        for lab, seasons in (("dev_2021_22", (2021, 2022)), ("hold_2023_25", (2023, 2024, 2025))):
            err, ll = [], []
            for s in seasons:
                c = P.fit_models(G, s - 1)
                g = G[(G.season == s) & (G.home_div == "fbs") & (G.away_div == "fbs") & G.margin.notna()].dropna(subset=cols)
                mm = P.apply(g, c["margin"])
                err.append((g.margin - mm).abs())
                p = 0.5 * (1 + np.vectorize(math.erf)(mm / (P.SIGMA * math.sqrt(2))))
                y = (g.margin > 0).astype(float)
                ll.append(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))[g.margin != 0])
            per[lab] = {"mae": round(float(pd.concat(err).mean()), 3), "log_loss": round(float(pd.concat(ll).mean()), 4), "n": int(sum(len(e) for e in err))}
        res[name] = per
    P.MCOLS = orig
    b = res["baseline"]
    w = res["with_havoc"]
    w["d_mae_dev"] = round(w["dev_2021_22"]["mae"] - b["dev_2021_22"]["mae"], 3)
    w["d_mae_hold"] = round(w["hold_2023_25"]["mae"] - b["hold_2023_25"]["mae"], 3)
    w["adopt"] = bool(w["d_mae_dev"] <= -0.05 and w["d_mae_hold"] <= -0.05)
    out["model"] = res
    # market test
    fb = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.res_m.notna()]
    a = MR.ols_z(fb[fb.season.isin([2020, 2021, 2022])].res_m, fb[fb.season.isin([2020, 2021, 2022])].havoc_d)
    h = MR.ols_z(fb[fb.season.isin([2023, 2024, 2025])].res_m, fb[fb.season.isin([2023, 2024, 2025])].havoc_d)
    sign = 1 if (a.get("b") or 0) >= 0 else -1
    p = 0.5 * math.erfc(sign * h["t"] / math.sqrt(2)) if h.get("t") is not None else 1.0
    out["market"] = {"dev_2020_22": a, "hold_2023_25": h, "tested_sign": sign, "holdout_p_one_sided": round(p, 4), "pass": bool(p < 0.05)}
    print(json.dumps(out, indent=1, default=str))
    (OUT / "cfb_havoc.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
