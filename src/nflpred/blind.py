"""Blind evaluation: new models are judged on data nobody has looked at, once.

Two blind views (set up 2026-10-30, at Tyler's request: "test new models independently without peeking"):

1. SEALED VAULT: NFL seasons 2002-2011. No model study has used them (every model, feature and market study used
   2012+ or 2015+; only the situational-factor screen scripts/research/walters_factors.py read 2003-11 *schedules*,
   so the vault is not used for situational-factor ideas). data.load_pbp / load_injuries / load_snaps refuse vault
   seasons unless called inside `vault_access()`, which only `open_vault` uses.
   A model must first be REGISTERED (name + its pre-registration file, hashed and committed). `open_vault` then runs
   it ONCE, walk-forward by season (train on vault seasons before S, from 2002; test S = 2005..2011), next to the
   production model rebuilt the same way, and the result is written permanently to blind/registry.json. A second
   call for the same model is refused; a changed model needs a new name.

2. PROSPECTIVE LOG: every full run writes each registered live candidate's probabilities for upcoming games (before
   kickoff) to blind/prospective/<name>.jsonl; the git commit time proves they were made before the games. Scores
   are computed only at fixed CHECKPOINTS (blind/checkpoints.json), for games before each checkpoint, by `reveal`,
   and only once that date has passed. Nothing about them is shown on the page before then.

Judged by log loss and Brier vs the production model and vs the closing line, with a paired standard error.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "blind"
VAULT = tuple(range(2002, 2012))     # sealed seasons
WARMUP = 2001                         # features need one prior season (not scored)
FIRST_TEST = 2005                     # first vault season scored (3 seasons of training)
_ACCESS = {"open": False}


class VaultSealed(RuntimeError):
    pass


def guard(seasons) -> None:
    """Called by the data loaders: vault seasons only inside vault_access()."""
    if _ACCESS["open"]:
        return
    hit = sorted({int(s) for s in seasons} & set(VAULT) | ({WARMUP} & {int(s) for s in seasons}))
    if hit:
        raise VaultSealed(f"seasons {hit} are the sealed blind vault (src/nflpred/blind.py); register a model and "
                          f"use blind.open_vault instead")


@contextlib.contextmanager
def vault_access():
    _ACCESS["open"] = True
    try:
        yield
    finally:
        _ACCESS["open"] = False


def _registry(path: Path | None = None) -> dict:
    p = path or DIR / "registry.json"
    return json.loads(p.read_text()) if p.exists() else {"models": {}}


def _save(reg: dict, path: Path | None = None) -> None:
    p = path or DIR / "registry.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=1))


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def register(name: str, description: str, prereg: Path, kinds=("vault", "prospective"), path: Path | None = None) -> dict:
    """Record a candidate before any blind result exists. The pre-registration file's hash is stored, so the rules
    can't be changed after seeing results."""
    reg = _registry(path)
    if name in reg["models"]:
        raise ValueError(f"{name} is already registered; a changed model needs a new name")
    reg["models"][name] = {"description": description, "prereg": _rel(prereg),
                           "prereg_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(), "kinds": list(kinds),
                           "registered_at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "commit": _commit(),
                           "vault_result": None}
    _save(reg, path)
    return reg["models"][name]


def paired(y, p_new, p_ref) -> dict:
    """Log loss / Brier of new vs reference on the same games, with a paired SE (+ = new is better)."""
    rows = [(a, b, c) for a, b, c in zip(y, p_new, p_ref) if a == a and b == b and c == c]
    if not rows:
        return {"n": 0}
    ll = lambda yy, q: -(yy * math.log(min(max(q, 1e-6), 1 - 1e-6)) + (1 - yy) * math.log(1 - min(max(q, 1e-6), 1 - 1e-6)))
    d = [ll(a, c) - ll(a, b) for a, b, c in rows]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
    return {"n": len(d), "log_loss_new": round(sum(ll(a, b) for a, b, _ in rows) / len(d), 5),
            "log_loss_ref": round(sum(ll(a, c) for a, _, c in rows) / len(d), 5),
            "brier_new": round(sum((b - a) ** 2 for a, b, _ in rows) / len(d), 5),
            "brier_ref": round(sum((c - a) ** 2 for a, _, c in rows) / len(d), 5),
            "improvement": round(m, 5), "se": round(sd / math.sqrt(len(d)), 5) if len(d) > 1 else None,
            "z": round(m / (sd / math.sqrt(len(d))), 2) if len(d) > 1 and sd > 0 else None}


def open_vault(name: str, fit_predict, path: Path | None = None, data=None) -> dict:
    """Run a registered model ONCE on the vault.

    fit_predict(train_df, test_df) -> probabilities for test_df (home win). Both frames are production feature
    frames (features.build_features) for vault seasons; the model may add its own columns from them.
    `data` (tests only): a ready feature frame instead of loading the vault."""
    reg = _registry(path)
    m = reg["models"].get(name)
    if m is None:
        raise ValueError(f"register {name} first (blind.register)")
    if "vault" not in m["kinds"]:
        raise ValueError(f"{name} is registered for the prospective log only")
    if m.get("vault_result") is not None:
        raise ValueError(f"{name} has already had its one look at the vault ({m['vault_result']['at']})")
    import numpy as np
    from . import model as M
    if data is None:
        from . import data as D, features as F
        with vault_access():
            g = D.load_schedules()
            g = g[g.season.between(WARMUP, max(VAULT))]
            pbp = D.load_pbp(range(WARMUP, max(VAULT) + 1))
            old = F.prepare_schedule.__defaults__
            F.prepare_schedule.__defaults__ = (WARMUP,)
            try:
                data = F.build_features(g, pbp)
            finally:
                F.prepare_schedule.__defaults__ = old
    df = data[data.season.isin(VAULT) & data.home_win.notna()].copy()
    ys, pn, pb, pv = [], [], [], []
    for s in range(FIRST_TEST, max(VAULT) + 1):
        tr, te = df[df.season < s], df[df.season == s]
        if te.empty:
            continue
        pn += list(np.asarray(fit_predict(tr.copy(), te.copy()), float))
        base = M.make_model(M.PRODUCTION)
        base = base.fit(tr[M.FEATURES], (tr.home_score - tr.away_score).values) if M.PRODUCTION == "margin" \
            else base.fit(tr[M.FEATURES], tr.home_win.astype(int))
        pb += list(M.predict(base, te))
        ys += list(te.home_win.astype(float))
        pv += list(te.get("vegas_home_prob", te.home_win * float("nan")).astype(float))
    res = {"at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "commit": _commit(),
           "seasons_scored": f"{FIRST_TEST}-{max(VAULT)}", "vs_production": paired(ys, pn, pb),
           "vs_closing_line": paired(ys, pn, pv), "production_vs_closing_line": paired(ys, pb, pv)}
    reg["models"][name]["vault_result"] = res
    _save(reg, path)
    return res


# ---------------------------------------------------------------------------------------------- prospective log
def log_prospective(name: str, rows: list[dict], now: datetime | None = None, folder: Path | None = None) -> int:
    """rows: [{game_id, kickoff, home_win_prob}] for games not started yet. Appended with the logging time."""
    now = now or datetime.now(timezone.utc)
    folder = folder or DIR / "prospective"
    folder.mkdir(parents=True, exist_ok=True)
    keep = [r for r in rows if r.get("kickoff") and r["kickoff"] > now.isoformat()]
    with (folder / f"{name}.jsonl").open("a") as f:
        for r in keep:
            f.write(json.dumps({"logged_at": now.isoformat(timespec="minutes"), **r}) + "\n")
    return len(keep)


def last_before_kickoff(folder: Path, name: str) -> dict:
    """game_id -> the last probability logged before kickoff."""
    p = folder / f"{name}.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["logged_at"] < r["kickoff"]:
                if r["game_id"] not in out or r["logged_at"] >= out[r["game_id"]]["logged_at"]:
                    out[r["game_id"]] = r
    return out


def reveal(results: dict, now: datetime | None = None, folder: Path | None = None, checkpoints: Path | None = None) -> dict:
    """Score every prospective candidate at each checkpoint that has passed (games before that date only).
    results: game_id -> {home_win (1/0/0.5), vegas_home_prob (closing)}. Writes blind/revealed.json."""
    now = now or datetime.now(timezone.utc)
    folder = folder or DIR / "prospective"
    cps = json.loads((checkpoints or DIR / "checkpoints.json").read_text())["checkpoints"]
    base = last_before_kickoff(folder, "production")
    out = {"computed_at": now.isoformat(timespec="minutes"), "checkpoints": {}}
    names = sorted(p.stem for p in folder.glob("*.jsonl") if p.stem != "production")
    for cp in cps:
        if cp > now.date().isoformat():
            out["checkpoints"][cp] = "sealed until this date"
            continue
        blk = {}
        for n in names + ["production"]:
            mine = last_before_kickoff(folder, n)
            g = [k for k, r in mine.items() if r["kickoff"][:10] < cp and k in results and k in base]
            ys = [results[k]["home_win"] for k in g]
            blk[n] = {"vs_production": paired(ys, [mine[k]["home_win_prob"] for k in g], [base[k]["home_win_prob"] for k in g]),
                      "vs_closing_line": paired(ys, [mine[k]["home_win_prob"] for k in g],
                                                [results[k].get("vegas_home_prob", float("nan")) for k in g])}
        out["checkpoints"][cp] = blk
    (folder.parent / "revealed.json").write_text(json.dumps(out, indent=1))
    return out
