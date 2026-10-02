"""Roster snapshot and team check for AI-read news signals (news_llm.py).

The AI reader sometimes tags the wrong team ("Jalon Daniels will start" tagged KC; he is TB's QB). Full runs
(daily + Friday, the ones that pull Sleeper's player list in news.fetch_live) save a compact roster snapshot to
history/roster.json:

    {"as_of": ISO time, "source": "sleeper", "n_players": int,
     "players": {normalized name: [{"name", "team", "position", "gsis_id", "depth_order"}, ...]}}

Keys are normalized names (accents, punctuation and Jr./III suffixes removed, "J. J." == "J.J." == "JJ"); a key
maps to a list because names repeat across the league (two Josh Allens). Each signal then gets

    verify = {"roster": "match" | "team_mismatch" | "not_found" | "ambiguous", "roster_team", "gsis_id",
              "roster_source", "roster_as_of"}
    team_verified = the claimed team on "match", the roster team on "team_mismatch" (unambiguous correction;
                    Sleeper roster only), else None (team unclear)

The original AI fields are never changed. Before the first Sleeper snapshot exists, `from_nflverse` builds a
fallback roster from nflverse players.parquet (latest_team; can lag trades by days); rows checked against it
are re-checked once a Sleeper snapshot exists (`backfill`).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .player_stats import norm_name

NICK = {"mike": "michael", "matt": "matthew", "chris": "christopher", "josh": "joshua", "nick": "nicholas",
        "tony": "anthony", "jim": "james", "jimmy": "james", "joe": "joseph", "dan": "daniel", "danny": "daniel",
        "ben": "benjamin", "will": "william", "bill": "william", "bob": "robert", "rob": "robert",
        "nate": "nathan", "zach": "zachary", "zack": "zachary", "gabe": "gabriel", "alex": "alexander",
        "sam": "samuel", "tom": "thomas", "jon": "jonathan", "cam": "cameron", "dave": "david", "ken": "kenneth",
        "greg": "gregory", "jeff": "jeffrey", "steve": "steven", "tim": "timothy", "pat": "patrick"}


def key(name: str) -> str:
    """Normalized name key: norm_name (accents, punctuation, suffixes) + leading initials joined ("j j" -> "jj")."""
    toks = norm_name(name or "").split()
    out, buf = [], ""
    for t in toks:
        if len(t) == 1:
            buf += t
            continue
        if buf:
            out.append(buf)
            buf = ""
        out.append(t)
    if buf:
        out.append(buf)
    return " ".join(out)


def _first_ok(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a, b = NICK.get(a, a), NICK.get(b, b)
    return a.startswith(b) or b.startswith(a) or SequenceMatcher(None, a, b).ratio() >= 0.75


# ---------------------------------------------------------------------------------------------- build / save
def build(live: pd.DataFrame, source: str = "sleeper", as_of: datetime | None = None) -> dict:
    """Roster dict from a news.parse_sleeper frame (or any frame with full_name/team/position/gsis_id)."""
    players: dict[str, list] = {}
    n = 0
    for r in live.itertuples(index=False):
        team = getattr(r, "team", None)
        if not isinstance(team, str) or not team:
            continue
        name = getattr(r, "full_name", None)
        names = {name}
        fn, ln = getattr(r, "first_name", None), getattr(r, "last_name", None)
        if isinstance(fn, str) and isinstance(ln, str):
            names.add(f"{fn} {ln}")
        do = getattr(r, "depth_order", None)
        gid = getattr(r, "gsis_id", None)
        entry = {"name": name, "team": team, "position": getattr(r, "position", None),
                 "gsis_id": gid if isinstance(gid, str) and gid else None,
                 "depth_order": int(do) if do is not None and pd.notna(do) else None}
        added = False
        for nm in names:
            k = key(nm) if isinstance(nm, str) else ""
            if not k:
                continue
            lst = players.setdefault(k, [])
            if entry not in lst:
                lst.append(entry)
                added = True
        n += added
    return {"as_of": (as_of or datetime.now(timezone.utc)).isoformat(timespec="minutes"), "source": source,
            "n_players": n, "players": players}


def from_nflverse(players: pd.DataFrame, season: int) -> dict:
    """Fallback roster from nflverse players.parquet: players active last or this season with a latest team."""
    p = players[players["latest_team"].notna() & (pd.to_numeric(players["last_season"], errors="coerce") >= season - 1)
                & ~players["status"].isin(["CUT", "RET"])]
    from .news import TEAM_MAP
    frame = pd.DataFrame({"full_name": p["display_name"], "first_name": (p["football_name"].fillna(p["first_name"]) if "football_name" in p
                                                         else p["first_name"]),
                          "last_name": p["last_name"], "team": p["latest_team"].map(lambda t: TEAM_MAP.get(t, t)),
                          "position": p["position"], "gsis_id": p["gsis_id"], "depth_order": None})
    return build(frame, source="nflverse")


def save(roster: dict, history_dir: Path) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / "roster.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(roster, separators=(",", ":")))
    os.replace(tmp, path)
    return path


def load(history_dir: Path) -> dict | None:
    try:
        r = json.loads((history_dir / "roster.json").read_text())
        return r if r.get("players") else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------- verify
def _candidates(name: str, roster: dict) -> list[dict]:
    pl = roster["players"]
    k = key(name)
    if not k:
        return []
    if k in pl:
        return list(pl[k])
    toks = k.split()
    last = toks[-1]
    same_last = [(kk, e) for kk, lst in pl.items() if kk.split()[-1] == last for e in lst]
    if len(toks) == 1:  # last name only ("Daniels")
        out = [e for _, e in same_last]
    else:
        out = [e for kk, e in same_last if _first_ok(toks[0], kk.split()[0])]
    uniq = []
    for e in out:
        if e not in uniq:
            uniq.append(e)
    return uniq


def verify(signal: dict, roster: dict) -> tuple[dict, str | None]:
    """(verify dict, team_verified) for one AI signal against a roster snapshot."""
    claimed = signal.get("team")
    base = {"roster_source": roster.get("source"), "roster_as_of": roster.get("as_of")}
    c = _candidates(signal.get("player") or "", roster)
    pos = (signal.get("position") or "").upper() or ("QB" if signal.get("is_starting_qb_news") else "")
    if pos and len(c) > 1:
        same_pos = [e for e in c if (e.get("position") or "").upper() == pos]
        c = same_pos or c
    if not c:
        return {"roster": "not_found", **base}, None
    on_team = [e for e in c if e["team"] == claimed]
    if on_team:
        return {"roster": "match", "roster_team": claimed, "gsis_id": on_team[0].get("gsis_id"), **base}, claimed
    teams = {e["team"] for e in c}
    if len(teams) == 1:
        t = c[0]["team"]
        # Correct the team only from a fresh (Sleeper) roster; the nflverse fallback can lag trades by days.
        return {"roster": "team_mismatch", "roster_team": t, "gsis_id": c[0].get("gsis_id") if len(c) == 1 else None,
                **base}, (t if roster.get("source") == "sleeper" else None)
    return {"roster": "ambiguous", "roster_teams": sorted(teams), **base}, None


def attach(signal: dict, roster: dict | None) -> dict:
    """Add verify/team_verified in place (no-op without a roster). Returns the signal."""
    if roster:
        v, tv = verify(signal, roster)
        signal["verify"] = v
        signal["team_verified"] = tv
    return signal


def backfill(history_dir: Path, roster: dict | None, log_name: str = "news_llm.jsonl") -> dict:
    """Check existing signals: rows with no `verify`, and rows checked against the nflverse fallback once a
    Sleeper roster exists. Rewrites the log atomically (unparseable lines kept as-is). Returns counts."""
    path = history_dir / log_name
    counts = {"checked": 0}
    if not roster or not path.exists():
        return counts
    lines, out = path.read_text().splitlines(), []
    for line in lines:
        try:
            s = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if not isinstance(s, dict):
            out.append(line)
            continue
        v = s.get("verify")
        if v is None or (v.get("roster_source") != "sleeper" and roster.get("source") == "sleeper"):
            attach(s, roster)
            counts["checked"] += 1
        st = (s.get("verify") or {}).get("roster", "unchecked")
        counts[st] = counts.get(st, 0) + 1
        out.append(json.dumps(s))
    if counts["checked"]:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(out) + "\n")
        os.replace(tmp, path)
    return counts
