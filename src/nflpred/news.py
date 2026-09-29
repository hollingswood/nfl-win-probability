"""Live injury and depth-chart news from free public feeds.

Sources (no key needed; both are called from GitHub Actions):
  * Sleeper  https://api.sleeper.app/v1/players/nfl
      every NFL player: team, position, injury_status, practice_participation, depth-chart order,
      gsis_id, news_updated. ~5 MB; Sleeper asks that it be pulled at most once a day per app,
      and our schedule pulls it a few times per game day at most.
  * ESPN     https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=<id>
      per-game injury list (unofficial public endpoint), used to fill gaps and cross-check.

What it produces, for upcoming games only:
  1. Injury rows in the same shape as the nflverse injury report, so the existing injury and
     QB-availability code uses them unchanged. Live rows replace older nflverse rows for the
     same player and week.
  2. Projected starting QB per team from the depth chart: if the listed starter is Out, the
     highest healthy QB on the depth chart starts instead.
  3. A change log shown on the dashboard, so every automatic adjustment is visible.

Precedence: overrides.json (manual) > live feeds > nflverse weekly report.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"
TEAM_MAP = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}
STATUS_MAP = {  # -> nflverse report_status vocabulary
    "Questionable": "Questionable", "Doubtful": "Doubtful", "Out": "Out", "IR": "Out", "PUP": "Out",
    "Sus": "Out", "NFI": "Out", "COV": "Out", "Suspension": "Out", "Injured Reserve": "Out",
    "Day-To-Day": "Questionable",
}
PRACTICE_MAP = {"DNP": "Did Not Participate In Practice", "Limited": "Limited Participation in Practice",
                "LP": "Limited Participation in Practice", "Full": "Full Participation in Practice",
                "FP": "Full Participation in Practice"}
UA = {  # ESPN rejects non-browser clients (403 seen from GitHub Actions)
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get_json(url: str, timeout: float = 30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---------------------------------------------------------------- Sleeper
def parse_sleeper(players: dict) -> pd.DataFrame:
    rows = []
    for p in players.values():
        if not isinstance(p, dict) or not p.get("team") or p.get("position") is None:
            continue
        rows.append({
            "gsis_id": (p.get("gsis_id") or "").strip() or None,
            "full_name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "team": TEAM_MAP.get(p["team"], p["team"]),
            "position": p.get("position"),
            "status_raw": p.get("injury_status"),
            "report_status": STATUS_MAP.get(p.get("injury_status")),
            "practice_status": PRACTICE_MAP.get(p.get("practice_participation")),
            "depth_pos": p.get("depth_chart_position"),
            "depth_order": p.get("depth_chart_order"),
            "updated": pd.to_datetime(p["news_updated"], unit="ms", utc=True) if p.get("news_updated") else pd.NaT,
            "source": "Sleeper",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ESPN
def parse_espn_summary(summary: dict, espn_to_gsis: dict) -> pd.DataFrame:
    rows = []
    for block in summary.get("injuries", []) or []:
        team = TEAM_MAP.get(block.get("team", {}).get("abbreviation"), block.get("team", {}).get("abbreviation"))
        for inj in block.get("injuries", []) or []:
            ath = inj.get("athlete", {}) or {}
            rows.append({
                "gsis_id": espn_to_gsis.get(str(ath.get("id"))),
                "full_name": ath.get("displayName"),
                "team": team,
                "position": (ath.get("position") or {}).get("abbreviation"),
                "status_raw": inj.get("status"),
                "report_status": STATUS_MAP.get(inj.get("status")),
                "practice_status": None,
                "depth_pos": None, "depth_order": None,
                "updated": pd.to_datetime(inj.get("date"), utc=True, errors="coerce"),
                "source": "ESPN",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- combine
def combine(sleeper: pd.DataFrame | None, espn: pd.DataFrame | None) -> pd.DataFrame:
    """One row per player: Sleeper first (has practice + depth chart), ESPN fills players Sleeper lacks."""
    frames = [f for f in (sleeper, espn) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "full_name", "team", "position", "report_status",
                                     "practice_status", "depth_pos", "depth_order", "updated", "source"])
    d = pd.concat(frames, ignore_index=True)
    d["key"] = d["gsis_id"].fillna(d["full_name"].str.lower() + "|" + d["team"].fillna(""))
    d["rank"] = d["source"].map({"Sleeper": 0, "ESPN": 1})
    d = d.sort_values("rank").drop_duplicates("key", keep="first").drop(columns=["key", "rank"])
    return d


def injury_rows(live: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Live statuses as nflverse-style injury rows for the teams/weeks in `upcoming`."""
    teams = pd.concat([
        upcoming[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
        upcoming[["season", "week", "away_team"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates()
    hurt = live[live["report_status"].notna() & live["gsis_id"].notna()]
    out = hurt.merge(teams, on="team", how="inner")
    return out[["season", "week", "team", "gsis_id", "full_name", "position", "report_status", "practice_status", "source"]]


def merge_injuries(nflverse_inj: pd.DataFrame, live_rows: pd.DataFrame) -> pd.DataFrame:
    """Live rows replace nflverse rows for the same (season, week, team); older weeks untouched."""
    if live_rows is None or live_rows.empty:
        return nflverse_inj
    keys = live_rows[["season", "week", "team"]].drop_duplicates()
    base = nflverse_inj.merge(keys.assign(_live=1), on=["season", "week", "team"], how="left")
    base = base[base["_live"].isna()].drop(columns="_live")
    return pd.concat([base, live_rows.drop(columns=["source"])], ignore_index=True)


def projected_starters(live: pd.DataFrame) -> dict[str, dict]:
    """team -> {starter gsis/name, backup gsis/name} from the depth chart, skipping QBs who are Out."""
    qbs = live[(live["depth_pos"] == "QB") & live["depth_order"].notna()].sort_values("depth_order")
    res = {}
    for team, g in qbs.groupby("team"):
        healthy = g[g["report_status"] != "Out"]
        if healthy.empty:
            continue
        s = healthy.iloc[0]
        b = healthy.iloc[1] if len(healthy) > 1 else None
        res[team] = {"starter_id": s["gsis_id"], "starter": s["full_name"], "starter_status": s["report_status"],
                     "backup_id": None if b is None else b["gsis_id"], "backup": None if b is None else b["full_name"],
                     "depth_qb1": g.iloc[0]["full_name"], "depth_qb1_status": g.iloc[0]["report_status"]}
    return res


def apply_to_schedule(games: pd.DataFrame, starters: dict, live: pd.DataFrame,
                      game_ids: set | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Swap in the projected starter for upcoming games when the listed starter is Out (or missing)."""
    g = games.copy()
    status = live.dropna(subset=["gsis_id"]).set_index("gsis_id")["report_status"].to_dict()
    log = []
    upcoming = g["home_score"].isna() & (g["game_id"].isin(game_ids) if game_ids is not None else True)
    for side in ("home", "away"):
        if f"{side}_backup_qb_id" not in g:
            g[f"{side}_backup_qb_id"] = None
    for side in ("home", "away"):
        for i in g.index[upcoming]:
            team = g.at[i, f"{side}_team"]
            proj = starters.get(team)
            if not proj:
                continue
            listed = g.at[i, f"{side}_qb_id"]
            g.at[i, f"{side}_backup_qb_id"] = proj["backup_id"]
            if pd.isna(listed) or status.get(listed) == "Out":
                if proj["starter_id"] and proj["starter_id"] != listed:
                    log.append({"game_id": g.at[i, "game_id"], "team": team, "type": "starter",
                                "text": f"{team}: {g.at[i, f'{side}_qb_name'] or 'listed QB'} is Out; "
                                        f"{proj['starter']} projected to start"})
                    g.at[i, f"{side}_qb_id"] = proj["starter_id"]
                    g.at[i, f"{side}_qb_name"] = proj["starter"]
    return g, log


def status_changes(nflverse_inj: pd.DataFrame, live_rows: pd.DataFrame, upcoming: pd.DataFrame) -> list[dict]:
    """Human-readable list of starting-QB status changes vs the nflverse report."""
    log = []
    if live_rows is None or live_rows.empty:
        return log
    qbs = live_rows[live_rows["position"] == "QB"].sort_values("week").drop_duplicates(["team", "gsis_id"])
    old = nflverse_inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last").set_index(
        ["season", "week", "team", "gsis_id"])
    for r in qbs.itertuples():
        k = (r.season, r.week, r.team, r.gsis_id)
        prev = old["report_status"].get(k) if k in old.index else None
        if prev != r.report_status:
            log.append({"team": r.team, "type": "qb_status",
                        "text": f"{r.full_name} ({r.team}): {prev or 'not listed'} → {r.report_status}"
                                f"{' (' + r.practice_status.split(' ')[0].replace('Did', 'no practice') + ')' if isinstance(r.practice_status, str) else ''}"
                                f" via {r.source}"})
    return log


def fetch_live(upcoming: pd.DataFrame, players: pd.DataFrame,
               sources: tuple = ("sleeper", "espn")) -> tuple[pd.DataFrame, dict]:
    """Fetch Sleeper + ESPN. Never raises: returns whatever succeeded plus a source report.
    Sleeper asks apps to pull the full player list sparingly, so pre-kickoff runs use ESPN only."""
    report = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sl = es = None
    report["sleeper"] = "skipped this run"
    if "sleeper" in sources:
        try:
            sl = parse_sleeper(_get_json(SLEEPER_URL))
            n_hurt = int(sl["report_status"].notna().sum())
            report["sleeper"] = f"ok ({len(sl)} rostered players, {n_hurt} with an injury status)"
        except Exception as e:
            report["sleeper"] = f"failed: {e}"
    espn_to_gsis = {}
    if "espn_id" in players:
        m = players.dropna(subset=["espn_id", "gsis_id"])
        espn_to_gsis = dict(zip(m["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True), m["gsis_id"]))
    frames, ok = [], 0
    eids = upcoming.get("espn", pd.Series(dtype=object)).dropna().astype(str).str.replace(r"\.0$", "", regex=True)
    for eid in (eids if "espn" in sources else []):
        try:
            frames.append(parse_espn_summary(_get_json(ESPN_SUMMARY.format(eid)), espn_to_gsis))
            ok += 1
        except Exception as e:
            report.setdefault("espn_errors", []).append(f"{eid}: {e}")
    if frames:
        es = pd.concat(frames, ignore_index=True)
    report["espn"] = f"ok ({ok} games)" if ok else "failed or no games"
    return combine(sl, es), report
