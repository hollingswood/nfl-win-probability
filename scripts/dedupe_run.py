"""Decide whether this Weekly predictions run should skip because another run already did this slot's work.

Fixes found in the 2026-10-01..09 validation:
  * a run whose own `predict` job was skipped no longer counts as covering a slot (it used to, so skips chained);
  * GitHub's scheduled copies often start hours late. For a `schedule` event the slot starts at its cron time
    (github.event.schedule) minus 15 minutes, not 40 minutes before now, so a copy arriving 3 h late skips when the
    outside scheduler already ran that slot, and still runs (late beats never) when nothing did.
Usage (in the workflow): python3 dedupe_run.py  with env GH_TOKEN, REPO, RUN_ID, EVENT, SCHEDULE. Prints skip=true|false.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://api.github.com"


def slot_start(event: str, schedule: str, now: datetime) -> datetime:
    """Earliest creation time of a run that counts as covering this slot."""
    if event == "schedule" and schedule:
        minute, hour = schedule.split()[:2]
        t = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
        if t > now + timedelta(minutes=5):      # e.g. '5 0 * * 1' seen at 23:59 -> yesterday's slot
            t -= timedelta(days=1)
        return t - timedelta(minutes=15)
    return now - timedelta(minutes=40)


def did_work(jobs: list[dict]) -> bool:
    """True when the run's predict job ran or may still run (queued / in progress / succeeded)."""
    p = [j for j in jobs if j.get("name") == "predict"]
    if not p:                                    # predict not created yet: the check job is still deciding
        return True
    return p[0].get("conclusion") in (None, "success")


def decide(runs: list[dict], jobs_of, run_id: int, since: datetime) -> bool:
    for r in runs:
        if r["id"] >= run_id or r.get("conclusion") not in (None, "success"):
            continue
        if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) < since:
            continue
        if did_work(jobs_of(r["id"])):
            return True
    return False


def _get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> int:
    tok, repo = os.environ["GH_TOKEN"], os.environ["REPO"]
    run_id, event, sched = int(os.environ["RUN_ID"]), os.environ.get("EVENT", ""), os.environ.get("SCHEDULE", "")
    now = datetime.now(timezone.utc)
    since = slot_start(event, sched, now)
    q = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = _get(f"{API}/repos/{repo}/actions/workflows/weekly.yml/runs?created=%3E{q}&per_page=30", tok).get("workflow_runs", [])
    skip = decide(runs, lambda rid: _get(f"{API}/repos/{repo}/actions/runs/{rid}/jobs", tok).get("jobs", []), run_id, since)
    print(f"slot starts {q}; earlier run that did the work: {skip}", file=sys.stderr)
    print(f"skip={'true' if skip else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
