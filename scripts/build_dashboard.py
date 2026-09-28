"""Render output/*.json into a static dashboard.

Writes:
  site/index.html      full HTML document (GitHub Pages)
  site/artifact.html   body-only version (for publishing as a Claude artifact)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "scripts" / "dashboard_template.html").read_text()
payload = {
    "predictions": json.loads((ROOT / "output" / "predictions.json").read_text()),
    "backtest": json.loads((ROOT / "output" / "backtest.json").read_text()),
}
vv = ROOT / "output" / "vs_vegas.json"
payload["vs_vegas"] = json.loads(vv.read_text()) if vv.exists() else None
body = tpl.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
site = ROOT / "site"
site.mkdir(exist_ok=True)
(site / "artifact.html").write_text(body)
(site / "index.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'
    + body + "</body></html>")
print("wrote", site / "index.html")
