"""Mistake-proofing for published picks (poka-yoke): the AI reads articles, these deterministic checks decide what is kept.

Layer 1 (before the AI): picks TABLES are rewritten as explicit rows ("Tom Fornelli picks Alabama"), with the row's market
          taken from its label (a team and a number = against the spread, SU = straight up, O/U = total), so the AI never
          has to line up columns. (2026-10-16 CBS Alabama-Tennessee table: column shift + spread row read as moneyline.)
Layer 2 (after the AI, needs the article text) -> `validate`: a pick is dropped, with the reason logged, when
          - its quote is not in the article (invented or paraphrased), or its analyst's name is not in the article;
          - the picked side is not one of the two teams (players, props);
          - it contradicts that analyst's row in a picks table, or labels a spread row as a moneyline (or the reverse);
          - the same analyst has both sides of the same game and market in the article.
          The AI's 'reason' (a short summary of why the analyst likes the pick) is kept only if most of its words are
          in the article (`ground_reason`); otherwise it's blanked, so a made-up rationale is never shown.
Layer 3 (at grading, needs the closing line; picks_score) -> `grade_check`: a stated total far from the game total is a team
          total or prop; a stated spread far from the game's spread is a different line; such picks are not graded.
Layer 4 (re-reads; picks_score) -> picks an article no longer shows on its latest read are dropped (self-correcting).
          Articles read before the current VERSION of these checks are re-read once (picks_log.version_reread_due).
"""
from __future__ import annotations

VERSION = 2   # 2: picks carry a grounded 'reason' summary. bump when a check changes: picks_log re-reads recent articles once under the new checks (layer 4)

import html
import re
from html.parser import HTMLParser

TABLE_ROW = re.compile(r'\[PICKS TABLE\] row "(?P<label>[^"]*)" \((?P<kind>[a-z ]+)\): (?P<cells>.*?) \[END ROW\]')
SPREAD_LABEL = re.compile(r"(?<![\w.])[+-]\s?\d+(\.\d+)?(?![\w.])")
TOTAL_LABEL = re.compile(r"\b(o/u|over/under|over|under|total)\b", re.I)
SU_LABEL = re.compile(r"\b(su|straight[- ]?up|moneyline|ml|winner|to win)\b", re.I)
KIND_MARKET = {"against the spread": "spread", "straight up": "moneyline", "total": "total"}


def label_kind(label: str) -> str | None:
    if TOTAL_LABEL.search(label):
        return "total"
    if SU_LABEL.search(label):
        return "straight up"
    if SPREAD_LABEL.search(label):
        return "against the spread"
    return None


class _Tables(HTMLParser):
    """Collects every <table> as rows of cell texts; an <img> inside a cell contributes its alt/title (team logos)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self._stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._stack.append([])
        elif not self._stack:
            return
        elif tag == "tr":
            self._stack[-1].append([])
        elif tag in ("td", "th") and self._stack[-1]:
            self._stack[-1][-1].append(["", ""])                 # [visible text, image alt text]
        elif tag == "img" and self._stack[-1] and self._stack[-1][-1]:
            alt = (a.get("alt") or a.get("title") or "").strip()
            if alt:
                self._stack[-1][-1][-1][1] += f" {alt} "

    def handle_endtag(self, tag):
        if tag == "table" and self._stack:
            self.tables.append(self._stack.pop())

    def handle_data(self, data):
        if self._stack and self._stack[-1] and self._stack[-1][-1]:
            self._stack[-1][-1][-1][0] += data


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()


def _dedupe_words(s: str) -> str:
    """'Alabama Alabama' (logo alt + caption) -> 'Alabama'."""
    w = s.split()
    half = len(w) // 2
    if half and len(w) % 2 == 0 and w[:half] == w[half:]:
        return " ".join(w[:half])
    return s


def table_rows(table_html: str) -> list[str]:
    """A picks table -> explicit lines, one per labeled row; [] if it doesn't look like analysts x picks."""
    p = _Tables()
    p.feed(table_html)
    out = []
    for rows in p.tables:
        # a cell's visible text wins; a logo-only cell uses its image alt text (team logos)
        rows = [[_dedupe_words(_clean(c[0]) or _clean(c[1])) for c in r] for r in rows if r]
        if len(rows) < 2:
            continue
        head = rows[0]
        names = head[1:]
        if len([n for n in names if n]) < 2 or any(len(n.split()) > 4 for n in names if n):
            continue
        for r in rows[1:]:
            label, vals = (r[0] if r else ""), r[1:]
            kind = label_kind(label)
            if not kind:
                continue
            cells = [f"{n} picks {v}" for n, v in zip(names, vals) if n and v]
            if cells:
                out.append(f'[PICKS TABLE] row "{label}" ({kind}): ' + "; ".join(cells) + " [END ROW]")
    return out


def html_to_text(raw: str, limit: int = 90000) -> str:
    """Article HTML -> text for the AI; picks tables become explicit rows (layer 1)."""
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)

    def tab(m):
        rows = table_rows(m.group(0))
        return (" \n" + "\n".join(rows) + "\n ") if rows else m.group(0)
    raw = re.sub(r"(?is)<table\b.*?</table>", tab, raw)
    t = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\s*\n\s*", "\n", t)[:limit]


def records(text: str) -> list[dict]:
    """[{label, kind, picks: {analyst: value}}] parsed back from the explicit table rows in the text."""
    out = []
    for m in TABLE_ROW.finditer(text):
        cells = {}
        for c in m.group("cells").split("; "):
            if " picks " in c:
                n, v = c.split(" picks ", 1)
                cells[n.strip()] = v.strip()
        out.append({"label": m.group("label"), "kind": m.group("kind"), "picks": cells})
    return out


# ---------------------------------------------------------------- layer 2: validate the AI's picks against the article
def _norm(s: str) -> str:
    s = str(s or "").lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.'%/+\- ]", " ", s)).strip()


def _tokens(s: str) -> set[str]:
    return {w for w in _norm(s).split() if len(w) >= 3}


def same_team(a: str, b: str) -> bool:
    """'Alabama' ~ 'Alabama Crimson Tide', 'Colts' ~ 'Indianapolis Colts'; 'Ohio State' !~ 'Michigan State';
    'Wan'Dale Robinson' !~ 'New York Giants'. One name's words must all appear in the other."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and (ta <= tb or tb <= ta)


def _name_in(name: str, text_n: str) -> bool:
    toks = [w for w in _norm(name).split() if len(w) >= 3]
    return not toks or toks[-1] in text_n          # initials-only names can't be checked


def _pick_side(p: dict) -> str:
    return _norm(p.get("pick"))


def _analyst_cell(rec: dict, analyst: str) -> str | None:
    if not analyst:
        return None
    a = _norm(analyst)
    for n, v in rec["picks"].items():
        nn = _norm(n)
        if nn == a or (nn.split() and a.split() and nn.split()[-1] == a.split()[-1]):
            return v
    return None


STOP = set("about after again against their there these those which while would could should being other since still where "
           "because think thinks likes expect expects should going point points games teams their's".split())


def ground_reason(reason, text_n: str, need: float = 0.6) -> str:
    """The AI's summary of why, kept only if >= 60% of its content words (5+ letters) appear in the article."""
    r = re.sub(r"\s+", " ", str(reason or "")).strip()
    words = [w.strip(".'%/+-") for w in _norm(r).split()]
    words = [w for w in words if len(w) >= 5 and w not in STOP]
    if not r or not words:
        return ""
    hit = sum(w in text_n or w.rstrip("s") in text_n for w in words) / len(words)
    return r[:400] if hit >= need else ""


def validate(picks: list[dict], text: str) -> tuple[list[dict], list[dict]]:
    """-> (kept, rejected); each rejected pick carries 'reject_reason'."""
    text_n = _norm(text)
    recs = records(text)
    kept, rej = [], []

    def drop(p, why):
        rej.append({**p, "reject_reason": why})

    for p in picks:
        mk, side = p.get("market"), _pick_side(p)
        teams = [p.get("away_team") or "", p.get("home_team") or ""]
        q = _norm(p.get("confidence_words"))
        if q and len(q) >= 8 and q not in text_n:
            drop(p, "quote not found in the article"); continue
        if p.get("analyst") and not _name_in(p["analyst"], text_n):
            drop(p, "analyst not named in the article"); continue
        if mk in ("spread", "moneyline") and not any(same_team(p.get("pick") or "", t) for t in teams if t):
            drop(p, "picked side is not one of the two teams"); continue
        if mk == "total" and side not in ("over", "under"):
            drop(p, "total pick is not over/under"); continue
        bad = None
        for r in recs:
            v = _analyst_cell(r, p.get("analyst") or "")
            if v is None or not any(same_team(v, t) or same_team(r["label"], t) for t in teams if t) and r["kind"] != "total":
                continue
            rm = KIND_MARKET[r["kind"]]
            if rm == mk:
                ok = (side in _norm(v)) if mk == "total" else same_team(v, p.get("pick") or "")
                if not ok:
                    bad = f"contradicts the picks table (row \"{r['label']}\": {v})"; break
            elif not any(KIND_MARKET[x["kind"]] == mk and _analyst_cell(x, p.get("analyst") or "") is not None for x in recs):
                bad = f"the table row is {r['kind']}, not {mk}"
        if bad:
            drop(p, bad); continue
        p = dict(p, reason=ground_reason(p.get("reason"), text_n))
        kept.append(p)
    # same analyst, same game and market, both sides -> drop both
    by = {}
    for p in kept:
        k = (_norm(p.get("analyst")), _norm(p.get("away_team")), _norm(p.get("home_team")), p.get("market"))
        by.setdefault(k, set()).add(_pick_side(p))
    final = []
    for p in kept:
        k = (_norm(p.get("analyst")), _norm(p.get("away_team")), _norm(p.get("home_team")), p.get("market"))
        if p.get("analyst") and len(by[k]) > 1:
            drop(p, "the article has this analyst on both sides")
        else:
            final.append(p)
    return final, rej


def offline_reasons(picks: list[dict]) -> list[str | None]:
    """Checks that need no article text, for picks already logged (also run at grading): the picked side is one of the
    two teams, and an analyst isn't on both sides of one game and market in one article."""
    def gk(p):
        return (p.get("link"), _norm(p.get("analyst")), _norm(p.get("away_team")), _norm(p.get("home_team")), p.get("market"))
    sides = {}
    for p in picks:
        if p.get("analyst") and p.get("link"):
            sides.setdefault(gk(p), set()).add(_pick_side(p))
    out = []
    for p in picks:
        mk, teams = p.get("market"), [p.get("away_team") or "", p.get("home_team") or ""]
        if mk in ("spread", "moneyline") and not any(same_team(p.get("pick") or "", t) for t in teams if t):
            out.append("picked side is not one of the two teams")
        elif mk == "total" and _pick_side(p) not in ("over", "under"):
            out.append("total pick is not over/under")
        elif p.get("analyst") and p.get("link") and len(sides.get(gk(p), ())) > 1:
            out.append("the article has this analyst on both sides")
        else:
            out.append(None)
    return out


# ---------------------------------------------------------------- layer 3: grading-time check against the closing line
TOTAL_GAP, SPREAD_GAP = 10.0, 10.0


def grade_check(market: str, stated, close_for_side) -> str | None:
    """None = fine to grade; else why not (stated number can't be this game's line)."""
    if not isinstance(stated, (int, float)) or close_for_side is None:
        return None
    if market == "total" and abs(stated - close_for_side) > TOTAL_GAP:
        return "stated total is far from the game total (team total or prop)"
    if market == "spread" and abs(stated - close_for_side) > SPREAD_GAP:
        return "stated spread doesn't match this game's line"
    return None
