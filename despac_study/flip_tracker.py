#!/usr/bin/env python3
"""Live de-SPAC ticker-flip tracker.

Single-shot, cron-friendly, fully standalone (stdlib + requests only).
Scans EDGAR for SPACs that just passed their merger vote (8-K Item 5.07) or
closed the deal (Item 2.01), scores each against the flip-trade checklist
from the despac study, and emails when a new candidate appears:

  +2.0  redemptions >= 85% of public shares  (microfloat - the #1 signal)
  +1.0  last close <= $10.50   (+0.5 more if <= $9.90: post-deadline flush)
  +1.0  hot theme (space/defense/AI/nuclear/fusion/quantum/crypto/...)
  +1.0  deal CLOSED (flip imminent, usually next morning)   [+0.5 if voted]

Usage:
  python3 flip_tracker.py            # normal cron run (emails on new finds)
  python3 flip_tracker.py --dry-run  # print only, no email, no state update

Env (or .env next to this file): POLYGON_API_KEY, GMAIL_PASSWORD,
TRACKER_EMAIL (default zmburr@gmail.com).

Study reference: backtester/despac_study/ - 90%+ redemption deals show median
10-day max runup +39.5% vs +15.7% baseline; sell into the flip-day pop.
"""

import argparse
import json
import logging
import os
import re
import smtplib
import sys
import time
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "tracker_state.json"
LOG_FILE = HERE / "flip_tracker.log"

EDGAR_UA = "zmburr despac flip tracker zmburr@gmail.com"
LOOKBACK_DAYS = 12
EMAIL_TO = os.getenv("TRACKER_EMAIL", "zmburr@gmail.com")
MIN_SCORE_TO_ALERT = 2.0

HOT_THEMES = {
    "fusion":  ["fusion energy", "nuclear fusion"],
    "quantum": ["quantum"],
    "nuclear": ["nuclear", "uranium", "small modular reactor", " smr "],
    "space":   ["space", "satellite", "orbital", "launch vehicle", "lunar", "rocket"],
    "defense": ["defense", "military", "drone", "hypersonic"],
    "ai":      ["artificial intelligence", " ai ", "machine learning", "robotics", "autonomous"],
    "crypto":  ["bitcoin", "crypto", "blockchain", "digital asset", "stablecoin"],
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("flip_tracker")


def load_env():
    envf = HERE / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------- EDGAR
_sess = requests.Session()
_sess.headers.update({"User-Agent": EDGAR_UA})
_last = [0.0]


def _eget(url, params=None):
    wait = 0.25 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        _last[0] = time.time()
        try:
            r = _sess.get(url, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return r
        except requests.RequestException as e:
            log.warning("EDGAR GET failed (%s): %s", url, e)
            time.sleep(2 * (attempt + 1))
    return None


def fts_recent_spac_8ks():
    """8-Ks from the last LOOKBACK_DAYS mentioning 'trust account' whose item
    list includes a merger vote (5.07) or completion (2.01)."""
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = date.today().isoformat()
    hits, frm = [], 0
    while True:
        params = {"q": '"trust account"', "forms": "8-K", "dateRange": "custom",
                  "startdt": start, "enddt": end}
        if frm:
            params["from"] = frm
        r = _eget("https://efts.sec.gov/LATEST/search-index", params)
        if not r or r.status_code != 200:
            break
        data = r.json()
        page = data.get("hits", {}).get("hits", [])
        if not page:
            break
        hits.extend(page)
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        frm += 10
        if frm >= min(total, 2000):
            break
    out = []
    for h in hits:
        s = h.get("_source", {})
        items = s.get("items") or []
        stage = "closed" if "2.01" in items else ("voted" if "5.07" in items else None)
        if not stage:
            continue
        doc_id = h.get("_id", "")
        doc = doc_id.split(":", 1)[1] if ":" in doc_id else ""
        out.append({
            "cik": int((s.get("ciks") or ["0"])[0]),
            "adsh": s.get("adsh", ""),
            "doc": doc,
            "file_date": s.get("file_date", ""),
            "display": (s.get("display_names") or [""])[0],
            "stage": stage,
            "items": items,
        })
    return out


def get_submissions(cik):
    r = _eget(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    return r.json() if r and r.status_code == 200 else {}


def get_filing_text(cik, adsh, doc):
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{doc}"
    r = _eget(url)
    if not r or r.status_code != 200:
        return ""
    txt = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", r.text)
    txt = re.sub(r"<[^>]+>", " ", txt)
    import html as html_lib
    return re.sub(r"\s+", " ", html_lib.unescape(txt))


# ------------------------------------------------- filing interpretation
_NUM = r"([\d,]{6,})"
RE_REDEEMED = [
    re.compile(r"holders of (?:an aggregate of )?" + _NUM + r"[^.]{0,220}?redee?m", re.I),
    re.compile(r"exercised (?:their|its) (?:right|rights) to redeem (?:an aggregate of )?" + _NUM, re.I),
    re.compile(r"elected to redeem (?:an aggregate of )?" + _NUM, re.I),
    re.compile(r"redemption of (?:an aggregate of )?" + _NUM + r"[^.]{0,60}(?:shares|stock)", re.I),
    re.compile(_NUM + r"[^.]{0,120}?(?:were|have been|had been|was)\s+(?:validly\s+)?(?:submitted|tendered)?\s*(?:for\s+)?redee?m", re.I),
]
RE_PCT = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%[^.]{0,140}?(?:public shares|class a|ordinary shares)|redee?m[^.]{0,160}?(\d{1,2}(?:\.\d+)?)\s*%", re.I)
RE_APPROVED = re.compile(r"(business combination|merger)[^.]{0,120}?(?:was|were|been)?\s*approv", re.I)


def parse_filing(text):
    out = {"redeemed_shares": None, "redemption_pct_stated": None, "approved": False}
    if not text:
        return out
    out["approved"] = bool(RE_APPROVED.search(text))
    for sent in re.split(r"(?<=[.;])\s+", text):
        if "redee" not in sent.lower() and "redemption" not in sent.lower():
            continue
        for rx in RE_REDEEMED:
            m = rx.search(sent)
            if m:
                try:
                    v = int(m.group(1).replace(",", ""))
                    if v < 2_000_000_000 and (not out["redeemed_shares"] or v > out["redeemed_shares"]):
                        out["redeemed_shares"] = v
                except ValueError:
                    pass
        for m in RE_PCT.finditer(sent):
            v = next((g for g in m.groups() if g), None)
            if v and 0 < float(v) <= 100:
                cur = out["redemption_pct_stated"]
                if cur is None or float(v) > cur:
                    out["redemption_pct_stated"] = float(v)
    return out


def tag_theme(text_blob):
    blob = " " + text_blob.lower() + " "
    for theme, kws in HOT_THEMES.items():
        if any(k in blob for k in kws):
            return theme
    return None


# ---------------------------------------------------------------- Polygon
POLY = "https://api.polygon.io"


def poly_get(path, **params):
    params["apiKey"] = os.environ["POLYGON_API_KEY"]
    try:
        r = requests.get(POLY + path, params=params, timeout=20)
        return r.json() if r.status_code == 200 else {}
    except requests.RequestException:
        return {}


def price_context(ticker):
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=45)).isoformat()
    d = poly_get(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                 adjusted="false", limit=60)
    bars = d.get("results") or []
    if not bars:
        return {}
    closes = [b["c"] for b in bars]
    first_ts = datetime.fromtimestamp(bars[0]["t"] / 1000)
    last_ts = datetime.fromtimestamp(bars[-1]["t"] / 1000)
    return {
        "last_close": closes[-1],
        "max_recent": max(closes),
        "first_bar_date": first_ts.strftime("%Y-%m-%d"),
        "last_bar_date": last_ts.strftime("%Y-%m-%d"),
        "still_trading": (date.today() - last_ts.date()).days <= 4,
        "avg_vol": sum(b["v"] for b in bars[-5:]) / max(1, len(bars[-5:])),
    }


def classify_flip_status(c):
    """Where in the flip lifecycle is this deal?

    EDGAR's ticker for the CIK is the CURRENT symbol - which is the old SPAC
    shell before the flip, but already the NEW symbol after it. Disambiguate
    with the symbol's first traded bar:
      - symbol older than the probe window + deal closed  -> either an
        already-public acquirer (Bed Bath & Beyond case) or a shell whose
        flip hasn't printed yet; shells trade near trust, so use price
      - symbol first printed within the window             -> flip happened
        on that first bar date (SECZ case)
    """
    fb = c.get("first_bar_date")
    if not fb:
        return "unknown"
    # 30-day freshness margin: illiquid shells can skip the first days of the
    # 45-day probe window without being fresh listings
    fresh = fb > (date.today() - timedelta(days=30)).isoformat()
    if fresh:
        if fb >= date.today().isoformat():
            return "flip_today"
        return "flipped_already"
    # pre-flip candidate must actually BE a SPAC shell, not an already-public
    # acquirer that mentioned a trust account (Bed Bath & Beyond case)
    is_shell = (str(c.get("sic", "")) == "6770"
                or re.search(r"acquisition|spac|blank check", c.get("name", ""), re.I))
    if not is_shell:
        return "not_a_shell"
    lc = c.get("last_close") or 0
    if not (3 <= lc <= 60):
        return "not_a_shell"
    return "buy_window"


def shares_outstanding(ticker, asof):
    d = poly_get(f"/v3/reference/tickers/{ticker}", date=asof)
    return (d.get("results") or {}).get("share_class_shares_outstanding")


# ---------------------------------------------------------------- scoring
def score_candidate(c):
    pts, why = 0.0, []
    red = c.get("redemption_pct")
    if red is not None and red >= 85:
        pts += 2.0
        why.append(f"redemptions {red:.0f}% (microfloat)")
    elif red is not None and red >= 60:
        pts += 1.0
        why.append(f"redemptions {red:.0f}%")
    lc = c.get("last_close")
    if lc is not None and lc <= 10.5:
        pts += 1.0
        if lc <= 9.9:
            pts += 0.5
            why.append(f"flush entry ${lc:.2f} (below trust)")
        else:
            why.append(f"entry ${lc:.2f} near trust")
    if c.get("theme"):
        pts += 1.0
        why.append(f"hot theme: {c['theme']}")
    if c.get("flip_status") == "flip_today":
        pts += 1.5
        why.append("FLIPPED THIS MORNING - pop window open")
    elif c["stage"] == "closed":
        pts += 1.0
        why.append("DEAL CLOSED - flip imminent")
    elif c.get("approved"):
        pts += 0.5
        why.append("vote passed")
    c["score"] = round(pts, 1)
    c["why"] = "; ".join(why)
    return c


# ---------------------------------------------------------------- email
def send_email(subject, html):
    pw = os.environ.get("GMAIL_PASSWORD")
    if not pw:
        log.error("GMAIL_PASSWORD not set - cannot email")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_TO
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls()
            s.login(EMAIL_TO, pw)
            s.send_message(msg)
        return True
    except Exception as e:
        log.error("email failed: %s", e)
        return False


def build_email(new, watch):
    rows = []
    for c in sorted(new + watch, key=lambda x: -x["score"]):
        tag = "NEW" if c in new else "watch"
        red = f"{c['redemption_pct']:.0f}%" if c.get("redemption_pct") is not None else "?"
        lc = f"${c['last_close']:.2f}" if c.get("last_close") else "?"
        rows.append(
            f"<tr><td><b>{c.get('ticker','?')}</b></td><td>{c['name'][:38]}</td>"
            f"<td>{c['stage']}</td><td>{c['file_date']}</td><td>{red}</td>"
            f"<td>{lc}</td><td>{c.get('theme') or ''}</td>"
            f"<td><b>{c['score']}</b></td><td>{tag}</td></tr>")
    return f"""
    <h3>De-SPAC flip tracker</h3>
    <p>Checklist: redemptions&ge;85% (+2) | entry &le;$10.50 (+1, +0.5 below $9.90)
    | hot theme (+1) | closed (+1) / voted (+0.5). Playbook: buy last close under
    old ticker, sell into flip-morning pop.</p>
    <table border=1 cellpadding=4 style="border-collapse:collapse;font-size:13px">
    <tr><th>Ticker</th><th>Company</th><th>Stage</th><th>8-K date</th>
    <th>Redeemed</th><th>Last close</th><th>Theme</th><th>Score</th><th></th></tr>
    {''.join(rows)}</table>
    <p style="color:#888;font-size:11px">backtester/despac_study - study medians:
    90%+ redemption deals run +39.5% (10d max) vs +15.7% baseline.</p>"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-email", action="store_true")
    args = ap.parse_args()
    load_env()
    if "POLYGON_API_KEY" not in os.environ:
        log.error("POLYGON_API_KEY not set")
        sys.exit(1)

    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    seen = set(state.get("seen", []))

    filings = fts_recent_spac_8ks()
    log.info("found %d vote/close 8-Ks in last %dd", len(filings), LOOKBACK_DAYS)

    # dedupe: keep the most advanced stage per CIK
    by_cik = {}
    for f in filings:
        cur = by_cik.get(f["cik"])
        if cur is None or (f["stage"] == "closed" and cur["stage"] != "closed") \
                or (f["stage"] == cur["stage"] and f["file_date"] > cur["file_date"]):
            by_cik[f["cik"]] = f

    candidates = []
    for cik, f in by_cik.items():
        subs = get_submissions(cik)
        tickers = [t for t in (subs.get("tickers") or []) if t and "." not in t and len(t) <= 5]
        name = subs.get("name", f["display"])
        ticker = min(tickers, key=len) if tickers else None
        text = get_filing_text(cik, f["adsh"], f["doc"]) if f["doc"] else ""
        parsed = parse_filing(text)
        c = {"cik": cik, "name": name, "ticker": ticker, "sic": subs.get("sic", ""),
             **f, **parsed}
        # redemption %: stated, else count / pre-vote shares outstanding
        c["redemption_pct"] = parsed["redemption_pct_stated"]
        if c["redemption_pct"] is None and parsed["redeemed_shares"] and ticker:
            asof = (datetime.strptime(f["file_date"], "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
            so = shares_outstanding(ticker, asof)
            if so:
                pct = 100.0 * parsed["redeemed_shares"] / so
                if 0 <= pct <= 100:
                    c["redemption_pct"] = round(pct, 1)
        c["theme"] = tag_theme(name + " " + text[:30000])
        if ticker:
            c.update(price_context(ticker))
        c["flip_status"] = classify_flip_status(c)
        if c["flip_status"] == "not_a_shell":
            log.info("  skip %-6s %s (already-public company, not a shell)",
                     str(ticker), name[:40])
            continue
        if c["flip_status"] == "flipped_already":
            c["stage"] = f"flipped {c.get('first_bar_date', '?')}"
        elif c["flip_status"] == "flip_today":
            c["stage"] = "FLIP TODAY"
        score_candidate(c)
        candidates.append(c)
        log.info("  %-18s %-6s %s score=%.1f %s", c["stage"], str(ticker),
                 name[:34], c["score"], c["why"])

    alertable = [c for c in candidates if c["score"] >= MIN_SCORE_TO_ALERT and c.get("ticker")]
    # email triggers: live buy windows and same-day flips; stale flips are
    # context only
    live = [c for c in alertable if c["flip_status"] in ("buy_window", "flip_today", "unknown")]
    new = [c for c in live if f"{c['cik']}:{c['stage']}" not in seen]
    watch = [c for c in alertable if c not in new]

    if args.dry_run:
        print(f"\nDRY RUN: {len(candidates)} candidates, {len(alertable)} alertable, {len(new)} new")
        return

    if new or args.force_email:
        prime = [c for c in new if c["score"] >= 3.5]
        subj = (f"FLIP TRACKER: {len(prime)} prime / {len(new)} new de-SPAC flips"
                if new else "FLIP TRACKER: status")
        if send_email(subj, build_email(new, watch)):
            log.info("emailed: %d new (%d prime)", len(new), len(prime))
            seen.update(f"{c['cik']}:{c['stage']}" for c in new)
    else:
        log.info("no new alertable candidates (%d already seen)", len(watch))

    state["seen"] = sorted(seen)[-500:]
    state["last_run"] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
