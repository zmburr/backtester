"""
Playbook email: the trader's own bounce / reversal lessons, in the inbox the
night a cluster forms.

Why (2026-10-15): the 3:15 PM report on 10/14 already had the semis bounce
(SOXL 6/6, MXL 5/6). The night-before prep note costs sleep, and on 10/15 the
lessons that would have paid were all in the vault with nothing putting them
in front of him: the three-shot risk budget, "listen to my metrics" (NBIS 6/6
GO, not traded), a leg in the relative-strength name (SKHY).

Gate (user, 2026-10-15): 2+ GO on one side of the priority report. Over the
75 evening reports 3/18-10/15 that fired on 10 bounce evenings (~2-3 a month),
including 10/14.
  evening (cron 17:00 ET): today's 3:15 PM report -> email for the next session
  morning (cron 07:30 ET): today's 5:45 AM report, plus the premarket scanner's
           GO alerts on the bounce side -> only if no evening email already went
           out for that side and session (overnight news can build the cluster)
At most one email per side per session (state: data/playbook_email_state.json).

The lessons are the trader's Obsidian notes (Evernote/Consolidated Playbook/
Bounce Lessons.md and Reversal Lessons.md), read fresh at send time: `## `
sections in note order, `- ` bullets, a `## Notes` section stays out. The code
only adds tonight's names table and the R tiers from ExitMonitor. The bounce
GO level is the evening board's go_gap_trigger: the open that makes the name
5/6 with everything but price fixed.

Usage:
    python -m scripts.playbook_email --session evening
    python -m scripts.playbook_email --session morning
    python -m scripts.playbook_email --session evening --asof 2026-10-14 --dry --out x.html
    python -m scripts.playbook_email --session evening --asof 2026-10-14 --force  # resend, no state
    python -m scripts.playbook_email --replay     # fire dates over the archive, no network
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

from analyzers.bounce_scorer import BouncePretrade
from scanners.premarket_bounce_scanner import (
    ALERT_SCORE, CAP_CACHE_FILE, STATE_FILE as PM_STATE_FILE,
    _load_json, build_static, get_cap, metrics_from_price,
)
from scripts.evening_signal_log import go_gap_trigger, next_trading_day
from support import risk_source
from support.config import send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("playbook_email")

EMAIL_TO = "zmburr@gmail.com"
BOARD_URL = "https://zacharys-mac-mini.tailca90a1.ts.net:8445/"   # Setup Board (board/server.py)
MIN_GO = 2           # GO names on one side before the email goes out
MAX_ROWS = 12         # names table cap (GO first); 7/28 had 19 windows open
SIDES = ("bounce", "reversal")
SIDE_WORD = {"bounce": "BOUNCE", "reversal": "REVERSAL"}

DATA_DIR = CAP_CACHE_FILE.parent
SIGNALS_DIR = DATA_DIR / "priority_signals"
STATE_FILE = DATA_DIR / "playbook_email_state.json"

NOTE_DIR = Path("Evernote") / "Consolidated Playbook"
NOTE_NAMES = {"bounce": "Bounce Lessons.md", "reversal": "Reversal Lessons.md"}
VAULT_CANDIDATES = (
    Path.home() / "ObsidianVault",                                     # Mac mini / Air
    Path(r"C:\Users\zmbur\OneDrive\Documents\Obsidian Vault"),         # desk PC
)

# 3DGapFade classification gate (analyzers/reversal_pretrade.py): gap >= 4%.
REVERSAL_GAP_GATE = 0.04

# Leveraged products whose GO is inflated by the leverage (10/15: SOXL and
# KORU scored 5/6 on a 1-2/6 underlying). The email points at the principal.
PRINCIPALS = {
    "SOXL": "SMH, MU or SKHY", "KORU": "EWY or SKHY",
    "TQQQ": "QQQ", "TNA": "IWM", "UPRO": "SPY", "SPXL": "SPY",
    "NVDL": "NVDA", "NVDX": "NVDA", "MSTX": "MSTR", "MSTU": "MSTR",
    "AGQ": "SLV", "LABU": "XBI", "FAS": "XLF", "TSLL": "TSLA",
    "CONL": "COIN", "BITX": "IBIT", "ETHU": "ETHE", "YINN": "FXI",
}


# ---------------------------------------------------------------------------
# Reports and gate
# ---------------------------------------------------------------------------

def load_report(date: str, session: str) -> dict | None:
    path = SIGNALS_DIR / f"{date}_{session}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning(f"unreadable report {path.name}: {e}")
        return None


def side_signals(report: dict, side: str) -> list[dict]:
    return [s for s in report.get("signals", []) if s.get("bucket") == side]


def go_tickers(report: dict, side: str) -> list[str]:
    return [s["ticker"] for s in side_signals(report, side) if s.get("recommendation") == "GO"]


def premarket_go(date: str) -> dict:
    """Bounce names the premarket scanner alerted as GO today: {ticker: score}."""
    state = _load_json(PM_STATE_FILE)
    if state.get("date") != date:
        return {}
    return {t: n for t, n in (state.get("alerted") or {}).items() if n >= ALERT_SCORE}


def load_state() -> dict:
    return _load_json(STATE_FILE)


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=1))


def already_sent(state: dict, side: str, target: str) -> bool:
    return (state.get(side) or {}).get("target") == target


# ---------------------------------------------------------------------------
# Lessons note
# ---------------------------------------------------------------------------

def find_note(side: str) -> Path | None:
    roots = [Path(os.environ["OBSIDIAN_VAULT"])] if os.environ.get("OBSIDIAN_VAULT") else []
    for root in roots + list(VAULT_CANDIDATES):
        p = root / NOTE_DIR / NOTE_NAMES[side]
        if p.exists():
            return p
    return None


_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


def parse_lessons(text: str) -> list[tuple[str, list[str]]]:
    """`## ` sections in note order with their bullets. Text before the first
    section is the note's own preamble; `## Notes` is the trader's scratch space.
    An indented non-bullet line continues the bullet above it."""
    sections: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
            continue
        if not sections:
            continue
        bullets = sections[-1][1]
        stripped = line.strip()
        if _BULLET.match(stripped):
            bullets.append(_BULLET.sub("", stripped))
        elif stripped and bullets and raw[:1] in (" ", "\t"):
            bullets[-1] += " " + stripped
    return [(h, b) for h, b in sections if b and h.lower() != "notes"]


def md_inline(text: str) -> str:
    """Escape, then the little markdown the notes use: **bold**, *italic*,
    `code` and [[wikilinks]]."""
    s = html.escape(text, quote=False)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return s


# ---------------------------------------------------------------------------
# Names table (network: one Polygon daily fetch per name)
# ---------------------------------------------------------------------------

def _ordered(signals: list[dict]) -> list[dict]:
    rank = {"GO": 0, "CAUTION": 1}
    sigs = [s for s in signals if s.get("recommendation") in rank]
    sigs.sort(key=lambda s: (rank[s["recommendation"]], -(s.get("intensity") or 0)))
    return sigs[:MAX_ROWS]


def bounce_rows(signals: list[dict], target: str) -> list[dict]:
    """Statics as-of the target session, so tonight's close is the prior close
    (the same view tomorrow's 4:15 AM premarket scan starts from)."""
    checker = BouncePretrade()
    caps = _load_json(CAP_CACHE_FILE)
    rows = []
    for s in _ordered(signals):
        t = s["ticker"]
        cap = s.get("cap") or get_cap(t, caps)
        row = {"ticker": t, "rec": s.get("recommendation"), "score": s.get("score"), "cap": cap,
               "p_bounce5": (s.get("odds") or {}).get("p_bounce5"),
               "source": s.get("source"), "principal": PRINCIPALS.get(t)}
        try:
            static = build_static(t, target)
        except Exception as e:
            logger.warning(f"{t}: daily bars failed: {e}")
            static = None
        if static:
            close = float(static["prior_close"])
            m = metrics_from_price(static, close)
            go = go_gap_trigger(t, static, cap, checker)
            row.update(close=close, down_days=static["consecutive_down_days"],
                       off_30d=m.get("pct_off_30d_high"), atr=static["atr"],
                       atr_pct=static["atr"] / close if close else None,
                       go_gap=go, go_price=close * (1 + go) if go is not None else None,
                       mid_bb=static.get("mid_bb"))
        rows.append(row)
    return rows


def reversal_rows(signals: list[dict], target: str) -> list[dict]:
    rows = []
    for s in _ordered(signals):
        t = s["ticker"]
        m = s.get("metrics") or {}
        row = {"ticker": t, "rec": s.get("recommendation"), "score": s.get("score"),
               "cap": s.get("cap"), "setup": s.get("setup_type") or "",
               "above_9ema": m.get("pct_from_9ema"), "chg_3d": m.get("pct_change_3"),
               "principal": PRINCIPALS.get(t)}
        try:
            static = build_static(t, target)
        except Exception as e:
            logger.warning(f"{t}: daily bars failed: {e}")
            static = None
        if static:
            close = float(static["prior_close"])
            row.update(close=close, atr=static["atr"],
                       atr_pct=static["atr"] / close if close else None,
                       gap_price=close * (1 + REVERSAL_GAP_GATE))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pct(x, signed=True) -> str:
    if x is None:
        return "–"
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"


def _px(x) -> str:
    if x is None:
        return "–"
    return f"{x:,.2f}"


def _go_cell(row: dict) -> str:
    if "close" not in row:
        return "–"
    go = row.get("go_gap")
    if go is None:
        return "not within −20%"
    if go >= 0:
        return "GO at a flat open"
    return f"≤ {_px(row['go_price'])} ({_pct(go)})"


def _day_word(d: datetime.date) -> str:
    return f"{d:%a} {d.month}/{d.day}"


def tiers_line() -> str | None:
    """'1R = $3,000 · A 15R = 3 shots of 5R …' from ExitMonitor, or None."""
    one_r, tiers = risk_source.one_r_dollars(), risk_source.tier_multiples()
    if not one_r or not tiers:
        return None
    parts = []
    for label in ("A", "B", "C"):
        r = tiers.get(label)
        if r:
            shot = r / 3
            parts.append(f"{label} {r:g}R = 3 shots of {shot:.1f}R (${shot * one_r / 1000:,.1f}k)")
    return f"1R = ${one_r:,.0f}. " + " · ".join(parts) if parts else None


GO_COLOR, CAUTION_COLOR = "#1a7f37", "#9a6700"


def _name_block(row: dict, line1: list[str], line2: list[str]) -> str:
    """One name = two short wrapping lines (a table runs off a phone screen)."""
    rec = row.get("rec") or ""
    color = GO_COLOR if rec == "GO" else CAUTION_COLOR
    head = (f'<b>{html.escape(row["ticker"])}</b> '
            f'<b style="color:{color}">{html.escape(str(row.get("score") or ""))} {rec}</b>')
    first = " · ".join([head] + [x for x in line1 if x])
    second = " · ".join(x for x in line2 if x)
    return (f'<div style="margin:5px 0;padding:4px 8px;border-left:3px solid {color};'
            f'background:#f6f8fa">{first}'
            + (f'<br><span style="font-size:13px">{second}</span>' if second else "")
            + "</div>")


def names_html(side: str, rows: list[dict]) -> str:
    blocks = []
    for r in rows:
        if side == "bounce":
            line1 = [r.get("cap"), f"close {_px(r['close'])}" if "close" in r else None,
                     f"down {r['down_days']}d" if r.get("down_days") is not None else None,
                     f"{_pct(r['off_30d'])} off 30d high" if r.get("off_30d") is not None else None]
            line2 = [f"GO level {_go_cell(r)}" if "close" in r else "no daily bars",
                     f"ATR {_pct(r['atr_pct'], signed=False)}" if r.get("atr_pct") else None,
                     f"mid-BB {_px(r['mid_bb'])}" if r.get("mid_bb") else None,
                     f"{_pct(r['p_bounce5'], signed=False)} odds of a 5%+ bounce"
                     if r.get("p_bounce5") is not None else None]
        else:
            line1 = [html.escape(r["setup"]) if r.get("setup") else None, r.get("cap"),
                     f"close {_px(r['close'])}" if "close" in r else None,
                     f"{_pct(r['above_9ema'])} above 9EMA" if r.get("above_9ema") is not None else None,
                     f"{_pct(r['chg_3d'])} in 3 days" if r.get("chg_3d") is not None else None]
            line2 = [f"+4% gap at {_px(r['gap_price'])}" if r.get("gap_price") else "no daily bars",
                     f"ATR {_pct(r['atr_pct'], signed=False)}" if r.get("atr_pct") else None]
        if r.get("principal"):
            line2.append(f"leveraged: check the principal ({html.escape(r['principal'])})")
        if r.get("source") == "premarket scanner":
            line2.append("from the premarket scanner, not the report")
        blocks.append(_name_block(r, line1, line2))
    note = ("GO level = the open that makes it 5/6, everything else fixed. Mid-BB = swing exit."
            if side == "bounce" else
            "+4% gap = the 3DGapFade gap gate off tonight's close (cap-blind; ETFs rarely get it).")
    return "".join(blocks) + f'<div style="font-size:12px;color:#666">{note}</div>'


def render(side: str, session: str, report: dict, target: datetime.date, rows: list[dict],
           sections: list[tuple[str, list[str]]] | None, note_path: Path | None,
           tiers: str | None) -> tuple[str, str]:
    go = [r["ticker"] for r in rows if r.get("rec") == "GO"]
    n_caution = sum(1 for r in rows if r.get("rec") == "CAUTION")
    when = "tonight" if session == "evening" else "this morning"
    if session == "evening":
        subject = f"{SIDE_WORD[side]} SETUP for {_day_word(target)}: {' '.join(go)} ({len(go)} GO)"
    else:
        subject = f"{SIDE_WORD[side]} SETUP today {target.month}/{target.day}: {' '.join(go)} ({len(go)} GO)"

    stamp = (report.get("generated_at") or "")[11:16]
    report_name = f"{report.get('session', session)} report{f' ({stamp})' if stamp else ''}"
    lead = f"{len(go)} GO / {n_caution} CAUTION on the {side} side in the {report_name}"
    cluster = report.get("cluster_signal") or {}
    if side == "bounce" and cluster.get("bounce_open") is not None:
        lead += f" · {cluster['bounce_open']} bounce windows open"
        if cluster.get("level"):
            lead += f" ({cluster['level']})"
    lead += f". Your playbook for {_day_word(target)}, sent {when}."

    parts = ['<meta charset="utf-8">',
             f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
             f'font-size:14px;line-height:1.45;color:#1f2328;max-width:760px">',
             f'<div style="font-size:15px;margin-bottom:6px">{html.escape(lead)}</div>',
             f'<div style="margin:0 0 8px"><a href="{BOARD_URL}" style="font-weight:600">'
             f'Open the Setup Board</a> <span style="color:#666;font-size:13px">'
             f'(live scores, scenarios and charts)</span></div>',
             names_html(side, rows)]

    if sections is None:
        parts.append(f'<p style="color:#cf222e"><b>Lessons note not found</b> '
                     f'({html.escape(str(NOTE_DIR / NOTE_NAMES[side]))} in the vault). '
                     f'Sending the table only.</p>')
        sections = [("Risk budget", [])] if tiers else []
    elif tiers and not any(h.lower().startswith("risk budget") for h, _ in sections):
        sections = [("Risk budget", [])] + sections

    for heading, bullets in sections:
        items = [md_inline(b) for b in bullets]
        if tiers and heading.lower().startswith("risk budget"):
            items.insert(0, f"<i>{html.escape(tiers)}</i>")
        lis = "".join(f'<li style="margin:2px 0">{i}</li>' for i in items)
        parts.append(f'<h3 style="font-size:15px;margin:14px 0 4px">{md_inline(heading)}</h3>'
                     f'<ul style="margin:0;padding-left:20px">{lis}</ul>')

    src = (f"Obsidian › Consolidated Playbook › {NOTE_NAMES[side][:-3]}" if note_path
           else "the lessons note (missing)")
    parts.append(f'<p style="font-size:12px;color:#666;margin-top:16px">Lessons from {src}. '
                 f"Edit them there; the next email picks them up. Source: "
                 f"{html.escape(str(report.get('date', '')))} {report_name}.</p></div>")
    return subject, "".join(parts)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(session: str, asof: datetime.date, dry: bool = False, force: bool = False,
        sides=SIDES, out: Path | None = None) -> list[str]:
    """Send (or render, with dry) one email per side that passes the gate.
    Returns the sides that went out. force: ignore the gate and the state, and
    don't record the send (resending a past evening as a sample)."""
    date = asof.isoformat()
    report = load_report(date, session)
    if report is None:
        logger.info(f"no {session} report for {date}; nothing to do")
        return []
    target = next_trading_day(asof) if session == "evening" else asof
    state = load_state()
    sent = []
    for side in sides:
        gos = go_tickers(report, side)
        signals = side_signals(report, side)
        if session == "morning" and side == "bounce":
            for t, n in premarket_go(date).items():
                if t not in gos:
                    gos.append(t)
                    signals = [s for s in signals if s.get("ticker") != t]
                    signals.append({"ticker": t, "bucket": "bounce", "recommendation": "GO",
                                    "score": f"{n}/6", "source": "premarket scanner"})
        if len(gos) < MIN_GO and not force:
            logger.info(f"{side}: {len(gos)} GO ({' '.join(gos) or '-'}), below {MIN_GO}; quiet")
            continue
        if already_sent(state, side, target.isoformat()) and not force:
            logger.info(f"{side}: playbook for {target} already sent "
                        f"({state[side].get('session')}); quiet")
            continue

        rows = (bounce_rows if side == "bounce" else reversal_rows)(signals, target.isoformat())
        note = find_note(side)
        sections = parse_lessons(note.read_text(encoding="utf-8")) if note else None
        if note is None:
            logger.warning(f"{side}: lessons note not found under {VAULT_CANDIDATES}")
        subject, body = render(side, session, report, target, rows, sections, note, tiers_line())

        if dry:
            path = (out.with_name(f"{out.stem}_{side}{out.suffix}") if out
                    else DATA_DIR / f"playbook_email_{side}_preview.html")
            path.write_text(body, encoding="utf-8")
            logger.info(f"--dry: {subject!r} -> {path}")
            sent.append(side)
            continue

        send_email(EMAIL_TO, subject, body, is_html=True)
        logger.info(f"sent: {subject}")
        sent.append(side)
        if not force:
            state[side] = {"target": target.isoformat(), "session": session,
                           "sent_at": datetime.datetime.now().isoformat(timespec="seconds")}
            save_state(state)
    return sent


def replay() -> dict:
    """Which evenings and mornings in the archive would have fired, per side.
    Gate only (report files), no network; the premarket scanner's alerts are not
    archived, so morning catch-ups count the 5:45 report alone."""
    fired = {side: [] for side in SIDES}
    evening_targets = {side: set() for side in SIDES}
    files = sorted(SIGNALS_DIR.glob("*_evening.json")) + sorted(SIGNALS_DIR.glob("*_morning.json"))
    n_evenings = 0
    for path in files:
        date_str, session = path.stem.split("_", 1)
        if session not in ("evening", "morning"):
            continue
        report = json.loads(path.read_text())
        day = datetime.date.fromisoformat(date_str)
        n_evenings += session == "evening"
        for side in SIDES:
            gos = go_tickers(report, side)
            if len(gos) < MIN_GO:
                continue
            if session == "evening":
                evening_targets[side].add(next_trading_day(day))
            elif day in evening_targets[side]:
                continue
            fired[side].append((date_str, session, gos))
    print(f"{n_evenings} evening reports in {SIGNALS_DIR}")
    for side in SIDES:
        rows = sorted(fired[side])
        by = Counter(s for _, s, _ in rows)
        print(f"\n{side}: {by['evening']} evening + {by['morning']} morning catch-up")
        for date_str, session, gos in rows:
            print(f"  {date_str} {session:8} {len(gos)} GO  {' '.join(gos)}")
    return fired


def _today_et() -> datetime.date:
    import pandas as pd
    return pd.Timestamp.now(tz="US/Eastern").date()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", choices=["evening", "morning"])
    ap.add_argument("--asof", help="report date YYYY-MM-DD (default: today ET)")
    ap.add_argument("--side", choices=list(SIDES) + ["both"], default="both")
    ap.add_argument("--dry", action="store_true", help="render HTML, send nothing, keep state")
    ap.add_argument("--out", type=Path, help="with --dry: preview path (side is appended)")
    ap.add_argument("--force", action="store_true", help="ignore gate and state; don't record")
    ap.add_argument("--replay", action="store_true", help="archive fire dates, no network")
    args = ap.parse_args()
    if args.replay:
        replay()
        return
    if not args.session:
        ap.error("--session is required (or --replay)")
    asof = datetime.date.fromisoformat(args.asof) if args.asof else _today_et()
    sides = SIDES if args.side == "both" else (args.side,)
    run(args.session, asof, dry=args.dry, force=args.force, sides=sides, out=args.out)


if __name__ == "__main__":
    main()
