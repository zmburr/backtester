"""Playbook email: gate, once-per-session state, morning catch-up, the lessons
note parser and the rendered email. No network: the daily-bar statics, the
GO-level sweep, the risk bridge and the mail call are all stubbed."""

import datetime
import json
from unittest.mock import Mock

import pytest

from scripts import playbook_email as pe

NOTE = """# Bounce Lessons

Preamble the email ignores.

## Opens (A / B / C)
- **A: gaps through the GO level**. Buy the first 2-min break.
- B: opens above it.
  continues on the next line

## Risk budget
- Three shots per idea.

## Notes
- scratch the email must not show

## Empty section

## Head
1. Trust the tools <not html>.
"""


def _sig(ticker, rec, score="5/6", bucket="bounce", **kw):
    return {"ticker": ticker, "bucket": bucket, "recommendation": rec, "score": score,
            "cap": kw.pop("cap", "ETF"), "intensity": kw.pop("intensity", 40), **kw}


def _report(path, signals, session="evening", cluster=None):
    path.write_text(json.dumps({
        "date": path.stem.split("_")[0], "session": session,
        "generated_at": path.stem.split("_")[0] + "T15:15:41",
        "signals": signals, "cluster_signal": cluster or {"level": None, "bounce_open": len(signals)},
    }))


STATIC = {"prior_close": 100.0, "consecutive_down_days": 4, "atr": 8.0, "mid_bb": 130.0,
          "sma200": None, "close_30_ago": None, "close_3_ago": None, "prior_day_range": 9.0,
          "high_30d": 150.0, "high_52wk": 200.0, "selloff_first_open": 140.0}


@pytest.fixture
def env(tmp_path, monkeypatch):
    sig_dir = tmp_path / "priority_signals"
    sig_dir.mkdir()
    vault = tmp_path / "vault"
    (vault / pe.NOTE_DIR).mkdir(parents=True)
    (vault / pe.NOTE_DIR / "Bounce Lessons.md").write_text(NOTE, encoding="utf-8")
    monkeypatch.setattr(pe, "SIGNALS_DIR", sig_dir)
    monkeypatch.setattr(pe, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(pe, "PM_STATE_FILE", tmp_path / "pm_state.json")
    monkeypatch.setattr(pe, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pe, "VAULT_CANDIDATES", (vault,))
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    monkeypatch.setattr(pe, "next_trading_day", lambda d: d + datetime.timedelta(days=1))
    monkeypatch.setattr(pe, "build_static", lambda t, date: dict(STATIC))
    monkeypatch.setattr(pe, "go_gap_trigger", lambda t, s, cap, checker: -0.05)
    monkeypatch.setattr(pe, "BouncePretrade", Mock)
    monkeypatch.setattr(pe.risk_source, "one_r_dollars", lambda: 3000.0)
    monkeypatch.setattr(pe.risk_source, "tier_multiples",
                        lambda: {"A": 15.0, "B": 4.0, "C": 1.0, "D": 0.2})
    mail = Mock()
    monkeypatch.setattr(pe, "send_email", mail)
    return {"dir": sig_dir, "mail": mail, "tmp": tmp_path}


DAY = datetime.date(2026, 10, 14)


# ---------------------------------------------------------------- gate + state

def test_two_go_on_one_side_sends_once_for_the_next_session(env):
    _report(env["dir"] / "2026-10-14_evening.json",
            [_sig("SOXL", "GO", "6/6"), _sig("MXL", "GO"), _sig("NBIS", "CAUTION", "4/6")])
    assert pe.run("evening", DAY) == ["bounce"]
    to, subject, body = env["mail"].call_args.args
    assert to == pe.EMAIL_TO
    assert subject == "BOUNCE SETUP for Thu 10/15: SOXL MXL (2 GO)"
    assert json.loads(pe.STATE_FILE.read_text())["bounce"]["target"] == "2026-10-15"
    # a second run the same evening stays quiet
    assert pe.run("evening", DAY) == []
    assert env["mail"].call_count == 1


def test_one_go_is_quiet_and_sides_never_mix(env):
    _report(env["dir"] / "2026-10-14_evening.json",
            [_sig("SOXL", "GO"), _sig("NBIS", "CAUTION"),
             _sig("PENG", "GO", "5/5", bucket="reversal")])
    assert pe.run("evening", DAY) == []
    env["mail"].assert_not_called()
    assert not pe.STATE_FILE.exists()


def test_reversal_side_fires_on_its_own_gos(env):
    _report(env["dir"] / "2026-10-14_evening.json",
            [_sig("STX", "GO", "5/5", bucket="reversal", setup_type="3DGapFade",
                  metrics={"pct_from_9ema": 0.31, "pct_change_3": 0.22}),
             _sig("MU", "GO", "5/5", bucket="reversal", setup_type="2DGapFade", metrics={}),
             _sig("SOXL", "GO")])
    assert pe.run("evening", DAY) == ["reversal"]
    subject, body = env["mail"].call_args.args[1:3]
    assert subject.startswith("REVERSAL SETUP for Thu 10/15: STX MU")
    assert "104.00" in body          # +4% gap level off the 100 close
    assert "+31%" in body            # above the 9EMA


def test_no_report_is_quiet(env):
    assert pe.run("evening", DAY) == []
    env["mail"].assert_not_called()


def test_morning_catch_up_uses_the_premarket_scanner_and_skips_after_an_evening_send(env):
    today = datetime.date(2026, 10, 15)
    _report(env["dir"] / "2026-10-15_morning.json", [_sig("NBIS", "GO", "6/6")], session="morning")
    pe.PM_STATE_FILE.write_text(json.dumps(
        {"date": "2026-10-15", "alerted": {"NBIS": 6, "SOXL": 5}, "brief_sent": False}))
    assert pe.run("morning", today) == ["bounce"]
    subject, body = env["mail"].call_args.args[1:3]
    assert subject == "BOUNCE SETUP today 10/15: NBIS SOXL (2 GO)"
    assert body.count("from the premarket scanner, not the report") == 1        # SOXL only
    # the evening before already covered 10/15 -> the catch-up stays quiet
    env["mail"].reset_mock()
    pe.STATE_FILE.write_text(json.dumps({"bounce": {"target": "2026-10-15", "session": "evening"}}))
    assert pe.run("morning", today) == []
    env["mail"].assert_not_called()


def test_stale_premarket_state_is_ignored(env):
    today = datetime.date(2026, 10, 15)
    _report(env["dir"] / "2026-10-15_morning.json", [_sig("NBIS", "GO")], session="morning")
    pe.PM_STATE_FILE.write_text(json.dumps({"date": "2026-10-14", "alerted": {"SOXL": 6}}))
    assert pe.run("morning", today) == []


def test_force_resends_without_recording(env):
    _report(env["dir"] / "2026-10-14_evening.json", [_sig("SOXL", "GO")])
    assert pe.run("evening", DAY, force=True) == ["bounce", "reversal"]
    assert env["mail"].call_count == 2
    assert not pe.STATE_FILE.exists()


def test_dry_writes_the_preview_and_sends_nothing(env):
    _report(env["dir"] / "2026-10-14_evening.json", [_sig("SOXL", "GO"), _sig("MXL", "GO")])
    out = env["tmp"] / "preview.html"
    assert pe.run("evening", DAY, dry=True, out=out) == ["bounce"]
    env["mail"].assert_not_called()
    assert not pe.STATE_FILE.exists()
    assert "SOXL" in (env["tmp"] / "preview_bounce.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- email body

def test_body_carries_levels_tiers_and_the_note_in_order(env):
    _report(env["dir"] / "2026-10-14_evening.json",
            [_sig("SOXL", "GO", "6/6", odds={"p_bounce5": 0.64}), _sig("MXL", "GO", cap="Medium")])
    pe.run("evening", DAY)
    body = env["mail"].call_args.args[2]
    assert "≤ 95.00 (-5%)" in body                       # GO level off the 100 close
    assert "130.00" in body and "64%" in body           # mid-BB, odds
    assert body.count("leveraged: check the principal (SMH, MU or SKHY)") == 1   # SOXL only
    assert "1R = $3,000. A 15R = 3 shots of 5.0R ($15.0k)" in body
    assert "D 0.2R" not in body
    # note sections in note order; Notes and empty sections stay out; markdown handled
    assert body.index("Opens (A / B / C)") < body.index("Risk budget") < body.index("Head")
    assert "scratch the email must not show" not in body and "Empty section" not in body
    assert "<b>A: gaps through the GO level</b>" in body
    assert "opens above it. continues on the next line" in body
    assert "&lt;not html&gt;" in body
    assert body.index("1R = $3,000") < body.index("Three shots per idea")


def test_missing_note_still_sends_the_table_with_a_warning(env, monkeypatch):
    monkeypatch.setattr(pe, "VAULT_CANDIDATES", (env["tmp"] / "nowhere",))
    _report(env["dir"] / "2026-10-14_evening.json", [_sig("SOXL", "GO"), _sig("MXL", "GO")])
    assert pe.run("evening", DAY) == ["bounce"]
    body = env["mail"].call_args.args[2]
    assert "Lessons note not found" in body and "SOXL" in body
    assert "1R = $3,000" in body


def test_go_level_cells():
    assert pe._go_cell({"close": 100.0, "go_gap": None}) == "not within −20%"
    assert pe._go_cell({"close": 100.0, "go_gap": 0.0, "go_price": 100.0}) == "GO at a flat open"
    assert pe._go_cell({"ticker": "X"}) == "–"


def test_parse_lessons_and_inline_markdown():
    sections = pe.parse_lessons(NOTE)
    assert [h for h, _ in sections] == ["Opens (A / B / C)", "Risk budget", "Head"]
    assert sections[2][1] == ["Trust the tools <not html>."]
    assert pe.md_inline("see [[Bounce  Play|the study]] and *this* `x`") == \
        "see the study and <i>this</i> x"


def test_vault_env_var_wins(tmp_path, monkeypatch):
    (tmp_path / pe.NOTE_DIR).mkdir(parents=True)
    (tmp_path / pe.NOTE_DIR / "Reversal Lessons.md").write_text("## X\n- y\n")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(tmp_path))
    assert pe.find_note("reversal") == tmp_path / pe.NOTE_DIR / "Reversal Lessons.md"


# ---------------------------------------------------------------- replay

def test_replay_counts_evenings_and_morning_catch_ups(env, capsys):
    _report(env["dir"] / "2026-10-13_evening.json", [_sig("SOXL", "GO"), _sig("MXL", "GO")])
    _report(env["dir"] / "2026-10-14_morning.json", [_sig("SOXL", "GO"), _sig("NBIS", "GO")],
            session="morning")                                    # covered by 10/13 evening
    _report(env["dir"] / "2026-10-15_morning.json", [_sig("NBIS", "GO"), _sig("BE", "GO")],
            session="morning")                                    # catch-up
    _report(env["dir"] / "2026-10-15_evening.json", [_sig("PENG", "GO")])
    fired = pe.replay()
    assert [(d, s) for d, s, _ in fired["bounce"]] == [
        ("2026-10-13", "evening"), ("2026-10-15", "morning")]
    assert fired["reversal"] == []
    assert "1 evening + 1 morning catch-up" in capsys.readouterr().out
