"""Quiet runs keep their records but never send empty trading alerts."""

from unittest.mock import Mock

import pandas as pd
import pytest

from scanners import premarket_bounce_scanner as pm
from scripts import generate_report as gr
from scripts import priority_report as pr
from scripts import signal_scorecard as sc


def bounce_row(ticker="CBRS", rec="NO-GO", score=3):
    return {
        "ticker": ticker, "rec": rec, "score": score, "cap": "Medium",
        "price": 184.95, "gap_pct": -0.051, "selloff": -0.127,
        "off_30d": -0.235, "pm_low_time": "07:10", "off_pm_low": 0.004,
        "mid_bb": 195.80, "to_mid_bb": 0.059,
    }


@pytest.mark.parametrize("rows", [[], [bounce_row()], [bounce_row(score=6)],
                                     [bounce_row(rec="CAUTION", score=2)]])
def test_premarket_brief_has_nothing_to_send(rows):
    assert not pm.brief_is_actionable(rows)


def test_premarket_brief_excludes_no_go_even_in_a_mixed_watchlist():
    rows = [bounce_row(), bounce_row("GOOD", "GO", 5),
            bounce_row("WATCH", "CAUTION", 4), bounce_row("VETO", score=6)]
    subject, html = pm.format_brief_email(rows, pd.Timestamp("2026-09-30 08:04", tz=pm.TZ))
    assert "1 GO / 1 CAUTION" in subject
    assert "GOOD" in html and "WATCH" in html
    assert "CBRS" not in html and "VETO" not in html and "NO-GO" not in html


def test_premarket_holds_empty_brief_then_sends_later_caution_once(monkeypatch):
    monkeypatch.setattr(pm.argparse.ArgumentParser, "parse_args",
                        lambda self: Mock(replay=None, once=False, dry=False))
    monkeypatch.setattr(pm, "load_watchlist", lambda: ["CBRS"])
    monkeypatch.setattr(pm, "build_static", lambda *args: {})
    monkeypatch.setattr(pm, "get_cap", lambda *args: "Medium")
    monkeypatch.setattr(pm, "_load_json", lambda *args: {})
    monkeypatch.setattr(pm, "trillium_prices", lambda *args: {})
    scans = [[bounce_row()], [bounce_row(), bounce_row("WATCH", "CAUTION", 4)],
             [bounce_row("WATCH", "CAUTION", 4)]]
    monkeypatch.setattr(pm, "scan", Mock(side_effect=scans))
    times = ["08:04"] * 4 + ["08:09"] * 2 + ["08:14"] * 2 + ["09:31"]
    monkeypatch.setattr(pm, "_now_et", Mock(side_effect=[
        pd.Timestamp(f"2026-09-30 {t}", tz=pm.TZ) for t in times]))
    state = {"date": "2026-09-30", "alerted": {}, "brief_sent": False}
    monkeypatch.setattr(pm, "load_state", lambda date: state)
    states = []
    monkeypatch.setattr(pm, "save_state", lambda s: states.append(s["brief_sent"]))
    monkeypatch.setattr(pm.time, "sleep", lambda seconds: None)
    mail = Mock()
    monkeypatch.setattr(pm, "send_email", mail)

    pm.main()

    mail.assert_called_once()
    assert "0 GO / 1 CAUTION" in mail.call_args.args[1]
    assert "WATCH" in mail.call_args.args[2] and "CBRS" not in mail.call_args.args[2]
    assert states[0] is False and state["brief_sent"] is True


def test_premarket_rejected_name_cannot_fire_score_crossing_email(monkeypatch):
    mail = Mock()
    monkeypatch.setattr(pm, "send_email", mail)
    monkeypatch.setattr(pm, "save_state", Mock())
    state = {"alerted": {}}
    pm.process_alerts([bounce_row(score=6)], state,
                      pd.Timestamp("2026-09-30 08:04", tz=pm.TZ), do_email=True)
    mail.assert_not_called()
    assert state["alerted"] == {}


def test_premarket_go_crossing_still_sends_and_deduplicates(monkeypatch):
    mail = Mock()
    monkeypatch.setattr(pm, "send_email", mail)
    monkeypatch.setattr(pm, "save_state", Mock())
    monkeypatch.setattr(pm, "format_alert_email", lambda *args: ("GO alert", "GO body"))
    state = {"alerted": {}}
    row = bounce_row("GOOD", "GO", 5)
    asof = pd.Timestamp("2026-09-30 08:04", tz=pm.TZ)
    pm.process_alerts([row], state, asof, do_email=True)
    pm.process_alerts([row], state, asof, do_email=True)
    mail.assert_called_once()
    assert state["alerted"] == {"GOOD": 5}


def test_priority_zero_windows_skips_email_and_clears_stale_speech(monkeypatch, tmp_path):
    monkeypatch.setattr(pr, "_SIGNAL_DIR", tmp_path)
    (tmp_path / "latest_tts.txt").write_text("Previous report sent. 2 windows open.")
    mail = Mock()
    monkeypatch.setattr(pr, "send_email", mail)
    pr._send_report("empty report", 0, 0)
    mail.assert_not_called()
    assert (tmp_path / "latest_tts.txt").read_text() == ""


@pytest.mark.parametrize("go,caution", [(1, 0), (0, 1), (2, 1)])
def test_priority_open_windows_still_send(monkeypatch, tmp_path, go, caution):
    monkeypatch.setattr(pr, "_SIGNAL_DIR", tmp_path)
    mail = Mock()
    monkeypatch.setattr(pr, "send_email", mail)
    pr._send_report("actionable report", go, caution)
    mail.assert_called_once()
    assert f"{go + caution} Window" in mail.call_args.kwargs["subject"]
    assert mail.call_args.kwargs["body"] == "actionable report"


def test_priority_quiet_run_keeps_ledger_and_clears_signal_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pr.ss, "watchlist", ["CBRS"])
    monkeypatch.setattr(pr.ss, "get_all_stocks_data", lambda tickers: {"CBRS": {}})
    monkeypatch.setattr(pr.gr, "ReportCache", Mock())
    monkeypatch.setattr(pr.gr, "get_pretrade_metrics", lambda *args: {})
    monkeypatch.setattr(pr.gr, "route_playbook", lambda *args, **kw: ("reversal", "", False))
    monkeypatch.setattr(pr.gr, "get_ticker_cap", lambda ticker: "Medium")
    monkeypatch.setattr(pr.gr, "score_pretrade_setup", lambda *args, **kw:
                        {"recommendation": "NO-GO", "score": 3, "max_score": 5})
    ledger, signals, mail = Mock(), Mock(), Mock()
    monkeypatch.setattr(pr, "log_signals", ledger)
    monkeypatch.setattr(pr, "_save_signals_to_json", signals)
    monkeypatch.setattr(pr, "_SIGNAL_DIR", tmp_path / "signals")
    monkeypatch.setattr(pr, "send_email", mail)

    html = pr.generate_priority_report()

    assert "Priority Report" in html
    mail.assert_not_called()
    assert ledger.call_args.args[2][0]["rec"] == "NO-GO"
    assert signals.call_args_list[0].args == ([], 0, 0)


def signal(ticker="CBRS", target="2026-09-29", cohort="surfaced", atr=0.10):
    return {"ticker": ticker, "bucket": "bounce", "cohort": cohort,
            "signal_date": target, "target_date": target, "session": "morning",
            "recommendation": "GO", "score": "5/6", "atr_pct": atr, "metrics": {}}


def run_scorecard(monkeypatch, signals, bars, argv=None):
    monkeypatch.setattr(sc.sys, "argv", argv or ["signal_scorecard.py"])
    monkeypatch.setattr(sc, "collect_signals", lambda: {
        (s["target_date"], s["ticker"], s["bucket"], s["cohort"]): s for s in signals})
    monkeypatch.setattr(sc, "load_outcomes", lambda: {})
    monkeypatch.setattr(sc, "last_completed_trading_day", lambda: "2026-09-29")
    monkeypatch.setattr(sc, "trading_days_from", lambda target, horizon: [target])
    monkeypatch.setattr(sc, "_polygon_client", lambda: object())
    monkeypatch.setattr(sc, "fetch_daily_bars", lambda client, ticker, start, end:
                        {start: bars[0]} if bars else {})
    monkeypatch.setattr(sc, "rolling_summary", lambda rows: {})
    monkeypatch.setattr(sc.time, "sleep", lambda seconds: None)
    save, mail, render = Mock(), Mock(), Mock(return_value="scorecard")
    monkeypatch.setattr(sc, "save_outcomes", save)
    monkeypatch.setattr(sc, "send_email", mail)
    monkeypatch.setattr(sc, "format_email", render)
    sc.main()
    return save, mail, render


MISS = [{"open": 100, "high": 102, "low": 98, "close": 101}]
HIT = [{"open": 100, "high": 110, "low": 98, "close": 106}]


@pytest.mark.parametrize("signals,bars", [([], []), ([signal()], MISS),
                                         ([signal()], []), ([signal(atr=None)], HIT),
                                         ([signal(target="2026-09-28")], HIT)])
def test_scorecard_zero_tradeable_daily_signals_stays_quiet_and_saves(monkeypatch, signals, bars):
    save, mail, render = run_scorecard(monkeypatch, signals, bars)
    save.assert_called_once()
    assert len(save.call_args.args[0]) == len(signals)
    mail.assert_not_called()
    render.assert_not_called()


def test_scorecard_tradeable_signal_still_sends(monkeypatch):
    save, mail, render = run_scorecard(monkeypatch, [signal()], HIT)
    save.assert_called_once()
    mail.assert_called_once()
    assert "bnc 1/1" in mail.call_args.args[1]
    assert render.call_args.args[0][0]["tradeable_3d"] is True


@pytest.mark.parametrize("flag,writes", [("--no-email", True), ("--dry", False)])
def test_scorecard_explicit_no_email_modes_still_respected(monkeypatch, flag, writes):
    save, mail, render = run_scorecard(monkeypatch, [signal()], HIT,
                                     ["signal_scorecard.py", flag])
    assert save.called is writes
    mail.assert_not_called()
    render.assert_not_called()


@pytest.mark.parametrize("rec,send", [("NO-GO", False), (None, False),
                                    ("GO", True), ("CAUTION", True)])
def test_watchlist_email_requires_an_actionable_setup(monkeypatch, tmp_path, rec, send):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gr.ss, "watchlist", ["CBRS"])
    monkeypatch.setattr(gr.ss, "get_all_stocks_data", lambda tickers: {"CBRS": {}})
    monkeypatch.setattr(gr.ss, "calculate_percentiles", lambda *args: {})
    monkeypatch.setattr(gr, "ReportCache", Mock())
    monkeypatch.setattr(gr, "get_pretrade_metrics", lambda *args: {})
    monkeypatch.setattr(gr, "route_playbook", lambda *args, **kw: ("reversal", "", False))
    row = {"ticker": "CBRS", "bucket": "reversal", "rec": rec} if rec else None
    monkeypatch.setattr(gr, "_build_ticker_html", lambda *args, **kw: ("CBRS report", [], row))
    monkeypatch.setattr(gr, "create_daily_chart", lambda *args, **kw: tmp_path / "chart.png")
    monkeypatch.setattr(gr, "_png_to_data_uri", lambda path: "data:image/png;base64,stub")
    monkeypatch.setattr(gr, "create_rs_momentum_heatmap", lambda *args, **kw: None)
    monkeypatch.setattr(gr, "create_rs_absolute_heatmap", lambda *args, **kw: None)
    ledger, pdf, mail = Mock(), Mock(), Mock()
    monkeypatch.setattr(gr, "log_signals", ledger)
    monkeypatch.setattr(gr, "_save_report_pdf", pdf)
    monkeypatch.setattr(gr, "send_email", mail)

    html = gr.generate_report()

    assert "CBRS report" in html
    ledger.assert_called_once()
    pdf.assert_called_once_with(html)
    assert mail.called is send
