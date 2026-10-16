"""Method: the clock, open scenarios, history episodes, a refresh pass on
stubbed market data, and the server's routes. No network."""

import datetime
import json
import sys
import types

import pandas as pd
import pytest

from method import core
from method import refresh as rf

ET = "US/Eastern"


def at(hhmm, day="2026-10-15"):
    return pd.Timestamp(f"{day} {hhmm}", tz=ET).to_pydatetime()


# ---------------------------------------------------------------- clock

@pytest.mark.parametrize("hhmm,mode", [
    ("03:59", "closed"), ("04:00", "premarket"), ("09:29", "premarket"), ("09:30", "market"),
    ("15:59", "market"), ("16:00", "closed"), ("16:29", "closed"), ("16:30", "evening"),
    ("19:59", "evening"), ("20:00", "closed")])
def test_mode_windows(hhmm, mode):
    assert core.mode_for(at(hhmm), session_day=True) == mode


def test_non_session_day_is_closed():
    assert core.mode_for(at("08:00"), session_day=False) == "closed"


def test_evening_plans_the_next_session():
    nxt = lambda d: d + datetime.timedelta(days=1)
    today = datetime.date(2026, 10, 15)
    assert core.session_for("evening", today, nxt) == datetime.date(2026, 10, 16)
    assert core.session_for("premarket", today, nxt) == today


def test_cadence_skips_passes_between():
    last = at("10:00")
    assert not core.due("market", last, at("10:05"))
    assert core.due("market", last, at("10:14"))       # a minute of cron slack
    assert core.due("premarket", at("08:00"), at("08:04"))
    assert core.due("market", None, at("10:00"))


# ---------------------------------------------------------------- scenarios

def test_reversal_scenarios():
    assert core.reversal_scenario(0.05) == "A"
    assert core.reversal_scenario(0.04) == "A"
    assert core.reversal_scenario(0.01) == "B"
    assert core.reversal_scenario(-0.02) == "C"


def test_clean_makes_numpy_and_nan_json_safe():
    import numpy as np
    out = core.clean({"a": np.float64(1.5), "b": float("nan"), "c": [np.int64(2)]})
    assert out == {"a": 1.5, "b": None, "c": [2]}
    json.dumps(out)


# ---------------------------------------------------------------- history

def _sig(t, rec, bucket="bounce", score="5/6"):
    return {"ticker": t, "bucket": bucket, "recommendation": rec, "score": score, "cap": "ETF"}


def test_episodes_merge_reports_for_one_session_and_join_outcomes():
    nxt = lambda d: d + datetime.timedelta(days=1)
    reports = {
        ("2026-10-14", "evening"): {"signals": [_sig("SOXL", "GO"), _sig("MXL", "GO"),
                                                _sig("NBIS", "CAUTION", score="4/6")]},
        ("2026-10-15", "morning"): {"signals": [_sig("NBIS", "GO"), _sig("SOXL", "GO")]},
        ("2026-10-15", "evening"): {"signals": [_sig("PENG", "GO")]},             # 1 GO: no episode
    }
    outcomes = [
        {"target_date": "2026-10-15", "bucket": "bounce", "ticker": "SOXL", "session": "evening",
         "d0_pct": "0.01", "mfe_atr_d0": "0.5", "exit_r": "", "tradeable_d0": "False"},
        {"target_date": "2026-10-15", "bucket": "bounce", "ticker": "SOXL", "session": "morning",
         "d0_pct": "0.078", "mfe_atr_d0": "1.9", "exit_r": "2.5", "tradeable_d0": "True"},
    ]
    eps = core.build_episodes(reports, outcomes, nxt)
    assert len(eps) == 1
    ep = eps[0]
    assert ep["target"] == "2026-10-15" and ep["side"] == "bounce"
    assert ep["reports"] == ["2026-10-14 evening", "2026-10-15 morning"]
    names = {n["ticker"]: n for n in ep["names"]}
    assert names["NBIS"]["rec"] == "GO"                   # CAUTION upgraded by the later GO
    assert names["SOXL"]["d0_pct"] == pytest.approx(0.078)  # morning row wins
    assert names["SOXL"]["tradeable_d0"] is True
    assert names["MXL"]["d0_pct"] is None
    assert ep["n_go"] == 3


# ---------------------------------------------------------------- refresh pass

STATIC = {"prior_close": 100.0, "consecutive_down_days": 4, "atr": 8.0, "mid_bb": 130.0,
          "sma200": None, "close_30_ago": 160.0, "close_15_ago": 140.0, "close_3_ago": 120.0,
          "prior_day_range": 9.0, "high_30d": 165.0, "high_52wk": 200.0,
          "selloff_first_open": 140.0}


# a ladder in the game plan's shape: B from 96 down, A from 92, A+ from 85
LADDER = {
    "setup_type": "GapFade_weakstock",
    "levels": [{"price": 96.0, "gap": -0.04, "grade": "B", "score": 5, "intensity": 50.0, "events": ["grade B"]},
               {"price": 92.0, "gap": -0.08, "grade": "A", "score": 6, "intensity": 65.0, "events": ["grade A"]},
               {"price": 85.0, "gap": -0.15, "grade": "A+", "score": 6, "intensity": 80.0, "events": ["grade A+"]}],
    "zones": [{"grade": "C", "top": 102.0, "bottom": 96.0}, {"grade": "B", "top": 96.0, "bottom": 92.0},
              {"grade": "A", "top": 92.0, "bottom": 85.0}, {"grade": "A+", "top": 85.0, "bottom": 60.0}],
    "ceiling": {"score": 6, "grade": "A+", "blocked": [], "floor_price": 60.0},
}


def fake_grade_at(s, cap, price, ticker):
    """The real checklist at `price`, graded off LADDER's price bands."""
    from analyzers.bounce_scorer import BouncePretrade
    from scanners.premarket_bounce_scanner import metrics_from_price
    m = metrics_from_price(s, price)
    res = BouncePretrade().validate(ticker, m, cap=cap, force_setup=s.get("setup_type"))
    grade = "A+" if price <= 85 else "A" if price <= 92 else "B" if price <= 96 else "C"
    return {"grade": grade, "intensity": 70.0, "score": res.score, "rec": res.recommendation,
            "result": res, "metrics": m}


def _bars(day, rows):
    """rows: [(HH:MM, open, high, low, close)] -> Polygon-shaped minute frame."""
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day} {t}", tz=ET) for t, *_ in rows])
    return pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
                         "volume": 1000.0}, index=idx)


@pytest.fixture
def env(tmp_path, monkeypatch):
    sig = tmp_path / "priority_signals"
    sig.mkdir()
    monkeypatch.setattr(rf, "METHOD_DIR", tmp_path / "method")
    monkeypatch.setattr(rf, "SNAPSHOT", tmp_path / "method" / "snapshot.json")
    monkeypatch.setattr(rf, "SIGNALS_DIR", sig)
    monkeypatch.setattr(rf, "load_watchlist", lambda: ["SOXL", "MXL"])
    monkeypatch.setattr(rf, "_load_json", lambda p: {})
    monkeypatch.setattr(rf, "get_cap", lambda t, c: "ETF" if t == "SOXL" else "Medium")
    monkeypatch.setattr(rf, "is_session_day", lambda d: True)
    monkeypatch.setattr(rf, "next_trading_day", lambda d: d + datetime.timedelta(days=1))
    built = []

    def fake_daily(t, session):
        built.append((t, session))
        return pd.DataFrame({"close": [100.0] * 20})
    monkeypatch.setattr(rf, "completed_daily", fake_daily)
    monkeypatch.setattr(rf, "static_from_hist", lambda hist: dict(STATIC))
    monkeypatch.setattr(rf, "go_gap_trigger", lambda t, s, cap, checker: -0.04)
    monkeypatch.setattr(rf, "ladder_payload", lambda s, cap, t: dict(LADDER))
    monkeypatch.setattr(rf, "grade_at", fake_grade_at)
    monkeypatch.setattr(rf, "historical_grade_stats", lambda: {"A": {"n": 15, "med_open_close": 0.117}})
    monkeypatch.setattr(rf, "lookup_bounce_odds", lambda score, cap: {"p_bounce5": 0.6})
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))       # no phone subscribed: no pushes
    monkeypatch.setattr(rf.time, "sleep", lambda s: None)
    rev = []
    monkeypatch.setattr(rf, "reversal_pass", lambda cands, caps, today, setups: rev.append(cands) or [])
    (sig / "2026-10-14_evening.json").write_text(json.dumps({
        "date": "2026-10-14", "session": "evening", "generated_at": "2026-10-14T15:15:41",
        "signals": [_sig("SOXL", "GO", score="6/6") | {"odds": {"p_bounce5": 0.64}}],
        "cluster_signal": {"level": None, "bounce_open": 1}}))
    return {"dir": tmp_path, "built": built, "rev": rev, "monkeypatch": monkeypatch}


def test_premarket_pass_grades_and_tracks_breadth(env):
    day = "2026-10-15"
    env["monkeypatch"].setattr(rf.pq, "get_intraday", lambda t, d, m, s: _bars(day, [
        ("04:00", 99, 99, 95, 96), ("07:06", 96, 96, 90, 91 if t == "SOXL" else 98)]))
    snap = rf.run(now=at("08:35"))
    assert snap["mode"] == "premarket" and snap["session"] == day
    assert snap["data_time"] == "07:06"
    rows = {r["ticker"]: r for r in snap["bounce"]["rows"]}
    soxl, mxl = rows["SOXL"], rows["MXL"]
    assert soxl["gap"] == pytest.approx(-0.09) and soxl["grade"] == "A"    # 91 is in the A band
    assert soxl["next"]["grade"] == "A+" and soxl["next"]["price"] == 85.0
    assert soxl["next"]["pct_away"] == pytest.approx(85 / 91 - 1)
    assert soxl["go_price"] == pytest.approx(96.0)
    assert soxl["pm_low"] == 90 and soxl["pm_low_time"] == "07:06"
    assert mxl["grade"] == "C" and mxl["next"]["grade"] == "B"            # 98: next is B at 96
    assert soxl["ladder"]["levels"][0]["price"] == 96.0 and soxl["odds_pop"]["p_bounce5"] == 0.6
    assert "scenario" not in soxl
    assert snap["grade_stats"]["A"]["n"] == 15
    assert soxl["report_rec"] == "GO" and soxl["odds"]["p_bounce5"] == 0.64
    assert soxl["principal"] and not mxl.get("principal")
    assert json.loads(rf.SNAPSHOT.read_text())["breadth"][0]["t"] == "08:35"
    # statics are cached for the session: a second pass doesn't rebuild them
    n_built = len(env["built"])
    rf.run(now=at("08:41"))
    assert len(env["built"]) == n_built
    assert len(rf.read_breadth(day)) == 2


def test_cadence_skip_and_closed(env):
    env["monkeypatch"].setattr(rf.pq, "get_intraday",
                                 lambda *a: _bars("2026-10-15", [("10:00", 99, 99, 95, 96)]))
    assert rf.run(now=at("10:00")) is not None
    assert rf.run(now=at("10:05")) is None                       # market cadence 15 min
    assert rf.run(now=at("10:05"), force=True) is not None
    assert rf.run(now=at("03:00", day="2026-10-17")) is None     # Saturday 3 AM: closed


def test_market_pass_grades_the_open_and_now(env):
    env["monkeypatch"].setattr(rf.pq, "get_intraday", lambda *a: _bars("2026-10-15", [
        ("09:00", 99, 99, 97, 98), ("09:30", 94, 95, 93, 94), ("10:30", 99, 101, 98, 101)]))
    snap = rf.run(now=at("10:45"))
    soxl = next(r for r in snap["bounce"]["rows"] if r["ticker"] == "SOXL")
    assert soxl["open"] == 94 and soxl["grade_open"] == "B"      # opened 94: B band
    assert soxl["grade"] == "C"                                  # now 101
    assert soxl["day_low"] == 93 and soxl["day_low_time"] == "09:30"


def test_evening_plans_tomorrow_and_rebuilds_a_stale_static(env):
    calls = []

    def daily(t, session):
        calls.append(session)
        return pd.DataFrame({"close": [100.0] * 20})
    # first build saw yesterday's close; the tape closed at 90
    env["monkeypatch"].setattr(rf, "completed_daily", daily)
    env["monkeypatch"].setattr(rf, "static_from_hist", lambda hist: dict(
        STATIC, prior_close=100.0 if len(calls) <= 2 else 90.0))
    env["monkeypatch"].setattr(rf.pq, "get_intraday", lambda *a: _bars("2026-10-15", [
        ("15:59", 91, 91, 90, 90), ("17:30", 89, 89, 87, 88)]))
    snap = rf.run(now=at("17:45"))
    assert snap["mode"] == "evening" and snap["session"] == "2026-10-16"
    assert set(calls) == {"2026-10-16"}
    soxl = next(r for r in snap["bounce"]["rows"] if r["ticker"] == "SOXL")
    assert soxl["prior_close"] == 90.0                           # rebuilt from the tape
    assert soxl["gap"] == pytest.approx(88 / 90 - 1)
    assert soxl["ah_low"] == 87


@pytest.mark.parametrize("day,hhmm,expected", [
    ("2026-10-15", "20:00", True), ("2026-10-15", "19:59", False),   # Thu evening
    ("2026-10-16", "03:59", True), ("2026-10-16", "04:00", False),   # Fri early hours
    ("2026-10-16", "21:00", False),                                  # Fri night: no Blue Ocean
    ("2026-10-17", "02:00", False), ("2026-10-18", "21:00", True),   # Sat early / Sun evening
    ("2026-10-19", "01:00", True)])                                  # Mon early hours
def test_blue_ocean_nights(day, hhmm, expected):
    assert core.is_overnight(at(hhmm, day=day)) is expected


def _overnight_feeds(monkeypatch, yahoo=None, manual=None):
    import scanners.overnight_bounce_tracker as ot
    monkeypatch.setattr(ot, "yahoo_overnight_prices", lambda tickers: dict(yahoo or {}))
    monkeypatch.setattr(ot, "load_prices_file", lambda *a: dict(manual or {}))


def test_overnight_grades_the_newest_print_for_tomorrow(env):
    mp = env["monkeypatch"]
    ts = lambda hhmm, day="2026-10-15": pd.Timestamp(f"{day} {hhmm}", tz=ET)
    core.write_json(rf.SNAPSHOT, {"generated_at": ts("19:59").isoformat(), "mode": "evening",
                                  "bounce": {"rows": [{"ticker": "SOXL", "price": 99.0},
                                                      {"ticker": "MXL", "price": 98.0}]},
                                  "reversal": {"rows": [{"ticker": "PENG", "price": 50.0, "prior_close": 50.0,
                                                         "gap": 0.0, "scenario": "B", "rec": "GO",
                                                         "score": 5}]}})
    _overnight_feeds(mp, yahoo={"SOXL": (91.0, ts("23:20")), "MXL": (97.5, ts("22:00")),
                                "PENG": (52.5, ts("23:10"))},
                     manual={"MXL": (97.0, ts("23:25"))})
    snap = rf.run(now=at("23:30"))
    assert snap["mode"] == "overnight" and snap["session"] == "2026-10-16"
    assert snap["source"] == "Blue Ocean overnight via Yahoo" and snap["data_time"] == "23:25"
    rows = {r["ticker"]: r for r in snap["bounce"]["rows"]}
    assert (rows["SOXL"]["price"], rows["SOXL"]["src"], rows["SOXL"]["grade"]) == (91.0, "Y", "A")
    assert (rows["MXL"]["price"], rows["MXL"]["src_label"]) == (97.0, "your quote")    # newer than Yahoo
    peng = snap["reversal"]["rows"][0]
    assert peng["price"] == 52.5 and peng["gap"] == pytest.approx(0.05) and peng["scenario"] == "A"


def test_overnight_falls_back_to_the_evening_price(env):
    mp = env["monkeypatch"]
    core.write_json(rf.SNAPSHOT, {"generated_at": pd.Timestamp("2026-10-15 19:59", tz=ET).isoformat(),
                                  "mode": "evening", "bounce": {"rows": [{"ticker": "SOXL", "price": 99.0}]}})
    _overnight_feeds(mp)                                         # Yahoo down, no quotes
    snap = rf.run(now=at("01:00", day="2026-10-16"))
    assert snap["session"] == "2026-10-16"                       # after midnight: today's session
    soxl = snap["bounce"]["rows"][0]
    assert (soxl["ticker"], soxl["price"], soxl["src"]) == ("SOXL", 99.0, "P")
    assert snap["data_time"] is None                             # nothing new overnight


def test_reversal_candidates_from_reports_and_9ema_stretch():
    reports = [{"signals": [_sig("PENG", "GO", bucket="reversal"), _sig("SOXL", "GO")]}]
    statics = {"MU": {"pct_from_9ema": 0.2}, "AMD": {"pct_from_9ema": 0.05}, "PENG": {}}
    assert rf.reversal_candidates(reports, statics) == ["PENG", "MU"]


def test_reversal_pass_uses_the_report_scorer(monkeypatch):
    fake = types.ModuleType("scripts.generate_report")
    fake.get_pretrade_metrics = lambda t, d: {"prior_close": 50.0, "gap_pct": 0.06,
                                              "pct_from_9ema": 0.35, "ema_9": 37.0,
                                              "pct_change_3": 0.4, "atr_pct": 0.1}
    fake.score_pretrade_setup = lambda t, m, cap: {
        "score": 4, "max_score": 5, "recommendation": "GO", "range_gate_passed": True,
        "archetype_passed": False, "momentum_pctile_passed": True,
        "criteria": [{"name": "RVOL", "passed": False, "actual": 0.8, "threshold": 1.5},
                     {"name": "Gap Up", "passed": True, "actual": 0.06, "threshold": 0.04}]}
    monkeypatch.setitem(sys.modules, "scripts.generate_report", fake)
    rows = rf.reversal_pass(["PENG"], {"PENG": "Small"}, "2026-10-15", {"PENG": "3DGapFade"})
    r = rows[0]
    assert r["rec"] == "GO" and r["setup"] == "3DGapFade" and r["scenario"] == "A"
    assert r["price"] == pytest.approx(53.0) and r["gap_level"] == pytest.approx(52.0)
    assert r["failing"] == [{"name": "RVOL", "actual": "0.80x", "threshold": "1.50x"}]
    assert r["gates_failed"] == ["archetype gate"]


# ---------------------------------------------------------------- server

@pytest.fixture
def client(tmp_path, monkeypatch):
    from method import server as sv
    sig = tmp_path / "priority_signals"
    sig.mkdir()
    bdir = tmp_path / "method"
    monkeypatch.setattr(rf, "METHOD_DIR", bdir)
    monkeypatch.setattr(sv, "SNAPSHOT", bdir / "snapshot.json")
    monkeypatch.setattr(sv, "SHOTS", bdir / "shots.json")
    monkeypatch.setattr(sv, "SIGNALS_DIR", sig)
    monkeypatch.setattr(sv, "OUTCOMES", tmp_path / "signal_outcomes.csv")
    monkeypatch.setattr(sv, "next_trading_day", lambda d: d + datetime.timedelta(days=1))
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))
    sv._history_cache.update(key=None, data=None)
    sv._bars_cache.clear()
    app = sv.create_app()
    app.testing = True
    return {"c": app.test_client(), "sv": sv, "dir": tmp_path, "sig": sig, "bdir": bdir}


def test_snapshot_route_adds_breadth(client):
    c = client["c"]
    assert c.get("/api/snapshot").get_json()["empty"] is True
    core.write_json(client["bdir"] / "snapshot.json",
                    {"generated_at": "2026-10-15T08:35:00-04:00", "mode": "premarket",
                     "session": "2026-10-15", "bounce": {"rows": [], "counts": {}}})
    (client["bdir"] / "breadth_2026-10-15.jsonl").write_text('{"t": "08:30", "bounce_go": 2}\n')
    snap = c.get("/api/snapshot").get_json()
    assert snap["mode"] == "premarket" and snap["breadth"] == [{"t": "08:30", "bounce_go": 2}]


def test_index_and_health(client):
    c = client["c"]
    assert b"Method" in c.get("/").data
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/api/health").get_json()["ok"] is True


def test_lessons_route_reads_the_note(client, monkeypatch, tmp_path):
    from scripts import playbook_email as pe
    note = tmp_path / "vault" / pe.NOTE_DIR / "Bounce Lessons.md"
    note.parent.mkdir(parents=True)
    note.write_text("## Opens (A / B / C)\n- **A** through it\n## Notes\n- private\n", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT", str(tmp_path / "vault"))
    monkeypatch.setattr(pe, "tiers_line", lambda: "1R = $3,000.")
    body = client["c"].get("/api/lessons?side=bounce").get_json()
    assert body["found"] and body["tiers"] == "1R = $3,000."
    assert body["sections"] == [{"heading": "Opens (A / B / C)", "items": ["<b>A</b> through it"]}]
    assert client["c"].get("/api/lessons?side=nope").status_code == 400


def test_shots_count_undo_and_validate(client):
    c = client["c"]
    post = lambda body: c.post("/api/shots?date=2026-10-15", json=body, headers={"X-Method": "1"})
    post({"side": "bounce", "ticker": "soxl", "kind": "shot"})
    post({"side": "bounce", "ticker": "SOXL", "kind": "stop"})
    r = post({"side": "bounce", "ticker": "KORU", "kind": "stop"}).get_json()
    assert [(e["ticker"], e["kind"]) for e in r["entries"]] == [
        ("SOXL", "shot"), ("SOXL", "stop"), ("KORU", "stop")]
    r = post({"kind": "undo"}).get_json()
    assert len(r["entries"]) == 2
    assert post({"side": "bounce", "kind": "shot"}).status_code == 400          # no ticker
    assert post({"side": "momo", "ticker": "X", "kind": "shot"}).status_code == 400
    assert c.get("/api/shots?date=2026-10-15").get_json()["entries"][0]["ticker"] == "SOXL"


def test_history_route(client):
    (client["sig"] / "2026-10-14_evening.json").write_text(json.dumps(
        {"signals": [_sig("SOXL", "GO"), _sig("MXL", "GO")]}))
    (client["dir"] / "signal_outcomes.csv").write_text(
        "signal_date,session,target_date,ticker,bucket,cohort,d0_pct,d1_pct,mfe_atr_d0,exit_r,tradeable_d0\n"
        "2026-10-14,evening,2026-10-15,SOXL,bounce,surfaced,0.05,,1.2,1.5,True\n"
        "2026-10-14,evening,2026-10-15,MXL,bounce,control,0.09,,1.2,1.5,True\n")
    eps = client["c"].get("/api/history").get_json()["episodes"]
    assert eps[0]["target"] == "2026-10-15"
    names = {n["ticker"]: n for n in eps[0]["names"]}
    assert names["SOXL"]["d0_pct"] == 0.05 and names["MXL"]["d0_pct"] is None   # control row skipped


def test_bars_route_validates_and_caches(client, monkeypatch):
    calls = []
    monkeypatch.setattr(client["sv"], "chart_bars",
                        lambda t: calls.append(t) or {"ticker": t, "daily": [], "intraday": []})
    c = client["c"]
    assert c.get("/api/bars?ticker=../etc").status_code == 400
    assert c.get("/api/bars?ticker=soxl").get_json()["ticker"] == "SOXL"
    c.get("/api/bars?ticker=SOXL")
    assert calls == ["SOXL"]
