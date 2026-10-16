"""The bounce feedback loop. Method records each session for the signal scorecard (the
breadth cohort); the scorecard keeps that cohort out of everything you act on; rules R fills
once a row's first day is over; and the edge table rebuilds from the frozen study plus the
records (the since-the-study cells, the watchlist cell, the drift warning). Polygon stubbed."""

import csv
import json
import sys

import pandas as pd
import pytest

from method import breadth as B
from method import core
from method import library
from method import refresh as rf
from scripts import backfill_entry_r as be
from scripts import bounce_day_edge as E
from scripts import signal_analysis as sa
from scripts import signal_scorecard as sc
from tests.unit.test_method_breadth import _built, _selloff, at, bdir, cand  # noqa: F401 (bdir is a fixture)


# ---------------------------------------------------------------- the day's record

def _store(tmp_path, hist, days):
    """A grouped-daily store holding SELL's history plus extra session rows {date: [rows]}."""
    store = tmp_path / "grouped"
    store.mkdir()
    for d, r in hist.iterrows():
        pd.DataFrame([{"ticker": "SELL", **r.to_dict()}]).to_pickle(store / f"{d}.pkl")
    for d, rows in days.items():
        pd.DataFrame(rows).to_pickle(store / f"{d}.pkl")
    return store


def test_every_session_gets_one_record_even_with_no_names(tmp_path, monkeypatch):
    hist = _selloff()
    s1 = (pd.Timestamp(hist.index[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    s2 = (pd.Timestamp(s1) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    monkeypatch.setattr(B, "LOOP_START", s1)
    monkeypatch.setattr(B, "METHOD_DIR", tmp_path / "method")
    c1 = B.compute_candidate("SELL", s1, hist)
    pc = c1["prior_close"]
    o1 = c1["flag_price"] * 0.98
    store = _store(tmp_path, hist, {
        s1: [{"ticker": "SELL", "open": o1, "high": pc, "low": o1 * 0.97, "close": pc * 0.99, "volume": 4e6}],
        s2: [{"ticker": "SELL", "open": pc * 1.02, "high": pc * 1.03, "low": pc, "close": pc, "volume": 3e6}]})
    core.write_json(B.live_path(s1), {"n": 1, "state": "normal", "basis": "open", "asof": f"{s1}T09:31:00-04:00"})
    out = tmp_path / "signals"
    out.mkdir()
    cands = {s1: {"SELL": c1}, s2: {"SELL": dict(c1, prior_close=pc * 0.99, flag_price=pc * 0.9, relaxed_price=None)}}
    done = B.record_sessions(s2, store=store, signals_dir=out, cands_of=cands.get, splits=set())
    assert done == [s1, s2]                                       # history days before LOOP_START aren't sessions
    r1 = json.loads(B.signal_path(s1, out).read_text())
    assert r1["n"] == 1 and r1["state"] == "normal" and r1["live"]["asof"].endswith("09:31:00-04:00")
    sig = r1["signals"][0]
    assert sig["ticker"] == "SELL" and sig["recommendation"] == "STRICT" and sig["score"] >= 4
    assert sig["metrics"]["open"] == pytest.approx(o1) and sig["metrics"]["atr_pct"] > 0
    assert sig["metrics"]["gap_pct"] == pytest.approx(o1 / pc - 1)  # the study's own metric at the open
    r2 = json.loads(B.signal_path(s2, out).read_text())
    assert r2["n"] == 0 and r2["signals"] == [] and r2["state"] == "normal"
    assert B.record_sessions(s2, store=store, signals_dir=out, cands_of=cands.get, splits=set()) == []


def test_relaxed_names_are_recorded_on_breadth_days_only():
    many = {f"N{i}": cand(relaxed=None) for i in range(11)}
    rel = {"R": dict(cand(), flag_price=None, relaxed_price=96.0)}
    opens = {t: 90.0 for t in many} | {"R": 95.0}
    rec = B.day_record("2026-10-15", many | rel, opens, lambda t: None)
    assert rec["state"] == "breadth" and {s["recommendation"] for s in rec["signals"] if s["ticker"] == "R"} == {"RELAXED"}
    small = B.day_record("2026-10-16", {"N0": many["N0"]} | rel, opens, lambda t: None)
    assert [s["ticker"] for s in small["signals"]] == ["N0"]


def test_score_at_the_open_adds_the_gap_check():
    c = cand(flag=92.0)                                            # static 3, gap threshold -8%
    assert B.score_at(c, 91.0) == 4 and B.score_at(c, 95.0) == 3
    assert B.score_at(c, 95.0, relaxed=True) == 4                  # -5% clears the halved threshold


def test_the_open_keeps_what_the_card_said_for_the_record(bdir, monkeypatch):
    monkeypatch.setattr(rf, "MARKET_BREADTH", bdir / "market_breadth.json")
    _built(bdir, A=cand(), B=cand())
    poly = {"A": {"price": 91.0, "ts": 1, "open": 91.0}, "B": {"price": 99.0, "ts": 1, "open": 99.0}}
    mb = rf.market_breadth("2026-10-15", "market", "2026-10-15", at("09:46"), {}, snapshot_prices=lambda d: poly)
    assert mb["locked"]
    live = core.read_json(B.live_path("2026-10-15"))
    assert live["n"] == 1 and live["names"] == ["A"] and live["basis"] == "open"


# ---------------------------------------------------------------- the scorecard's breadth cohort

def test_the_scorecard_reads_breadth_records_as_their_own_cohort(tmp_path, monkeypatch):
    for name in ("SIGNAL_DIR", "CONTROL_DIR", "BREADTH_DIR"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(sc, name, tmp_path / name)
    sig = {"ticker": "MUU", "bucket": "bounce", "cap": "ETF", "recommendation": "STRICT", "score": 5,
           "metrics": {"atr_pct": 0.08, "gap_pct": -0.06}}
    (tmp_path / "BREADTH_DIR" / "2026-10-15_morning.json").write_text(json.dumps({"signals": [sig]}))
    (tmp_path / "SIGNAL_DIR" / "2026-10-15_morning.json").write_text(json.dumps({"signals": [dict(sig, recommendation="GO")]}))
    got = sc.collect_signals()
    assert set(got) == {("2026-10-15", "MUU", "bounce", "surfaced"), ("2026-10-15", "MUU", "bounce", "breadth")}
    b = got[("2026-10-15", "MUU", "bounce", "breadth")]
    assert b["target_date"] == "2026-10-15" and b["atr_pct"] == 0.08 and b["recommendation"] == "STRICT"


def test_breadth_rows_stay_out_of_the_summary_episodes_and_every_reader(tmp_path, monkeypatch):
    rows = [{"ticker": "MUU", "bucket": "bounce", "target_date": "2099-01-05", "cohort": c, "complete": "True",
             "days_available": "4", "recommendation": "GO", "tradeable_3d": "True"} for c in ("surfaced", "breadth")]
    keyed = {(r["target_date"], r["ticker"], r["bucket"], r["cohort"]): r for r in rows}
    sc.assign_episodes(keyed)
    sc.assign_clusters(keyed)
    assert rows[0]["episode_signal_num"] == 1 and rows[1]["episode_signal_num"] == 1
    assert rows[1]["episode_id"] == "MUU_bounce_breadth_2099-01-05" and rows[0]["episode_id"] == "MUU_bounce_2099-01-05"
    assert sc.rolling_summary(rows)["total"] == 1
    assert [core.surfaced(r) for r in rows] == [True, False] and core.surfaced({"cohort": ""})

    path = tmp_path / "signal_outcomes.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target_date", "ticker", "bucket", "cohort", "session", "complete",
                                           "days_available"])
        w.writeheader()
        for c in ("surfaced", "control", "breadth"):
            w.writerow({"target_date": "2026-10-15", "ticker": "MUU", "bucket": "bounce", "cohort": c,
                        "session": "morning", "complete": "True", "days_available": "4"})
    assert [r["cohort"] for r in library.load_outcome_index(path).values()] == ["surfaced"]
    monkeypatch.setattr(sa, "OUTCOMES_FILE", path)
    assert sorted(r["cohort"] for r in sa.load_complete_rows()) == ["control", "surfaced"]


def test_rescoring_a_row_keeps_its_rules_r(monkeypatch):
    key = ("2026-10-13", "MUU", "bounce", "breadth")
    sig = {"signal_date": "2026-10-13", "session": "morning", "target_date": "2026-10-13", "ticker": "MUU",
           "bucket": "bounce", "cohort": "breadth", "cap": "ETF", "recommendation": "STRICT", "score": 5,
           "atr_pct": 0.08, "metrics": {}}
    old = {**{c: "" for c in sc.COLUMNS}, **{k: v for k, v in sig.items() if k != "metrics"},
           "days_available": "1", "complete": "False", "exit_r": "1.5", "entry_signals": "1", "entry_r": "2.0"}
    saved = {}
    monkeypatch.setattr(sc, "collect_signals", lambda: {key: sig})
    monkeypatch.setattr(sc, "load_outcomes", lambda: {key: old})
    monkeypatch.setattr(sc, "save_outcomes", lambda rows: saved.update(rows))
    monkeypatch.setattr(sc, "last_completed_trading_day", lambda: "2026-10-15")
    monkeypatch.setattr(sc, "trading_days_from", lambda start, n: ["2026-10-13", "2026-10-14", "2026-10-15"])
    monkeypatch.setattr(sc, "_polygon_client", lambda: None)
    bar = {"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5}
    monkeypatch.setattr(sc, "fetch_daily_bars", lambda c, t, a, b: {d: bar for d in ("2026-10-13", "2026-10-14", "2026-10-15")})
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["signal_scorecard", "--no-email"])
    sc.main()
    row = saved[key]
    assert row["days_available"] == 3 and row["exit_r"] == "1.5" and row["entry_signals"] == "1" and row["entry_r"] == "2.0"


def test_rules_r_runs_once_the_first_day_is_over():
    rows = [{"bucket": "bounce", "target_date": "2026-10-15", "days_available": "1", "entry_signals": ""},
            {"bucket": "bounce", "target_date": "2026-10-16", "days_available": "1", "entry_signals": ""},   # today
            {"bucket": "bounce", "target_date": "2026-10-14", "days_available": "4", "entry_signals": "0"},  # replayed
            {"bucket": "bounce", "target_date": "2026-10-14", "days_available": "0", "entry_signals": ""},   # no open yet
            {"bucket": "reversal", "target_date": "2026-10-14", "days_available": "4", "entry_signals": ""}]
    assert be.pending(rows, "2026-10-16") == [rows[0]]
    assert be.pending(rows, "2026-10-16", refresh=True) == [rows[0], rows[2]]


# ---------------------------------------------------------------- the table learns

def _hist():
    rows = []
    for i in range(40):                                            # a study breadth day per 10 names
        rows.append({"date": f"2025-0{1 + i // 10}-0{1 + i % 5}", "ticker": f"S{i}", "kind": "strict",
                     "score": 4 + i % 3, "n_flag": 12, "state": "breadth", "exit_r": 1.0, "r1": 1.0, "r2": None})
    rows.append({"date": "2025-05-01", "ticker": "L", "kind": "low", "score": 3, "n_flag": 12, "state": "breadth",
                 "exit_r": -1.0, "r1": -1.0, "r2": None})
    days = pd.DataFrame({"date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"],
                         "n_flag": [12, 12, 12, 12, 0], "state": ["breadth"] * 4 + ["normal"]})
    return pd.DataFrame(rows), days


def _rec(n_days, r=-0.5):
    names = pd.DataFrame([{"date": f"2026-07-0{d + 1}", "ticker": f"R{d}{k}", "kind": "strict", "score": 4,
                           "n_flag": 11, "state": "breadth", "exit_r": r, "r1": r, "r2": None}
                          for d in range(n_days) for k in range(3)], columns=E.NAME_COLS)
    days = pd.DataFrame({"date": [f"2026-07-0{d + 1}" for d in range(n_days)], "n_flag": 11, "state": "breadth"})
    return names, days


def test_recorded_days_join_the_cells_and_get_their_own_line():
    hist, days = _hist()
    rec, rec_days = _rec(2, r=2.0)
    t = E.build_table(hist, days, rec, rec_days)
    b = t["states"]["breadth"]
    assert b["n"] == 46 and b["days"] == 6 and b["mean_r"] == pytest.approx((40 * 1.0 + 6 * 2.0) / 46, abs=1e-3)
    assert b["since_study"] == {**b["since_study"], "n": 6, "days": 2, "mean_r": 2.0, "since": B.LOOP_START}
    assert b["by_score"]["low"]["raw"]["n"] == 1                    # the low cell stays the study's
    assert t["drift"] is None and t["states"]["normal"]["since_study"]["days"] == 0


def test_the_card_warns_when_five_recorded_breadth_days_run_negative():
    hist, days = _hist()
    assert E.build_table(hist, days, *_rec(4))["drift"] is None
    d = E.build_table(hist, days, *_rec(5))["drift"]
    assert d == {"state": "breadth", "days": 5, "n": 15, "mean_r": -0.5, "history_mean_r": d["history_mean_r"]}
    assert E.build_table(hist, days, *_rec(5, r=0.2))["drift"] is None


def test_records_replay_counted_names_once_the_day_is_over(tmp_path, monkeypatch):
    out = tmp_path / "signals"
    out.mkdir()
    sig = lambda t, counted=True: {"ticker": t, "recommendation": "STRICT", "score": 5, "counted": counted,
                                   "metrics": {"atr_pct": 0.05}}
    for d in ("2026-10-15", "2026-10-16"):
        (out / f"{d}_morning.json").write_text(json.dumps({"date": d, "n": 12, "state": "breadth", "n_etf": 3,
                                                           "signals": [sig("A"), sig("B"), sig("THIN", counted=False)]}))
    seen = []

    def split(rows):
        seen.extend(zip(rows.date, rows.ticker))
        return pd.DataFrame({"date": ["2026-10-15"] * 2, "ticker": ["A", "B"], "trades": [2, 0], "r1": [-1.0, None],
                             "r2": [3.5, None]})
    monkeypatch.setattr(E, "split_entries", split)
    names, days = E.recorded("2026-10-16", out)
    assert seen == [("2026-10-15", "A"), ("2026-10-15", "B")]          # today waits; THIN isn't counted
    assert list(names.ticker) == ["A"] and names.exit_r.iloc[0] == pytest.approx(2.5)   # B never entered
    assert list(days.date) == ["2026-10-15", "2026-10-16"] and set(days.state) == {"breadth"}


def test_the_watchlist_cell_is_your_surfaced_bounces_by_day(tmp_path):
    path = tmp_path / "o.csv"
    pd.DataFrame([
        {"target_date": "2026-03-02", "bucket": "bounce", "cohort": "surfaced", "recommendation": "GO", "entry_signals": "1", "exit_r": "2.0"},
        {"target_date": "2026-03-02", "bucket": "bounce", "cohort": "control", "recommendation": "NO-GO", "entry_signals": "1", "exit_r": "9"},
        {"target_date": "2026-03-02", "bucket": "bounce", "cohort": "breadth", "recommendation": "STRICT", "entry_signals": "1", "exit_r": "9"},
        {"target_date": "2026-03-03", "bucket": "bounce", "cohort": "", "recommendation": "CAUTION", "entry_signals": "0", "exit_r": ""},
    ]).to_csv(path, index=False)
    w = E.watchlist_rows({"2026-03-02": "normal", "2026-03-03": "normal"}, path)
    assert list(w.exit_r) == [2.0] and list(w.state) == ["normal"]


def test_the_frozen_study_rebuilds_the_studys_numbers():
    t = E.build_table(pd.read_csv(E.HISTORY), pd.read_csv(E.STUDY_DAYS),
                      pd.DataFrame(columns=E.NAME_COLS), pd.DataFrame(columns=["date", "n_flag", "state"]))
    b = t["states"]["breadth"]
    assert (b["n"], b["days"], b["mean_r"], b["win"]) == (295, 17, 1.202, 0.566)
    assert (b["entry_n"], b["entry_mean_r"]) == (433, 0.819) and t["relaxed"]["breadth"]["n"] == 221
    assert t["states"]["normal"]["days"] == 1080 and t["states"]["elevated"]["n"] == 203


def test_each_best_trade_gets_its_day_type():
    rows = [{"date": "2025-04-07"}, {"date": "2019-01-02"}]
    library.tag_days(rows, {"2025-04-07": (39, "breadth")})
    assert rows[0]["day_state"] == "breadth" and rows[0]["day_n"] == 39 and rows[1]["day_state"] is None
