"""Method and stock splits (method/splits.py): SNDK's 8-for-1 on 11/20/2026 graded A+ off a
-89% "gap" because the session's statics were cached in old shares. No network."""

import datetime
import types

import pandas as pd
import pytest

from method import core
from method import refresh as rf
from method import splits
from tests.unit.test_method import LADDER, STATIC, _bars, at, env  # noqa: F401  (env is a fixture)

ET = "US/Eastern"
SNDK = {"SNDK": [{"date": "2026-11-20", "factor": 0.125, "ratio": "8-for-1"}]}


def daily(closes, end="2026-11-19"):
    idx = pd.bdate_range(end=end, periods=len(closes), tz=ET)
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.02, "low": c * 0.98, "close": c, "volume": 1e6})


# ---------------------------------------------------------------- the daily history

def test_old_share_history_goes_to_new_shares():
    hist = daily([797.24, 776.22, 755.16])
    out, note = splits.adjust("SNDK", hist, "2026-11-20", SNDK, raw=lambda t, d: 755.16)
    assert out["close"].iloc[-1] == pytest.approx(94.395)            # Polygon's adjusted close
    assert out["high"].iloc[0] == pytest.approx(797.24 * 1.02 / 8)
    assert out["volume"].iloc[-1] == pytest.approx(8e6)
    assert note == {"date": "2026-11-20", "factor": 0.125, "ratio": "8-for-1", "by": "method"}
    assert hist["close"].iloc[-1] == 755.16                          # the caller's frame is untouched


def test_history_polygon_already_adjusted_is_kept():
    hist = daily([99.655, 97.0275, 94.395])
    out, note = splits.adjust("SNDK", hist, "2026-11-20", SNDK, raw=lambda t, d: 755.16)
    assert out is hist and note["by"] == "polygon"


@pytest.mark.parametrize("ticker,end,session", [
    ("MU", "2026-11-19", "2026-11-20"),          # no split
    ("SNDK", "2026-11-20", "2026-11-23"),        # the split is already in the bars' past
])
def test_no_split_in_between_leaves_history_alone(ticker, end, session):
    hist = daily([100.0, 101.0], end=end)
    out, note = splits.adjust(ticker, hist, session, SNDK,
                              raw=lambda t, d: pytest.fail("no raw close needed"))
    assert out is hist and note is None


def test_unknown_raw_close_fails_the_static_instead_of_guessing():
    with pytest.raises(RuntimeError):
        splits.adjust("SNDK", daily([755.16]), "2026-11-20", SNDK, raw=lambda t, d: None)


def test_reverse_split_label_and_factor():
    client = types.SimpleNamespace(list_splits=lambda **k: [
        types.SimpleNamespace(ticker="ZZ", execution_date="2026-11-20", split_from=10, split_to=1)])
    assert splits.pending("2026-11-20", client=client)["ZZ"] == [
        {"date": "2026-11-20", "factor": 10.0, "ratio": "1-for-10"}]


def test_split_list_is_cached_and_survives_a_failed_call(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DIR", str(tmp_path))
    calls = []

    def ok(**k):
        calls.append(k)
        return [types.SimpleNamespace(ticker="SNDK", execution_date="2026-11-20", split_from=1, split_to=8)]

    def down(**k):
        raise ConnectionError("polygon down")
    t0 = pd.Timestamp("2026-11-19 17:00", tz=ET)
    assert splits.pending("2026-11-20", client=types.SimpleNamespace(list_splits=ok), now=t0) == SNDK
    assert calls[0]["execution_date_gte"] == "2026-11-10" and calls[0]["execution_date_lte"] == "2026-11-20"
    splits.pending("2026-11-20", client=types.SimpleNamespace(list_splits=ok), now=t0 + pd.Timedelta("10min"))
    assert len(calls) == 1                                           # inside the TTL
    stale = splits.pending("2026-11-20", client=types.SimpleNamespace(list_splits=down),
                           now=t0 + pd.Timedelta("2h"))
    assert stale == SNDK


# ---------------------------------------------------------------- prices before the split executes

def test_nearest_basis():
    assert splits.nearest(776.0, 94.395, 0.125) == pytest.approx(97.0)       # old-share print
    assert splits.nearest(90.0, 94.395, 0.125) == 90.0                       # already new shares
    assert splits.nearest(90.0, 755.16, 8.0) == 720.0                        # reversal row: old-share close
    assert splits.nearest(90.0, 94.4, None) == 90.0


def test_evening_summary_scaled_only_when_it_is_in_old_shares():
    s = {"prior_close": 94.395, "split": {"factor": 0.125, "date": "2026-11-20", "ratio": "8-for-1"}}
    b = {"price": 760.0, "rth_close": 755.16, "ah_low": 750.0, "bar_time": "17:30"}
    out = splits.to_session(b, s)
    assert out["price"] == pytest.approx(95.0) and out["ah_low"] == pytest.approx(93.75)
    assert out["rth_close"] == pytest.approx(94.395) and out["bar_time"] == "17:30"
    assert splits.to_session({"price": 95.0, "rth_close": 94.4}, s)["price"] == 95.0
    assert splits.to_session(b, {"prior_close": 755.16}) is b


def test_needs_rebuild():
    old = {"prior_close": 755.16}                                    # cached before this check existed
    assert splits.needs_rebuild("SNDK", old, "2026-11-20", SNDK)
    assert not splits.needs_rebuild("MU", old, "2026-11-20", SNDK)
    built = {"prior_close": 94.395, "last_bar": "2026-11-19", "split": {"date": "2026-11-20"}}
    assert not splits.needs_rebuild("SNDK", built, "2026-11-20", SNDK)
    missed = {"prior_close": 755.16, "last_bar": "2026-11-19"}       # built before Polygon listed it
    assert splits.needs_rebuild("SNDK", missed, "2026-11-20", SNDK)
    after = {"prior_close": 94.0, "last_bar": "2026-11-20"}           # built from post-split bars
    assert not splits.needs_rebuild("SNDK", after, "2026-11-23", SNDK)


def test_warning_text():
    assert splits.warning({"split": {"ratio": "8-for-1", "date": "2026-11-20"}}) == \
        "8-for-1 split 11/20: prior prices in new shares"
    assert splits.warning({}) is None


# ---------------------------------------------------------------- the refresh pass

def _split_env(env, closes, end, split_day, raw_close):
    mp = env["monkeypatch"]
    built = []

    def completed(t, session):
        built.append((t, session))
        return daily(closes, end=end)
    mp.setattr(rf, "completed_daily", completed)
    mp.setattr(rf, "static_from_hist", lambda hist: dict(STATIC, prior_close=float(hist["close"].iloc[-1])))
    mp.setattr(rf.splits, "pending", lambda session, **k: (
        {"SOXL": [{"date": split_day, "factor": 0.125, "ratio": "8-for-1"}]} if split_day <= session else {}))
    mp.setattr(rf.splits, "raw_close", lambda t, d: raw_close)
    return built


def test_premarket_rebuilds_a_static_cached_in_old_shares(env):
    """The SNDK morning: statics cached the evening before in old shares, the split executes
    at the open, the first print is in new shares."""
    day = "2026-10-15"
    built = _split_env(env, [820.0, 800.0], "2026-10-14", day, raw_close=800.0)
    rf.METHOD_DIR.mkdir(parents=True, exist_ok=True)
    core.write_json(rf._statics_path(day), {"SOXL": dict(STATIC, prior_close=800.0, ladder=dict(LADDER))})
    env["monkeypatch"].setattr(rf.pq, "get_intraday", lambda *a: _bars(day, [("07:06", 92, 92, 90, 91)]))
    snap = rf.run(now=at("08:35"))
    soxl = next(r for r in snap["bounce"]["rows"] if r["ticker"] == "SOXL")
    assert ("SOXL", day) in built
    assert soxl["prior_close"] == pytest.approx(100.0)               # 800 in new shares
    assert soxl["gap"] == pytest.approx(-0.09)                       # not -89%
    assert soxl["warnings"][0] == "8-for-1 split 10/15: prior prices in new shares"
    cached = core.read_json(rf._statics_path(day))["SOXL"]
    assert cached["split"]["by"] == "method" and cached["last_bar"] == "2026-10-14"
    n = len(built)
    rf.run(now=at("08:41"))
    assert len(built) == n                                           # rebuilt once, then cached


def test_evening_before_a_split_grades_after_hours_in_new_shares(env):
    built = _split_env(env, [820.0, 800.0], "2026-10-15", "2026-10-16", raw_close=800.0)
    env["monkeypatch"].setattr(rf.pq, "get_intraday", lambda *a: _bars("2026-10-15", [
        ("15:59", 801, 801, 800, 800), ("17:30", 790, 790, 776, 784)]))
    snap = rf.run(now=at("17:45"))
    assert snap["session"] == "2026-10-16"
    soxl = next(r for r in snap["bounce"]["rows"] if r["ticker"] == "SOXL")
    assert soxl["prior_close"] == pytest.approx(100.0)
    assert soxl["price"] == pytest.approx(98.0) and soxl["gap"] == pytest.approx(-0.02)
    assert soxl["ah_low"] == pytest.approx(97.0)
    assert built.count(("SOXL", "2026-10-16")) == 1                  # the drift check doesn't loop


@pytest.mark.parametrize("print_,expected", [(776.0, 97.0), (95.0, 95.0)])
def test_overnight_print_in_either_basis(env, print_, expected):
    import scanners.overnight_bounce_tracker as ot
    mp = env["monkeypatch"]
    _split_env(env, [820.0, 800.0], "2026-10-15", "2026-10-16", raw_close=800.0)
    ts = pd.Timestamp("2026-10-15 23:20", tz=ET)
    mp.setattr(ot, "yahoo_overnight_prices", lambda tickers: {"SOXL": (print_, ts)})
    mp.setattr(ot, "load_prices_file", lambda *a: {})
    snap = rf.run(now=at("23:30"))
    soxl = next(r for r in snap["bounce"]["rows"] if r["ticker"] == "SOXL")
    assert soxl["price"] == pytest.approx(expected)
    assert soxl["gap"] == pytest.approx(expected / 100.0 - 1)


def test_overnight_reversal_row_keeps_its_old_share_close():
    prev = {"reversal": {"rows": [{"ticker": "SNDK", "price": 760.0, "prior_close": 755.16, "gap": 0.0,
                                   "scenario": "B"}]}}
    prices = {"SNDK": (95.0, pd.Timestamp("2026-11-19 23:00", tz=ET), "Y")}
    r = rf.overnight_reversal(prev, prices, SNDK, "2026-11-20")[0]
    assert r["price"] == pytest.approx(760.0) and r["gap"] == pytest.approx(760 / 755.16 - 1)
    r = rf.overnight_reversal(prev, prices, {}, "2026-11-20")[0]
    assert r["price"] == 95.0                                        # no split: untouched
