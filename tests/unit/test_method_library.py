"""Method's Best trades: reading your graded trades, the build (replays stubbed),
the shared replay cache rule, review()'s next day and daily-after, and /api/library."""

import datetime

import pandas as pd
import pytest

from method import core
from method import library as lib
from method import review as rv

ET = "US/Eastern"

BOUNCE_CSV = (
    "date,ticker,trade_grade,cap,Setup,gap_pct,bounce_open_high_pct,bounce_open_low_pct,bounce_open_close_pct,"
    "bounce_open_to_day_after_open_pct,time_of_low_price,time_of_bounce,consecutive_down_days,selloff_total_pct,"
    "pct_off_30d_high,atr_pct,spy_open_close_pct\n"
    "7/20/2026,ALAB,A,Medium,GapFade_weakstock,-0.12,0.24,-0.03,0.15,0.05,2026-07-20 09:46:00-04:00,"
    "2026-07-20 11:09:00-04:00,5,-0.4,-0.45,0.08,0.004\n"
    "7/28/2026,soxl ,B,ETF,GapFade_strongstock,-0.05,0.10,,0.06,,,,3,,,0.10,\n"
    "bad,XXX,A,ETF,,,,,,,,,,,,,\n"
    "7/20/2026,ALAB,A,Medium,,,,,,,,,,,,,\n")
REVERSAL_CSV = (
    "date,ticker,trade_grade,cap,setup,intraday_setup,gap_pct,reversal_open_low_pct,reversal_open_close_pct,"
    "reversal_open_post_low_pct,reversal_open_to_day_after_open_pct,time_of_high_price,time_of_reversal,time_of_low,"
    "consecutive_up_days,pct_from_9ema,atr_pct,spy_open_close_pct\n"
    "2/26/2026,PENG,A,Small,3DGapFade,,0.03,-0.20,-0.15,0.05,-0.03,2026-02-26 04:18:00-05:00,"
    "2026-02-26 09:30:00-05:00,,2,0.25,0.10,-0.006\n")


@pytest.fixture
def csvs(tmp_path, monkeypatch):
    b, r = tmp_path / "bounce.csv", tmp_path / "reversal.csv"
    b.write_text(BOUNCE_CSV)
    r.write_text(REVERSAL_CSV)
    monkeypatch.setattr(lib, "LIBRARY", tmp_path / "method" / "library.json")
    monkeypatch.setattr(lib, "REVIEW_DIR", tmp_path / "method" / "review")
    monkeypatch.setattr(lib, "OUTCOMES", tmp_path / "none.csv")
    return {"bounce": b, "reversal": r}


def test_trades_read_your_grade_move_and_log(csvs):
    b = lib.load_trades("bounce", csvs["bounce"])
    assert [t["key"] for t in b] == ["bounce:ALAB:2026-07-20", "bounce:SOXL:2026-07-28"]   # bad date + dup dropped
    alab = b[0]
    assert (alab["grade"], alab["cap"], alab["setup"]) == ("A", "Medium", "GapFade_weakstock")
    assert alab["move"] == pytest.approx(0.24) and alab["move_atr"] == pytest.approx(3.0)
    log = dict(alab["log"])
    assert log["Open → high"] == "+24.0%" and log["Open → low"] == "−3.0%" and log["Low at"] == "09:46"
    assert log["Bounce high at"] == "11:09" and log["Down days"] == "5" and log["ATR"] == "8.0%"
    assert "Open → low" not in dict(b[1]["log"])                       # empty cells are left out
    peng = lib.load_trades("reversal", csvs["reversal"])[0]
    assert peng["move"] == pytest.approx(0.20) and peng["move_atr"] == pytest.approx(2.0)   # fell 20%: +move
    assert dict(peng["log"])["High at"] == "04:18" and peng["setup"] == "3DGapFade"


def test_build_replays_once_and_keeps_csv_edits(csvs, monkeypatch):
    calls = []

    def fake(side, ticker, date, review_dir, cap=None, row=None, today=None):
        calls.append((side, ticker, cap))
        if ticker == "SOXL":
            return {"candles": [], "error": "no bars", "trades": []}
        return {"candles": [{"time": 1}], "trades": [{"r": 2.0}, {"r": -1.0}], "rules_r": 1.0,
                "at_open": {"grade": "A+", "score": 6, "rec": "GO", "intensity": 80.0, "gap": -0.12},
                "ladder": {"setup_type": "GapFade_weakstock"}}
    monkeypatch.setattr(lib, "cached_review", fake)
    out = lib.build(pause=0, sources=csvs)
    assert out == {"rows": 3, "built": 3, "failed": ["bounce:SOXL:2026-07-28"]}
    assert ("bounce", "ALAB", "Medium") in calls                        # the cap you logged, not today's
    rows = {r["key"]: r for r in core.read_json(lib.LIBRARY)["rows"]}
    alab = rows["bounce:ALAB:2026-07-20"]
    assert (alab["sys_grade"], alab["rules_r"], alab["n_trades"], alab["ok"]) == ("A+", 1.0, 2, True)
    assert rows["reversal:PENG:2026-02-26"]["sys_scenario"] == "C"     # opened -12%: below +4%
    calls.clear()
    csvs["bounce"].write_text(BOUNCE_CSV.replace("7/20/2026,ALAB,A,", "7/20/2026,ALAB,B,", 1))
    lib.build(pause=0, sources=csvs)
    assert calls == [("bounce", "SOXL", "ETF")]                         # only the failed row retried
    assert {r["key"]: r for r in core.read_json(lib.LIBRARY)["rows"]}["bounce:ALAB:2026-07-20"]["grade"] == "B"


def test_cached_review_caches_only_current_complete_days(tmp_path, monkeypatch):
    calls = []

    def fake_review(ticker, date, side, cap, outcome=None, live=False):
        calls.append(outcome)
        return {"v": rv.VERSION, "candles": [{"time": 1}], "complete": date != "2026-10-14"}
    monkeypatch.setattr(rv, "review", fake_review)
    today = datetime.date(2026, 10, 16)
    d = tmp_path / "review"
    lib.cached_review("bounce", "SOXL", "2026-07-28", d, cap="ETF", today=today)
    lib.cached_review("bounce", "SOXL", "2026-07-28", d, cap="ETF", today=today)
    assert len(calls) == 1                                              # old + complete: cached
    lib.cached_review("bounce", "MXL", "2026-10-14", d, cap="Medium", row={"exit_r": "1.5"}, today=today)
    assert not (d / "bounce_MXL_2026-10-14.json").exists()             # next day not over yet
    assert calls[-1]["exit_r"] == 1.5
    core.write_json(d / "bounce_KEEL_2026-07-01.json", {"v": rv.VERSION - 1, "candles": [{"time": 1}]})
    lib.cached_review("bounce", "KEEL", "2026-07-01", d, cap="Small", today=today)
    assert core.read_json(d / "bounce_KEEL_2026-07-01.json")["v"] == rv.VERSION      # old shape rebuilt


# ---------------------------------------------------------------- review(): next day + daily after

def _minutes(day, start="09:30", n=10, base=100.0):
    idx = pd.date_range(f"{day} {start}", periods=n, freq="1min", tz=ET)
    return pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base + 0.5,
                         "volume": 1000.0, "vwap": base + 0.2}, index=idx)


@pytest.fixture
def stub_review(monkeypatch):
    hist = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6},
                        index=pd.date_range("2026-01-01", periods=200, freq="B", tz=ET))
    monkeypatch.setattr(rv, "completed_daily", lambda t, d: hist)
    monkeypatch.setattr(rv, "static_from_hist", lambda h: {"prior_close": 100.0, "atr": 2.0, "mid_bb": 110.0})
    monkeypatch.setattr(rv, "ladder_payload", lambda s, cap, t: {"levels": [], "zones": []})
    monkeypatch.setattr(rv, "grade_at", lambda s, cap, p, t: {"grade": "B", "score": 4, "rec": "CAUTION", "intensity": 50})
    monkeypatch.setattr(rv, "bounce_trades", lambda bars, atr, live: [])
    monkeypatch.setattr(rv, "after_days", lambda t, d: [{"time": d}] * 6)
    monkeypatch.setattr(rv, "next_trading_day", lambda d: d + datetime.timedelta(days=1))
    fetched = []
    monkeypatch.setattr(rv, "day_bars", lambda t, d, live: fetched.append(d) or _minutes(d))
    return fetched


def test_review_adds_the_next_day_once_it_is_over(stub_review):
    d = rv.review("SOXL", "2026-07-28", "bounce", "ETF")
    assert d["v"] == rv.VERSION and d["complete"] is True
    assert len(d["candles"]) == 10 and d["candles"][0]["vw"] == pytest.approx(100.2)   # 1-min, bar VWAP
    assert d["next_date"] == "2026-07-29" and len(d["next_candles"]) == 10
    assert len(d["daily"]) == 180 + 6                                   # 180 before + the trade day on
    yesterday = (pd.Timestamp.now(tz=ET).date() - datetime.timedelta(days=1)).isoformat()
    stub_review.clear()
    y = rv.review("SOXL", yesterday, "bounce", "ETF")
    assert stub_review == [yesterday] and y["next_candles"] == [] and y["complete"] is False


# ---------------------------------------------------------------- endpoint

def test_library_endpoint_serves_the_built_list(csvs, monkeypatch):
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    from method import server as sv
    core.write_json(lib.LIBRARY, {"built_at": "x", "rows": [{"key": "bounce:ALAB:2026-07-20"}]})
    app = sv.create_app()
    app.testing = True
    assert app.test_client().get("/api/library").get_json()["rows"] == [{"key": "bounce:ALAB:2026-07-20"}]


def test_live_chart_falls_back_a_day_after_midnight(monkeypatch):
    """Polygon's delayed plan refuses a range ending on a day that hasn't traded (00:00-04:00)."""
    from method import server as sv
    today = pd.Timestamp.now(tz=ET).date()
    ends = []

    def levels(t, end, window, mult, span):
        ends.append(end)
        if end == today.isoformat():
            raise RuntimeError("NOT_AUTHORIZED")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
                            index=pd.DatetimeIndex([pd.Timestamp(end, tz=ET)]))

    def intraday(t, day, mult, span):
        if day == today.isoformat():
            raise RuntimeError("NOT_AUTHORIZED")
        return _minutes(day)
    monkeypatch.setattr(sv.pq, "get_levels_data", levels)
    monkeypatch.setattr(sv.pq, "get_intraday", intraday)
    d = sv.chart_bars("SOXL")
    yesterday = (today - datetime.timedelta(days=1)).isoformat()
    assert ends == [today.isoformat(), yesterday] and d["daily"][-1]["time"] == yesterday
    assert d["intraday_date"] == yesterday and len(d["intraday"]) == 10
