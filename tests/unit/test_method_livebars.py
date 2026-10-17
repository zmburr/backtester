"""Method's live chart: the PC's Trillium 1-min bars, validated, kept per name (memory and
disk), served by /api/bars while the name streams, and replayed by the rules (review.py)."""

import json

import pandas as pd
import pytest

from method import live, livebars
from method import pc_pusher as pp

ET = "US/Eastern"
DAY = "2026-10-15"


def ep(hhmm, day=DAY):
    """A bar's start in UTC seconds."""
    return int(pd.Timestamp(f"{day} {hhmm}", tz=ET).timestamp())


def row(hhmm, o, h, l, c, v=1000):
    return [ep(hhmm), o, h, l, c, v]


# ---------------------------------------------------------------- validation

def test_posted_bars_are_validated():
    good = [row("09:30", 10, 11, 9.5, 10.5), row("09:31", 10.5, 10.6, 10.0, 10.1, 0)]
    bad = [row("09:32", 10, 9, 9.5, 10.5),               # high under the open
           [ep("09:33"), 10, 11, 9, 10, -5],              # negative volume
           [True, 10, 11, 9, 10, 5],                      # not an epoch
           [ep("09:34", "2026-10-14"), 10, 11, 9, 10, 5], # another day
           [ep("09:35"), 10, 11, 9, float("nan"), 5],
           [ep("09:36"), 10, 11, 9]]
    assert live.clean_bars(good + bad, DAY) == [[ep("09:30"), 10.0, 11.0, 9.5, 10.5, 1000],
                                               [ep("09:31"), 10.5, 10.6, 10.0, 10.1, 0]]
    assert live.clean_bars([row("09:30", 1, 1, 1, 1)] * 1001, DAY) is None
    assert live.clean_bars("x", DAY) is None and live.clean_bars(bad, DAY) is None
    day, names = live.clean_post({"day": DAY, "names": {"soxl": {"price": 10.1, "bars": good}}})
    assert names["SOXL"]["bars"] == live.clean_bars(good, DAY)


# ---------------------------------------------------------------- the store

def test_store_upserts_by_bar_time_and_appends_only_changes(tmp_path):
    st = livebars.Store(tmp_path)
    have = st.merge(DAY, {"SOXL": [row("09:30", 10, 11, 9, 10.5), row("09:31", 10.5, 10.7, 10.4, 10.6)]})
    assert have == {"SOXL": ep("09:31")}
    st.merge(DAY, {"SOXL": [row("09:31", 10.5, 10.7, 10.4, 10.6)]})          # same bar again: nothing new
    st.merge(DAY, {"SOXL": [row("09:31", 10.5, 10.9, 10.4, 10.8), row("09:32", 10.8, 10.8, 10.7, 10.7)]})
    rows = st.rows(DAY, "SOXL")
    assert [r[0] for r in rows] == [ep("09:30"), ep("09:31"), ep("09:32")] and rows[1][2] == 10.9
    lines = (tmp_path / DAY / "SOXL.jsonl").read_text().splitlines()
    assert len(lines) == 4                                 # 2 + the revised 9:31 + 9:32
    again = livebars.Store(tmp_path)                       # the server restarted
    assert again.rows(DAY, "SOXL") == rows and again.have(DAY) == {"SOXL": ep("09:32")}


def test_store_freshness_new_day_and_pruning(tmp_path):
    old = tmp_path / "2026-09-20"
    old.mkdir()
    (old / "X.jsonl").write_text(json.dumps(row("09:30", 1, 1, 1, 1, 1)) + "\n")
    st = livebars.Store(tmp_path)
    st.merge(DAY, {"SOXL": [row("09:30", 10, 11, 9, 10.5)]}, now_s=1000.0)
    assert not old.exists()                                # more than two weeks before the day
    assert st.fresh(DAY, "SOXL", now_s=1000.0 + livebars.FRESH_S)
    assert not st.fresh(DAY, "SOXL", now_s=1001.0 + livebars.FRESH_S)
    assert not st.fresh(DAY, "MXL", now_s=1000.0)
    assert st.have("2026-10-16") == {}                     # a new day starts clean


def test_chart_rows_today_candle_and_replay_frame():
    rows = [row("08:00", 9, 9.5, 8.8, 9.2, 50), row("09:30", 10, 11, 9, 10.5, 100),
            row("09:31", 10.5, 10.7, 10.4, 10.6, 200), row("09:32", 10.6, 12, 10.6, 11.5, 300)]
    cr = livebars.chart_rows(rows)
    assert [(b["time"] // 60) % 1440 for b in cr] == [480, 570, 571, 572]      # ET wall-clock minutes
    daily = livebars.with_today([{"time": "2026-10-14", "close": 9.0}, {"time": DAY, "close": 1.0}], cr, DAY)
    assert daily[-1] == {"time": DAY, "open": 10, "high": 12, "low": 9, "close": 11.5, "value": 600.0}
    assert len(daily) == 2
    # 9:33:10: the 9:32 minute closed, but its 2-min bar (9:32-9:34) hasn't: the replay leaves it out
    f = livebars.frame(rows, now=pd.Timestamp(f"{DAY} 09:33:10", tz=ET))
    assert f.index[-1].strftime("%H:%M") == "09:31" and "vwap" in f
    f = livebars.frame(rows, now=pd.Timestamp(f"{DAY} 09:35:00", tz=ET))      # a minute with no trades
    assert f.index[-1].strftime("%H:%M") == "09:32"
    f = livebars.frame(rows, now=pd.Timestamp(f"{DAY} 09:32:30", tz=ET))      # 9:32 still forming
    assert f.index[-1].strftime("%H:%M") == "09:31"
    assert livebars.frame([], now=None) is None


# ---------------------------------------------------------------- the PC

def _ns(hhmm):
    return int(pd.Timestamp(f"{DAY} {hhmm}", tz=ET).tz_convert("UTC").value)


def test_pusher_sends_the_bars_the_server_lacks(monkeypatch):
    aggs = [{"close-time": _ns(t), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": v}
            for t, v in (("09:31", 100), ("09:32", 200), ("09:33", 300))]
    bars = pp.bars_frame(aggs)
    assert list(bars["volume"]) == [100, 200, 300]
    assert [r[0] for r in pp.bar_rows(bars)] == [ep("09:30"), ep("09:31"), ep("09:32")]
    assert [r[0] for r in pp.bar_rows(bars, since=ep("09:31"))] == [ep("09:31"), ep("09:32")]
    assert pp.bar_rows(bars)[0] == [ep("09:30"), 10.0, 11.0, 9.0, 10.5, 100]
    monkeypatch.setattr(pp, "trillium_bars", lambda tickers, day: iter([("SOXL", bars)]))
    out = pp.trillium_summaries(["SOXL"], DAY, have={"SOXL": ep("09:32")})
    assert [r[0] for r in out["SOXL"]["bars"]] == [ep("09:32")]
    assert "bars" not in pp.trillium_summaries(["SOXL"], DAY)["SOXL"]          # bars off (an older server)
    assert len(pp.trillium_summaries(["SOXL"], DAY, have={})["SOXL"]["bars"]) == 3
    # what goes over the wire passes the server's checks
    assert live.clean_bars(json.loads(json.dumps(out["SOXL"]["bars"])), DAY) == out["SOXL"]["bars"]


def test_an_older_server_turns_the_bars_off_for_a_while(monkeypatch):
    seen, calls = [], {"n": 0}
    monkeypatch.setattr(pp, "fetch_snapshot", lambda: {"bounce": {"rows": [{"ticker": "SOXL", "rec": "GO"}]}})
    monkeypatch.setattr(pp, "trillium_summaries", lambda tickers, day, rules_for=None, have=None:
                        seen.append(have) or {t: {"price": 1.0, "bar_time": "09:30"} for t in tickers})
    monkeypatch.setattr(pp, "read_token", lambda: "tok")
    monkeypatch.setattr(pp, "keep_awake", lambda on: None)
    monkeypatch.setattr(pp, "post", lambda day, names, token: {"regraded": 1})
    assert pp.run_loop(once=True) == 0
    assert seen == [{}]                                    # the first pass sends every name's day


# ---------------------------------------------------------------- the server

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    from method import server as sv
    monkeypatch.setattr(sv, "LIVE_BARS", livebars.Store(tmp_path / "live_bars"))
    monkeypatch.setattr(sv, "apply_live", lambda day, names: {"stored": len(names), "regraded": 0, "alerts": [],
                                                              "names": names})
    monkeypatch.setattr(sv, "daily_rows", lambda t: [{"time": "2026-10-14", "open": 9, "high": 9, "low": 9,
                                                      "close": 9, "value": 1}])
    monkeypatch.setattr(sv, "chart_bars", lambda t: {"ticker": t, "daily": [], "intraday": [], "src": "P"})
    sv._bars_cache.clear()
    sv._daily_cache.clear()
    app = sv.create_app()
    app.testing = True
    return app.test_client(), live.ensure_token(tmp_path / "home" / "live_token"), sv


def test_posted_bars_feed_the_chart(client, monkeypatch):
    c, tok, sv = client
    today = pd.Timestamp.now(tz=ET).date().isoformat()
    bars = [[ep("09:30", today), 10, 11, 9, 10.5, 100], [ep("09:31", today), 10.5, 10.7, 10.4, 10.6, 200]]
    body = {"day": today, "names": {"SOXL": {"price": 10.6, "bar_time": "09:31", "bars": bars}}}
    r = c.post("/api/live", json=body, headers={"X-Method": "1", "X-Method-Token": tok}).get_json()
    assert r["have"] == {"SOXL": bars[-1][0]}
    assert "bars" not in r["names"]["SOXL"]                # the stored feed never holds bars
    d = c.get("/api/bars?ticker=SOXL").get_json()
    assert d["src"] == "T" and [b["value"] for b in d["intraday"]] == [100, 200]
    assert d["daily"][-1]["time"] == today and d["daily"][-1]["high"] == 11
    since = d["intraday"][-1]["time"]
    inc = c.get(f"/api/bars?ticker=SOXL&since={since}").get_json()
    assert inc["src"] == "T" and len(inc["intraday"]) == 1
    # the PC went quiet: back to Polygon, and an increment asks the page to keep what it has
    monkeypatch.setattr(livebars, "FRESH_S", -1)
    assert c.get("/api/bars?ticker=SOXL").get_json()["src"] == "P"
    assert c.get(f"/api/bars?ticker=SOXL&since={since}").get_json() == {
        "ticker": "SOXL", "src": "P", "intraday": [], "intraday_date": today}
    assert sv.live_frame("SOXL", today) is None


def test_todays_replay_runs_on_the_live_bars(client):
    c, tok, sv = client
    from method import review as rv
    today = pd.Timestamp.now(tz=ET).date().isoformat()
    assert rv.LIVE_FRAME is sv.live_frame
    sv.LIVE_BARS.merge(today, {"SOXL": [[ep("09:30", today), 10, 11, 9, 10.5, 100]]})
    f = sv.live_frame("SOXL", today)
    assert f is None or f.index[0].strftime("%H:%M") == "09:30"     # None only while 9:30 is still forming
    assert sv.live_frame("SOXL", "2026-10-14") is None              # past days replay from Polygon
