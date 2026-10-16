"""Method's replays: the rules' trades on hand-built days (bounce: the morning
watcher's entry + trail exit; reversal: the study's entry + the cover matrix),
the ladder helpers, and the review endpoint's caching. No network."""

import datetime
import json

import pandas as pd
import pytest

from method import core
from method import ladder as ml
from method import refresh as rf
from method import review as rv

ET = "US/Eastern"
DAY = "2026-07-30"


def minute_bars(two_min, start="09:30", day=DAY):
    """[(o, h, l, c)] 2-min bars -> 1-min bars that aggregate back to exactly them."""
    rows, idx = [], []
    t0 = pd.Timestamp(f"{day} {start}", tz=ET)
    for k, (o, h, l, c) in enumerate(two_min):
        mid = (o + c) / 2
        t = t0 + pd.Timedelta(minutes=2 * k)
        rows += [(o, h, l, mid), (mid, max(mid, c), min(mid, c), c)]
        idx += [t, t + pd.Timedelta(minutes=1)]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx))
    df["volume"] = 1000.0
    df["vwap"] = df["close"]
    return df


def epoch(hhmm, day=DAY):
    return rv.et_epoch(pd.Timestamp(f"{day} {hhmm}", tz=ET))


# ---------------------------------------------------------------- bounce

BOUNCE_DAY = [
    (100, 100.5, 98, 98.5), (98.5, 98.6, 96, 96.2), (96.2, 96.3, 94, 94.1),
    (94.1, 94.2, 90, 90.5),                       # 9:36 the low of day
    (90.5, 91.0, 90.2, 90.8),
    (90.8, 92.5, 90.6, 92.4),                     # 9:40 closes above 91.0: buy 92.4, stop 90
    (92.4, 94.0, 92.3, 93.9), (93.9, 96.0, 93.8, 95.9),   # +1.5 ATR (95.4) reached: trail armed
    (95.9, 96.5, 95.5, 96.2),
    (96.2, 96.3, 95.0, 95.2),                     # 9:48 low under the prior bar's low 95.5: trail exit
    (95.2, 95.4, 95.0, 95.1), (95.1, 95.3, 95.0, 95.2),
]


def test_bounce_trades_follow_the_live_rules():
    trades = rv.bounce_trades(minute_bars(BOUNCE_DAY), atr=2.0, live=False)
    assert len(trades) == 1
    t = trades[0]
    assert (t["entry"], t["stop"], t["exit"], t["kind"]) == (92.4, 90.0, 95.5, "trail")
    assert t["entry_t"] == epoch("09:40") and t["exit_t"] == epoch("09:48")   # bar starts
    assert t["r"] == pytest.approx((95.5 - 92.4) / 2.4)


def test_bounce_trade_still_running_is_open_live_and_close_after():
    held = BOUNCE_DAY[:8]                                  # ends before the trail exit
    assert rv.bounce_trades(minute_bars(held), 2.0, live=True)[0]["kind"] == "open"
    assert rv.bounce_trades(minute_bars(held), 2.0, live=False)[0]["kind"] == "close"


# ---------------------------------------------------------------- reversal

def reversal_day():
    bars = [(100, 103, 99.8, 102.8),               # 9:30 runs to 103
            (102.8, 103.5, 102, 102.5),            # 9:32 high of day 103.5
            (102.5, 102.6, 101, 101.2),            # 9:34 closes under 102: short 101.2, stop 103.5
            (101.2, 101.3, 95, 95.5),
            (95.5, 95.6, 91.5, 92.0)]              # 9:38 through 92 (open - 4 ATR): cover half
    bars += [(92.2, 92.4, 92.0, 92.2)] * 85        # 9:40-12:28 drifts
    bars += [(92.2, 94.1, 92.1, 94.0)]             # 12:30 retraces 29% of the range: tripwire
    bars += [(94.0, 94.2, 93.8, 94.0)] * 10
    return minute_bars(bars)


def test_reversal_trades_follow_the_study_rule_and_cover_matrix():
    trades = rv.reversal_trades(reversal_day(), atr=2.0, live=False)
    assert len(trades) == 1
    t = trades[0]
    assert (t["entry"], t["stop"], t["kind"], t["exit"]) == (101.2, 103.5, "tripwire", 94.0)
    assert t["entry_t"] == epoch("09:34") and t["exit_t"] == epoch("12:30")
    assert t["half"] == {"t": epoch("09:38"), "price": 92.0}
    risk = 103.5 - 101.2
    assert t["r"] == pytest.approx(0.5 * (101.2 - 92.0) / risk + 0.5 * (101.2 - 94.0) / risk)


def test_reversal_stops_after_two_stop_outs():
    bars = [(100, 101, 99.5, 100.8), (100.8, 101.2, 100.5, 100.6),
            (100.6, 100.7, 100.0, 100.1),          # 9:34 short 100.1 (under 100.5), stop 101.2
            (100.1, 101.5, 100.0, 101.4),          # 9:36 stopped at 101.2
            (101.4, 101.45, 100.9, 101.0),         # 9:38 short 101.0 (under 101.4)? no: 101.0 > 100.0 prior low
            (101.0, 101.1, 99.0, 99.2),            # 9:40 short 99.2 (under 100.9), stop 101.5
            (99.2, 101.8, 99.1, 101.7),            # 9:42 stopped at 101.5: second stop-out
            (101.7, 101.8, 99.0, 99.1)]            # 9:44 would short again: locked
    bars += [(99.1, 99.2, 99.0, 99.1)] * 5
    trades = rv.reversal_trades(minute_bars(bars), atr=2.0, live=False)
    assert [t["kind"] for t in trades] == ["stop", "stop"]


# ---------------------------------------------------------------- ladder JSON

def test_next_better_skips_worse_and_higher_levels():
    levels = [{"price": 96.0, "gap": -0.04, "grade": "B"}, {"price": 92.0, "gap": -0.08, "grade": "A"},
              {"price": 85.0, "gap": -0.15, "grade": "A+"}]
    assert ml.next_better(levels, 94.0, "B")["grade"] == "A"
    assert ml.next_better(levels, 94.0, "B")["pct_away"] == pytest.approx(92 / 94 - 1)
    assert ml.next_better(levels, 90.0, "A")["grade"] == "A+"
    assert ml.next_better(levels, 80.0, "A+") is None


# ---------------------------------------------------------------- endpoint

@pytest.fixture
def client(tmp_path, monkeypatch):
    from method import server as sv
    import scanners.premarket_bounce_scanner as pbs
    bdir = tmp_path / "method"
    monkeypatch.setattr(rf, "METHOD_DIR", bdir)
    monkeypatch.setattr(sv, "REVIEW_DIR", bdir / "review")
    monkeypatch.setattr(sv, "OUTCOMES", tmp_path / "signal_outcomes.csv")
    sv._outcomes_cache.update(mtime=None, rows=[], index={})
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    monkeypatch.setattr(pbs, "get_cap", lambda t, c: "ETF")
    monkeypatch.setattr(pbs, "_load_json", lambda p: {})
    calls = []

    def fake_review(ticker, date, side, cap, outcome=None, live=False):
        calls.append((ticker, date, side, live, outcome))
        return {"ticker": ticker, "date": date, "side": side, "live": live, "candles": [{"time": 1}], "trades": []}
    monkeypatch.setattr(sv.rv, "review", fake_review)
    (tmp_path / "signal_outcomes.csv").write_text(
        "signal_date,session,target_date,ticker,bucket,cohort,d0_pct,d1_pct,mfe_atr_d0,exit_r,entry_r,atr_pct,entry_open\n"
        "2026-07-29,evening,2026-07-30,SNDK,bounce,surfaced,0.12,,2.1,3.45,4.0,0.08,1000\n")
    app = sv.create_app()
    app.testing = True
    return {"c": app.test_client(), "calls": calls, "dir": bdir / "review"}


def test_review_passes_the_scorecard_row_and_caches_past_days(client):
    c = client["c"]
    body = c.get("/api/review?side=bounce&ticker=sndk&date=2026-07-30").get_json()
    assert body["outcome"]["exit_r"] == 3.45 and body["outcome"]["atr_pct"] == 0.08
    ticker, date, side, live, outcome = client["calls"][0]
    assert (ticker, date, side, live) == ("SNDK", "2026-07-30", "bounce", False)
    assert outcome["entry_open"] == 1000
    assert (client["dir"] / "bounce_SNDK_2026-07-30.json").exists()
    c.get("/api/review?side=bounce&ticker=SNDK&date=2026-07-30")
    assert len(client["calls"]) == 1                         # served from the cache


def test_review_today_is_live_and_never_cached(client):
    today = pd.Timestamp.now(tz=ET).date().isoformat()
    c = client["c"]
    c.get(f"/api/review?side=reversal&ticker=PENG&date={today}")
    c.get(f"/api/review?side=reversal&ticker=PENG&date={today}")
    assert [call[3] for call in client["calls"]] == [True, True]
    assert not list(client["dir"].glob("*")) if client["dir"].exists() else True


def test_recent_day_without_a_scorecard_row_is_not_cached(client):
    recent = (pd.Timestamp.now(tz=ET).date() - datetime.timedelta(days=2)).isoformat()
    client["c"].get(f"/api/review?side=bounce&ticker=MXL&date={recent}")
    assert not (client["dir"] / f"bounce_MXL_{recent}.json").exists()


def test_review_validates(client):
    c = client["c"]
    assert c.get("/api/review?side=bounce&ticker=../x&date=2026-07-30").status_code == 400
    assert c.get("/api/review?side=momo&ticker=SNDK&date=2026-07-30").status_code == 400
    assert c.get("/api/review?side=bounce&ticker=SNDK&date=7/30").status_code == 400


def test_index_outcomes_prefers_morning():
    idx = core.index_outcomes([{"target_date": "d", "bucket": "bounce", "ticker": "X", "session": "evening", "v": 1},
                               {"target_date": "d", "bucket": "bounce", "ticker": "X", "session": "morning", "v": 2}])
    assert idx[("d", "bounce", "X")]["v"] == 2
    json.dumps(core.clean(idx[("d", "bounce", "X")]))
