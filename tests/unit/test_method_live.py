"""Method's real-time feed: bar summaries, the stored feed and its freshness, post
validation and the token, the server's /api/live, and the PC pusher (SHEL, the
network and git stubbed)."""


import pandas as pd
import pytest

from method import live
from method import pc_pusher as pp

ET = "US/Eastern"


def at(hhmm, day="2026-10-15"):
    return pd.Timestamp(f"{day} {hhmm}", tz=ET).to_pydatetime()


def _bars(rows, day="2026-10-15"):
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day} {t}", tz=ET) for t, *_ in rows])
    return pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                         "low": [r[3] for r in rows], "close": [r[4] for r in rows]}, index=idx)


# ---------------------------------------------------------------- summaries

def test_summary_splits_premarket_session_and_after_hours():
    b = _bars([("04:00", 99, 99, 95, 96), ("07:06", 96, 96, 90, 91), ("09:30", 94, 95, 93, 94),
               ("11:00", 94, 97, 88, 96), ("15:59", 96, 97, 95, 97), ("16:05", 97, 97, 92, 93)])
    s = live.summarize_bars(b, "2026-10-15")
    assert (s["price"], s["bar_time"]) == (93.0, "16:05")
    assert (s["pm_low"], s["pm_low_time"], s["pm_high"]) == (90.0, "07:06", 99.0)
    assert (s["open"], s["day_low"], s["day_low_time"], s["rth_close"]) == (94.0, 88.0, "11:00", 97.0)
    assert s["ah_low"] == 92.0
    early = live.summarize_bars(b, "2026-10-15", asof=at("08:00"))
    assert early["price"] == 91.0 and "open" not in early and "ah_low" not in early
    assert live.summarize_bars(b, "2026-10-15", asof=at("03:00")) is None
    assert live.summarize_bars(None, "2026-10-15") is None


# ---------------------------------------------------------------- the stored feed

def test_each_name_keeps_its_own_receive_time():
    feed = live.merge_feed(None, "2026-10-15", {"SOXL": {"price": 91.0}, "MXL": {"price": 70.0}}, at("08:00"))
    feed = live.merge_feed(feed, "2026-10-15", {"SOXL": {"price": 90.0}}, at("08:01"))     # active-only post
    fresh = live.fresh_names(feed, "2026-10-15", at("08:01:30"))
    assert fresh["SOXL"]["price"] == 90.0 and "MXL" in fresh
    assert list(live.fresh_names(feed, "2026-10-15", at("08:02:30"))) == ["SOXL"]         # MXL 150 s old
    assert live.fresh_names(feed, "2026-10-16", at("08:01:30")) == {}
    nxt = live.merge_feed(feed, "2026-10-16", {"KEEL": {"price": 2.5}}, at("04:00", "2026-10-16"))
    assert list(nxt["names"]) == ["KEEL"]                                                  # a new day starts clean


def test_posts_are_validated():
    day, names = live.clean_post({"day": "2026-10-15", "names": {
        "soxl": {"price": 91, "bar_time": "08:34", "junk": "x"},
        "BAD TICKER": {"price": 1.0}, "MXL": {"price": "70"}, "KEEL": {"price": True},
        "NBIS": {"price": 50.0, "pm_low": -3}}})
    assert day == "2026-10-15" and names == {"SOXL": {"price": 91.0, "bar_time": "08:34"}}
    with pytest.raises(ValueError):
        live.clean_post({"day": "yesterday", "names": {}})
    with pytest.raises(ValueError):
        live.clean_post({"day": "2026-10-15", "names": ["SOXL"]})


def test_token(tmp_path):
    path = tmp_path / "home" / "live_token"
    assert not live.token_ok(path, "anything")
    tok = live.ensure_token(path)
    assert live.ensure_token(path) == tok and len(tok) > 30
    assert live.token_ok(path, tok) and not live.token_ok(path, tok + "x") and not live.token_ok(path, None)


# ---------------------------------------------------------------- server

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    from method import server as sv
    calls = []
    monkeypatch.setattr(sv, "apply_live", lambda day, names: calls.append((day, names)) or
                        {"stored": len(names), "regraded": len(names), "alerts": []})
    app = sv.create_app()
    app.testing = True
    return app.test_client(), live.ensure_token(tmp_path / "home" / "live_token"), calls


def test_live_posts_need_the_token(client):
    c, tok, calls = client
    body = {"day": "2026-10-15", "names": {"SOXL": {"price": 91.0, "bar_time": "08:34"}}}
    assert c.post("/api/live", json=body, headers={"X-Method": "1"}).status_code == 403
    assert c.post("/api/live", json=body, headers={"X-Method": "1", "X-Method-Token": "nope"}).status_code == 403
    r = c.post("/api/live", json=body, headers={"X-Method": "1", "X-Method-Token": tok})
    assert r.get_json() == {"ok": True, "stored": 1, "regraded": 1, "alerts": []}
    assert calls == [("2026-10-15", {"SOXL": {"price": 91.0, "bar_time": "08:34"}})]
    bad = c.post("/api/live", json={"day": "x", "names": {}}, headers={"X-Method": "1", "X-Method-Token": tok})
    assert bad.status_code == 400


# ---------------------------------------------------------------- the PC pusher

def _ns(hhmm):
    return int(pd.Timestamp(f"2026-10-15 {hhmm}", tz=ET).tz_convert("UTC").value)


def test_shel_bars_are_restamped_to_the_bar_start():
    aggs = [{"close-time": _ns("09:30"), "open": 95.0, "high": 96.0, "low": 90.0, "close": 91.0, "volume": 5},
            {"close-time": _ns("09:31"), "open": 94.0, "high": 95.0, "low": 93.0, "close": 94.5, "volume": 9}]
    df = pp.bars_frame(aggs)
    assert [t.strftime("%H:%M") for t in df.index] == ["09:29", "09:30"]
    s = live.summarize_bars(df, "2026-10-15")
    assert (s["open"], s["pm_low"], s["price"], s["bar_time"]) == (94.0, 90.0, 94.5, "09:30")
    assert pp.bars_frame([]) is None


def test_active_names_are_the_live_ideas():
    snap = {"bounce": {"rows": [{"ticker": "SOXL", "rec": "GO"}, {"ticker": "MXL", "rec": "NO-GO", "report_rec": "GO"},
                                {"ticker": "INTC", "rec": "NO-GO"}]},
            "reversal": {"rows": [{"ticker": "PENG", "rec": "CAUTION"}]}}
    assert pp.pick_names(snap) == (["SOXL", "MXL", "PENG"], ["SOXL", "MXL", "INTC", "PENG"])


def test_one_pass_posts_the_whole_board(monkeypatch):
    monkeypatch.setattr(pp, "fetch_snapshot", lambda: {"bounce": {"rows": [{"ticker": "SOXL", "rec": "GO"},
                                                                           {"ticker": "INTC"}]}})
    monkeypatch.setattr(pp, "trillium_summaries", lambda tickers, day, rules_for=None:
                        {t: {"price": 1.0, "bar_time": "08:00"} for t in tickers})
    monkeypatch.setattr(pp, "read_token", lambda: "tok")
    monkeypatch.setattr(pp, "keep_awake", lambda on: None)
    posts = []
    monkeypatch.setattr(pp, "post", lambda day, names, token: posts.append((sorted(names), token)) or {"regraded": 2})
    assert pp.run_loop(once=True) == 0
    assert posts == [(["INTC", "SOXL"], "tok")]


@pytest.mark.parametrize("branch,dirty,updated", [("refs/heads/feat/x", "", False), ("", " M a.py", False),
                                                   ("", "", True)])
def test_self_update_only_moves_a_clean_detached_checkout(monkeypatch, branch, dirty, updated):
    calls = []

    def git(*args):
        calls.append(args)
        return {"symbolic-ref": branch, "status": dirty}.get(args[0]) or (
            "new" if args == ("rev-parse", "FETCH_HEAD") else "old" if args[0] == "rev-parse" else "")
    monkeypatch.setattr(pp, "_git", git)
    assert pp.self_update() is updated
    assert any(a[0] == "checkout" for a in calls) is updated
