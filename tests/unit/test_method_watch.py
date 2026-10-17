"""Method's entry watcher (method/watch.py): the 2-min entry rule on bars so far (it must
match the replay the edge numbers come from), the pushes on a breadth day, the dead-feed
push, the post validation, and the PC side (which names, when the passes start)."""

import datetime

import numpy as np
import pandas as pd
import pytest

from method import core, live, watch
from method import pc_pusher as pp

ET = "US/Eastern"
DAY = "2026-10-15"


def ts(hhmm, day=DAY):
    return pd.Timestamp(f"{day} {hhmm}", tz=ET)


def minute_bars(path, start="09:30", day=DAY):
    """1-min bars from (minutes from the start, close) anchors: closes move linearly between
    them with a small zigzag, so 2-min bars break their prior highs on the way up."""
    xs, ys = zip(*path)
    closes = np.interp(np.arange(xs[-1] + 1), xs, ys) + 0.08 * np.tile([1, -1], xs[-1] // 2 + 1)[:xs[-1] + 1]
    opens = np.r_[closes[0], closes[:-1]]
    idx = pd.date_range(ts(start, day), periods=len(closes), freq="1min")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 0.05,
                         "low": np.minimum(opens, closes) - 0.05, "close": closes}, index=idx)


# a flush to 90, a small bounce stopped by a flush to 88, then a run to 97 and a drop (the trail exit)
FLUSH = [(0, 100), (20, 90), (30, 91.5), (40, 88), (80, 97), (150, 97.2), (160, 95)]


def reference(bars, atr):
    """The replay: scripts/backfill_entry_r.rule_trades on the full day."""
    from bounce_entry_study.detectors import build_2min
    from scripts.backfill_entry_r import rule_trades
    b = bars.assign(volume=1.0, vwap=bars["close"])
    return rule_trades(build_2min(b), atr)


# ---------------------------------------------------------------- the rule on bars so far

def test_at_the_close_it_is_the_replay():
    bars = minute_bars(FLUSH)
    state = watch.rule_state(bars, 2.0, now=ts("12:15"))
    ref = reference(bars, 2.0)
    assert len(ref) == 2                                   # the stop-out and the re-entry
    got = [(e["t"], e["entry"], e["stop"], e["r"]) for e in state["entries"]]
    want = [(a.entry_time.strftime("%H:%M"), round(a.entry, 4), round(a.stop, 4), round(a.r, 2)) for a in ref]
    assert got == want
    assert [e["state"] for e in state["entries"]] == ["stopped", "out"]
    assert state["next"] is None                           # a winner banked ends the day


def test_entries_only_ever_add_up_over_the_morning():
    bars = minute_bars(FLUSH)
    final = [(e["t"], e["entry"], e["stop"]) for e in watch.rule_state(bars, 2.0, now=ts("12:15"))["entries"]]
    seen = []
    for m in range(2, 165):
        now = ts("09:30") + pd.Timedelta(minutes=m, seconds=3)
        state = watch.rule_state(bars[bars.index + pd.Timedelta(minutes=1) <= now], 2.0, now=now)
        got = [(e["t"], e["entry"], e["stop"]) for e in (state or {}).get("entries", [])]
        assert got == final[:len(got)]                     # never a different or vanished entry
        seen.append(len(got))
    assert seen == sorted(seen) and seen[-1] == 2


def test_an_open_entry_shows_r_now_and_where_the_trail_arms():
    bars = minute_bars(FLUSH)
    state = watch.rule_state(bars, 2.0, now=ts("10:44:03"))
    e = state["entries"][-1]
    assert e["n"] == 2 and e["state"] == "open"
    assert e["arm_at"] == round(e["entry"] + 1.5 * 2.0, 4)
    assert e["r"] == round((bars["close"][bars.index <= ts("10:43")].iloc[-1] - e["entry"]) / e["risk"], 2)
    assert e["armed"] == (bars["high"][(bars.index >= ts(e["t"])) & (bars.index <= ts("10:43"))].max() >= e["arm_at"])


def test_after_a_stop_out_it_says_what_the_re_entry_needs():
    bars = minute_bars(FLUSH)
    state = watch.rule_state(bars, 2.0, now=ts("10:11:03"))
    assert [e["state"] for e in state["entries"]] == ["stopped"]
    assert state["next"]["over"] > 0 and isinstance(state["next"]["fresh_low"], bool)


def test_a_half_formed_2min_bar_never_counts():
    bars = minute_bars(FLUSH)
    upto = bars[bars.index <= ts("09:40")]                 # the 9:40 minute is in, 9:41 isn't
    assert watch.two_min(upto, now=ts("09:41:05")).index[-1] == ts("09:38")
    assert watch.two_min(upto, now=ts("09:43:05")).index[-1] == ts("09:40")    # 9:41 had no trades
    with_41 = bars[bars.index <= ts("09:41")]
    assert watch.two_min(with_41, now=ts("09:41:30")).index[-1] == ts("09:38")  # 9:41 still forming
    assert watch.two_min(with_41, now=ts("09:42:03")).index[-1] == ts("09:40")
    assert watch.rule_state(bars, 2.0, now=ts("09:31:30")) is None


# ---------------------------------------------------------------- pushes (server)

def entry(n, t, e=12.40, s=11.80, state="open"):
    return {"n": n, "t": t, "entry": e, "stop": s, "risk": round(e - s, 4), "state": state, "r": 0.0}


def mb_for(band="16+", locked=True, state="breadth"):
    names, relaxed = ["MUU", "SNDU", "TINY"], ["CRWG"]
    return {"status": "ok", "session": DAY, "locked": locked, "state": state, "band": band,
            "n": 17 if band == "16+" else 12, "names": names, "relaxed": relaxed,
            "detail": {t: {"counted": t != "TINY"} for t in names + relaxed}}


ROWS = [{"ticker": "MUU", "score_open": 6, "atr": 1.0}, {"ticker": "SNDU", "score_open": 5, "atr": 2.0},
        {"ticker": "CRWG", "score_open": 4, "atr": 0.4}, {"ticker": "TINY", "score_open": 6, "atr": 0.1}]


@pytest.fixture
def pushes():
    sent = []

    def send(title, body, url="/", tag=""):
        sent.append({"title": title, "body": body, "url": url, "tag": tag})
        return 1
    send.sent = sent
    return send


def test_a_strong_day_pushes_both_entries_best_score_first_with_shares(tmp_path, pushes):
    names = {"MUU": {"rules": {"entries": [entry(1, "09:36", state="stopped"), entry(2, "09:42", 12.10, 11.50)]}},
             "SNDU": {"rules": {"entries": [entry(1, "09:42", 31.10, 29.95)]}},
             "CRWG": {"rules": {"entries": [entry(1, "09:40", 8.62, 8.31)]}},
             "TINY": {"rules": {"entries": [entry(1, "09:40", 5.0, 4.9)]}}}      # not counted: never
    path = tmp_path / f"entry_alerts_{DAY}.json"
    keys = watch.entry_alerts(DAY, mb_for(), ROWS, names, ts("09:42:03").to_pydatetime(), pushes, path)
    assert sorted(keys) == ["CRWG#1", "MUU#1", "MUU#2", "SNDU#1"]
    p = pushes.sent[0]
    assert p["title"] == "Strong breadth day · 4 entries" and p["url"] == "/#/bounce"
    assert p["body"].splitlines() == [
        "MUU 6/6 buy 12.40 stop 11.80 · 15,000 sh (3R)",       # 3R = $9k off 0.60 (over 0.5 ATR)
        "MUU re-entry buy 12.10 stop 11.50 · 15,000 sh (3R)",
        "SNDU 5/6 buy 31.10 stop 29.95 · 7,800 sh (3R)",       # off 1.15 (over 0.5 ATR = 1.00)
        "CRWG 4/6 buy 8.62 stop 8.31 · 29,000 sh (3R)",
    ]
    assert watch.entry_alerts(DAY, mb_for(), ROWS, names, ts("09:42:18").to_pydatetime(), pushes, path) == []
    assert len(pushes.sent) == 1                               # nothing twice


def test_an_11_to_15_day_pushes_first_entries_only_at_1r(tmp_path, pushes):
    names = {"MUU": {"rules": {"entries": [entry(1, "09:36", state="stopped"), entry(2, "09:42", 12.10, 11.50)]}}}
    keys = watch.entry_alerts(DAY, mb_for("11-15"), ROWS, names, ts("09:42:03").to_pydatetime(), pushes,
                              tmp_path / "a.json")
    assert keys == ["MUU#1"]                                   # the re-entry isn't alerted
    assert pushes.sent[-1]["title"] == "Breadth day · 1 entry" and pushes.sent[-1]["url"] == "/#/bounce/MUU"
    assert pushes.sent[-1]["body"] == "MUU 6/6 buy 12.40 stop 11.80 · 5,000 sh (1R)"


def test_nothing_before_the_lock_or_off_a_breadth_day(tmp_path, pushes):
    names = {"MUU": {"rules": {"entries": [entry(1, "09:36")]}}}
    now = ts("09:36:03").to_pydatetime()
    assert watch.entry_alerts(DAY, mb_for(locked=False), ROWS, names, now, pushes, tmp_path / "x.json") == []
    assert watch.entry_alerts(DAY, mb_for(state="elevated"), ROWS, names, now, pushes, tmp_path / "x.json") == []
    assert watch.entry_alerts("2026-10-16", mb_for(), ROWS, names, now, pushes, tmp_path / "x.json") == []
    assert pushes.sent == []
    # once it locks, an entry from before the lock still goes out if it's fresh
    assert watch.entry_alerts(DAY, mb_for(), ROWS, names, ts("09:38:03").to_pydatetime(), pushes,
                              tmp_path / "x.json") == ["MUU#1"]


def test_a_late_entry_shows_but_never_pushes(tmp_path, pushes):
    names = {"MUU": {"rules": {"entries": [entry(1, "09:36")]}}}
    path = tmp_path / "late.json"
    assert watch.entry_alerts(DAY, mb_for(), ROWS, names, ts("09:47:03").to_pydatetime(), pushes, path) == []
    assert pushes.sent == [] and core.read_json(path)["late"] == ["MUU#1"]


def test_an_entry_is_kept_only_once_a_phone_got_it(tmp_path):
    names = {"MUU": {"rules": {"entries": [entry(1, "09:36")]}}}
    path = tmp_path / "e.json"
    assert watch.entry_alerts(DAY, mb_for(), ROWS, names, ts("09:36:03").to_pydatetime(),
                              lambda *a, **k: 0, path) == []
    assert not path.exists()
    assert watch.entry_alerts(DAY, mb_for(), ROWS, names, ts("09:36:18").to_pydatetime(),
                              lambda *a, **k: 1, path) == ["MUU#1"]


def test_a_long_batch_fits_the_push_and_says_how_many_more():
    lines = [f"N{i:02d} 5/6 buy 123.45 stop 120.10 · 2,600 sh (3R)" for i in range(12)]
    body = watch.fit(lines)
    assert len(body) <= 240 and body.endswith("more on the board")
    shown = body.splitlines()[:-1]
    assert shown == lines[:len(shown)] and body.splitlines()[-1] == f"+{12 - len(shown)} more on the board"
    assert watch.fit(lines[:2]) == "\n".join(lines[:2])


def test_the_board_sizes_every_watched_entry_and_the_next_one():
    rows = [{"ticker": "MUU", "atr": 1.0, "rules": {"lod": 11.80, "entries": [entry(1, "09:36", state="stopped")],
                                                    "next": {"over": 12.40, "fresh_low": True}}},
            {"ticker": "TINY", "atr": 0.1, "rules": {"lod": 4.9, "entries": [], "next": {"over": 5.0, "fresh_low": True}}},
            {"ticker": "SOXL", "atr": 5.0}]
    watch.size_rows(rows, mb_for("16+"))
    ru = rows[0]["rules"]
    assert (ru["risk_r"], ru["alerted"]) == (3, 2)
    assert ru["entries"][0]["shares"] == 15000 and ru["next"]["shares"] == 15000      # 9000 / 0.60
    assert "risk_r" not in rows[1]["rules"] and "rules" not in rows[2]                 # not watched
    watch.size_rows(rows, mb_for("11-15"))
    assert rows[0]["rules"]["entries"][0]["shares"] == 5000 and rows[0]["rules"]["alerted"] == 1
    assert watch.size_rows(rows, mb_for(state="elevated")) is rows


def test_shares():
    assert watch.shares(12.40, 11.80, 0.60, 1.0, 9000) == 15000
    assert watch.shares(12.40, 11.80, 0.60, 2.0, 9000) == 9000          # half an ATR is the floor
    assert watch.shares(250.0, 248.0, 2.0, None, 3000) == 1500
    assert watch.shares(400.0, 380.0, 20.0, None, 3000) == 150
    assert watch.shares(10.0, 10.0, 0.0, None, 3000) == 0


# ---------------------------------------------------------------- a dead feed

def test_a_dead_feed_on_a_breadth_morning_pushes_once_and_rearms(tmp_path, pushes):
    path = tmp_path / "f.json"
    feed = {"day": DAY, "names": {"MUU": {"received_at": ts("09:40").isoformat()}}}
    now = ts("09:42").to_pydatetime()
    assert not watch.feed_down_alert(DAY, mb_for(), feed, now, pushes, path)        # 2 min old: fine
    now = ts("09:44").to_pydatetime()
    assert watch.feed_down_alert(DAY, mb_for(), feed, now, pushes, path)
    assert pushes.sent[0]["title"] == "Entry alerts paused" and "since 9:40" in pushes.sent[0]["body"]
    assert not watch.feed_down_alert(DAY, mb_for(), feed, ts("09:49").to_pydatetime(), pushes, path)
    back = {"day": DAY, "names": {"MUU": {"received_at": ts("09:50").isoformat()}}}
    assert not watch.feed_down_alert(DAY, mb_for(), back, ts("09:50:30").to_pydatetime(), pushes, path)
    assert watch.feed_down_alert(DAY, mb_for(), back, ts("09:55").to_pydatetime(), pushes, path)
    assert len(pushes.sent) == 2


def test_no_feed_push_off_a_breadth_day_or_outside_the_window(tmp_path, pushes):
    path = tmp_path / "f.json"
    assert not watch.feed_down_alert(DAY, mb_for(state="elevated"), None, ts("10:00").to_pydatetime(), pushes, path)
    assert not watch.feed_down_alert(DAY, mb_for(locked=False), None, ts("10:00").to_pydatetime(), pushes, path)
    assert not watch.feed_down_alert(DAY, mb_for(), None, ts("15:45").to_pydatetime(), pushes, path)
    assert watch.feed_down_alert(DAY, mb_for(), None, ts("10:00").to_pydatetime(), pushes, path)
    assert "this morning" in pushes.sent[0]["body"]


# ---------------------------------------------------------------- the post

def test_rule_state_survives_the_post_and_junk_is_dropped():
    bars = minute_bars(FLUSH)
    state = watch.rule_state(bars, 2.0, now=ts("10:44:03"))
    _, names = live.clean_post({"day": DAY, "names": {"MUU": {"price": 95.0, "rules": state}}})
    assert names["MUU"]["rules"]["entries"] == state["entries"]
    bad = dict(state, entries=[dict(state["entries"][0], state="maybe")])
    _, names = live.clean_post({"day": DAY, "names": {"MUU": {"price": 95.0, "rules": bad},
                                                      "SNDU": {"price": 30.0, "rules": "x"}}})
    assert "rules" not in names["MUU"] and "rules" not in names["SNDU"] and names["SNDU"]["price"] == 30.0


# ---------------------------------------------------------------- the PC side

SNAP = {"mode": "market",
        "bounce": {"rows": [{"ticker": "SOXL", "rec": "GO", "atr": 5.0}, {"ticker": "MUU", "rec": "NO-GO", "atr": 1.0},
                            {"ticker": "CRWG", "atr": 0.4}, {"ticker": "TINY", "atr": 0.1}]},
        "reversal": {"rows": []},
        "market_breadth": mb_for()}


def test_on_a_breadth_day_the_flagged_names_are_watched_every_15s():
    assert pp.rule_names(SNAP) == {"MUU": 1.0, "SNDU": None, "CRWG": 0.4}
    active, board = pp.pick_names(SNAP)
    assert active == ["SOXL", "MUU", "SNDU", "CRWG"] and set(active) <= set(board)
    assert pp.rule_names({**SNAP, "mode": "premarket"}) == {}
    assert pp.rule_names({**SNAP, "market_breadth": mb_for(state="elevated")}) == {}


def test_passes_start_just_after_each_15s_mark():
    base = datetime.datetime(2026, 10, 15, 13, 42, tzinfo=datetime.timezone.utc).timestamp()   # 9:42 ET
    assert pp.wait_s(base, 15) == pytest.approx(3.0)
    assert pp.wait_s(base + 2.5, 15) == pytest.approx(15.5)      # under 1 s away: the next mark
    assert pp.wait_s(base + 10, 15) == pytest.approx(8.0)
    assert pp.wait_s(base + 10, 60) == pytest.approx(53.0)


def test_summaries_carry_the_rule_only_for_watched_names(monkeypatch):
    bars = minute_bars(FLUSH)
    monkeypatch.setattr(pp, "trillium_bars", lambda tickers, day: ((t, bars) for t in tickers))
    out = pp.trillium_summaries(["MUU", "SOXL"], DAY, {"MUU": 2.0})
    assert "rules" in out["MUU"] and "rules" not in out["SOXL"]


# ---------------------------------------------------------------- the server's post handler

def test_a_post_with_new_entries_pushes_them(tmp_path, monkeypatch):
    from method import refresh as rf
    monkeypatch.setattr(rf, "METHOD_DIR", tmp_path)
    monkeypatch.setattr(rf, "SNAPSHOT", tmp_path / "snapshot.json")
    monkeypatch.setattr(rf, "LIVE", tmp_path / "live.json")
    monkeypatch.setattr(rf, "MARKET_BREADTH", tmp_path / "market_breadth.json")
    monkeypatch.setattr(rf.breadth, "evaluate", lambda *a, **k: mb_for())
    monkeypatch.setattr(rf.breadth, "edge_table", lambda: None)
    monkeypatch.setattr(rf, "breadth_alert", lambda mb: False)        # the day's own push is tested elsewhere
    sent = []
    monkeypatch.setattr(rf.push, "send", lambda title, body, url="/", tag="": sent.append(title) or 1)
    core.write_json(rf.SNAPSHOT, {"mode": "market", "session": DAY, "generated_at": f"{DAY}T09:30:00-04:00",
                                  "bounce": {"rows": [dict(r, rec="CAUTION", score=4, gap=-0.05) for r in ROWS]},
                                  "reversal": {"rows": []}})
    names = {"MUU": {"price": 12.5, "bar_time": "09:42", "rules": {"entries": [entry(1, "09:42")]}}}
    res = rf.apply_live(DAY, names, now=ts("09:42:05").to_pydatetime())
    assert "MUU#1" in res["alerts"] and sent == ["Strong breadth day · 1 entry"]
    assert core.read_json(rf.entry_alerts_path(DAY))["sent"] == ["MUU#1"]
