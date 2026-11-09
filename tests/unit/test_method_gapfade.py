"""Method's 3-day gap fade check (method/gapfade.py): the strict streaks, the status at the
open, the lists, the pushes, the market up-side build and a refresh pass. No network."""

import datetime
import json

import pandas as pd
import pytest

from method import breadth
from method import gapfade as gf
from method import refresh as rf
from method import watch

from tests.unit.test_method import LADDER, STATIC, at, env  # noqa: F401  (env is a fixture)

# SOXL daily closes 10/28 - 11/04/2026: 11/03 closed +7.3%, so 11/05 came in on ONE down close
# (the scorer's forgiving count reads 5)
SOXL_INTO_1105 = [107.35, 102.39, 77.05, 68.17, 73.12, 66.68]


# ---------------------------------------------------------------- the rule

def test_streaks_are_strict():
    assert gf.streaks(SOXL_INTO_1105) == (1, 0)
    assert gf.streaks(SOXL_INTO_1105 + [40.87]) == (2, 0)          # into 11/06
    assert gf.streaks([10, 11, 12, 13, 13.5]) == (0, 4)
    assert gf.streaks([10, 10, 10]) == (0, 0)                       # flat resets both
    assert gf.streaks([]) == (0, 0)


@pytest.mark.parametrize("side,streak,prior,open_,want", [
    ("long", 1, 66.68, 56.95, "off"),         # SOXL 11/05: gapped -14.6% but only one down close
    ("long", 2, 40.87, None, "setting_up"),   # SOXL the night of 11/05
    ("long", 2, 40.87, 48.785, "no_gap"),     # ... and it opened +19% on 11/06
    ("long", 2, 422.67, 408.66, "on"),        # QQQ 4/7/25
    ("short", 4, 100.0, 103.0, "on"),         # GLD 1/29/26: 4 up closes, +3%
    ("short", 3, 100.0, 99.0, "no_gap"),
    ("short", 1, 100.0, 105.0, "off"),
    ("long", None, 100.0, 95.0, "unknown"),
])
def test_status(side, streak, prior, open_, want):
    assert gf.status(side, streak, prior, open_) == want


def test_ladder_prices_cap_at_the_prior_close():
    s = dict(STATIC, ladder=LADDER)                              # B from 96, A from 92, prior 100
    assert gf.ladder_prices(s) == (96.0, 92.0)
    zones = [{"grade": "B", "top": 102.0, "bottom": 95.0}, {"grade": "A", "top": 95.0, "bottom": 60.0}]
    assert gf.ladder_prices(dict(STATIC, ladder={"zones": zones})) == (100.0, 95.0)   # B on any gap down
    assert gf.ladder_prices(dict(STATIC, ladder={"zones": [{"grade": "C", "top": 102.0, "bottom": 60.0}]})) == (None, None)


# ---------------------------------------------------------------- the lists

def _cand(t, prior=50.0, flag=48.0, adv=300e6, down=3, **kw):
    return {"ticker": t, "prior_close": prior, "flag_price": flag, "adv_dollars": adv, "down": down,
            "cap": "Medium", "etf": False, **kw}


def test_long_items_merge_statics_and_rank_ready_first():
    cands = {"MKT": _cand("MKT"), "THIN": _cand("THIN", adv=20e6), "RELAX": _cand("RELAX", flag=None)}
    statics = {
        "SOXL": dict(STATIC, ladder=LADDER, down_streak=2, up_streak=0),   # A 92 within 1 ATR (8)
        "MXL": dict(STATIC, ladder=LADDER, down_streak=1, up_streak=0),    # no structure
        "MKT": dict(STATIC, prior_close=50.0, atr=1.0, ladder=LADDER, down_streak=3, up_streak=0),
    }
    items = gf.long_items(cands, statics, watchlist=["SOXL", "MXL"], pinned=[])
    names = [it["ticker"] for it in items]
    assert "MXL" not in names and "RELAX" not in names          # one down close; B only when relaxed
    assert names[:2] == ["MKT", "SOXL"]                          # A-ready first, the nearest A leading
    assert names[-1] == "THIN"                                   # not ready, not liquid
    soxl = items[1]
    assert soxl["a_ready"] and soxl["b_price"] == 96.0 and soxl["a_price"] == 92.0 and soxl["src"] == "watchlist"
    mkt = next(it for it in items if it["ticker"] == "MKT")
    # the ladder caps at the prior close: LADDER's A zone tops at 92, above MKT's 50 close
    assert mkt["a_price"] == 50.0 and mkt["a_ready"] and mkt["adv"] == 300e6
    thin = next(it for it in items if it["ticker"] == "THIN")
    assert thin["a_price"] is None and not thin["a_ready"]


def test_short_items_market_build_and_your_stretched_names():
    cands = {"PARA": {"prior_close": 100.0, "up": 3, "ext9": 0.25, "atr": 5.0, "adv_dollars": 300e6}}
    rows = [{"ticker": "GLD", "prior_close": 380.0, "pct_from_9ema": 0.06, "atr_pct": 0.02, "price": 381.0},
            {"ticker": "FLAT", "prior_close": 50.0, "pct_from_9ema": 0.02, "atr_pct": 0.05, "price": 50.0},
            {"ticker": "ONEUP", "prior_close": 50.0, "pct_from_9ema": 0.2, "atr_pct": 0.05, "price": 50.0}]
    streak = {"GLD": (0, 4), "FLAT": (0, 3), "ONEUP": (0, 1)}
    items = gf.short_items(cands, {}, rows, lambda t: streak.get(t), watchlist=["GLD", "FLAT", "ONEUP"], pinned=[])
    by = {it["ticker"]: it for it in items}
    assert set(by) == {"PARA", "GLD"}                            # FLAT not stretched, ONEUP one up close
    assert by["PARA"]["a_price"] == pytest.approx(104.0) and by["PARA"]["a_ready"]   # +4% inside 1 ATR
    assert by["GLD"]["streak"] == 4 and not by["GLD"]["a_ready"]  # +4% is 2 ATRs for GLD


# ---------------------------------------------------------------- the open

def test_annotate_and_tag_rows_at_the_open():
    block = {"session": "2026-11-06", "long": [
        {"ticker": "SOXL", "side": "long", "streak": 2, "prior_close": 40.87, "b_price": 40.87, "a_price": 38.0},
        {"ticker": "MU", "side": "long", "streak": 2, "prior_close": 650.0, "b_price": 640.0, "a_price": 600.0}],
        "short": [{"ticker": "PARA", "side": "short", "streak": 3, "prior_close": 100.0, "a_price": 104.0}]}
    bounce = [{"ticker": "SOXL", "price": 49.0, "open": 48.785, "prior_close": 40.87, "grade_open": "-"},
              {"ticker": "AMD", "price": 200.0, "open": 195.0, "prior_close": 210.0}]
    feed = {"MU": {"price": 630.0, "open": 635.0}, "PARA": {"price": 105.0, "open": 105.5}}
    gf.annotate(block, "market", bounce, [], feed, None, None, grade_of=lambda it, o: "A")
    soxl, mu = block["long"]
    assert soxl["status"] == "no_gap" and soxl["open_gap"] == pytest.approx(48.785 / 40.87 - 1)
    assert mu["status"] == "on" and mu["grade_open"] == "A"
    para = block["short"][0]
    assert para["status"] == "on" and para["grade_open"] == "A"   # +5.5%: the reversal A
    gf.tag_rows(bounce, "long", block, lambda t: {"AMD": (1, 0)}.get(t), "market")
    assert bounce[0]["gapfade"] == "no_gap" and bounce[1]["gapfade"] == "off"
    # a live re-grade keeps the pass's own count
    bounce[1]["open"] = None
    gf.tag_rows(bounce, "long", block, None, "premarket")
    assert bounce[1]["gapfade"] == "off" and bounce[1]["gapfade_streak"] == 1


def test_annotate_carries_the_open_forward_and_reads_the_premarket():
    item = {"ticker": "MU", "side": "long", "streak": 2, "prior_close": 650.0, "b_price": 640.0, "a_price": 600.0}
    block = {"session": "2026-11-06", "long": [dict(item)], "short": []}
    prev = {"session": "2026-11-06", "long": [dict(item, open=630.0)], "short": []}
    gf.annotate(block, "market", [], [], {}, {}, prev)
    assert block["long"][0]["open"] == 630.0 and block["long"][0]["status"] == "on"
    pre = {"session": "2026-11-06", "long": [dict(item)], "short": []}
    gf.annotate(pre, "premarket", [], [], {"MU": {"price": 640.0}}, {}, None)
    assert pre["long"][0]["status"] == "setting_up" and pre["long"][0]["would_trigger"] is True


# ---------------------------------------------------------------- pushes

def _long(t, adv=2e9, ready=True, prior=100.0, a=95.0):
    return {"ticker": t, "side": "long", "streak": 2, "prior_close": prior, "b_price": prior, "a_price": a,
            "a_ready": ready, "src": "market", "adv": adv}


def _ready_block(session="2026-11-06"):
    """A flush setting up: three A-ready longs at $1B+ a day, one thin one, one stretched short."""
    return {"session": session, "built": True,
            "long": [_long("SOXL", prior=40.87, a=38.5), _long("MU", prior=650.0, a=620.0), _long("SNDK"),
                     _long("XYZ", adv=50e6)],
            "short": [{"ticker": "PARA", "side": "short", "streak": 3, "prior_close": 100.0, "b_price": 100.0,
                       "a_price": 104.0, "a_ready": True, "src": "market", "adv": 3e8}]}


def test_push_names_needs_a_cluster_or_a_pin():
    blk = _ready_block()
    cluster, pins = gf.push_names(blk, pinned=["PARA"])
    assert [it["ticker"] for it in cluster] == ["SOXL", "MU", "SNDK"]
    assert [it["ticker"] for it in pins] == ["PARA"]
    blk["long"] = blk["long"][:2] + blk["long"][3:]                 # two A-ready names: not a cluster
    assert gf.push_names(blk) == ([], [])


def test_evening_push_once_and_only_when_switched_on(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DIR", str(tmp_path))
    monkeypatch.delenv("GAPFADE_PUSH", raising=False)
    sent = []
    send = lambda title, body, url="/", tag="": sent.append((title, body, tag)) or 1
    now = at("17:50", day="2026-11-05")
    assert not gf.evening_push(_ready_block(), "2026-11-06", "evening", now, send, built=True)   # off by default
    (tmp_path / "gapfade_push.on").write_text("")
    assert not gf.evening_push(_ready_block(), "2026-11-06", "evening", at("17:00", day="2026-11-05"), send, built=False)
    blk = _ready_block()
    assert gf.evening_push(blk, "2026-11-06", "evening", now, send, built=True)
    title, body, tag = sent[0]
    assert title == "Gap fade cluster setting up · Fri 11/6" and tag == "gapfade-eve-2026-11-06"
    assert "SOXL long: below 40.87 triggers, A 38.50" in body and "MU long" in body
    assert "XYZ" not in body and "PARA" not in body and "(5 setting up on the board)" in body
    assert blk["pushed"]
    state = json.loads((tmp_path / "gapfade_push_2026-11-06.json").read_text())
    assert state["evening_names"] == ["SOXL", "MU", "SNDK"]
    assert not gf.evening_push(_ready_block(), "2026-11-06", "evening", now, send, built=True)   # once a session
    assert len(sent) == 1
    # no cluster, a pinned short: the pin alone pushes, and a build that never lands still pushes by 7:30
    lone = _ready_block("2026-11-09")
    lone["long"] = lone["long"][3:]
    assert not gf.evening_push(lone, "2026-11-09", "evening", at("19:31", day="2026-11-06"), send, built=False)
    assert gf.evening_push(lone, "2026-11-09", "evening", at("19:31", day="2026-11-06"), send, built=False,
                           pinned=["PARA"])
    assert sent[-1][0] == "Gap fade setting up · Mon 11/9" and "PARA short: above 100.00, A 104.00" in sent[-1][1]


def test_evening_push_needs_a_phone(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DIR", str(tmp_path))
    assert not gf.evening_push(_ready_block(), "2026-11-06", "evening", at("17:50"), lambda *a, **k: 0,
                               built=True, enabled=True)
    assert not (tmp_path / "gapfade_push_2026-11-06.json").exists()      # retried on the next pass


def test_open_pushes_the_evening_names_and_pins_that_open_on(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DIR", str(tmp_path))
    (tmp_path / "gapfade_push_2026-11-06.json").write_text(json.dumps({"evening": "x", "evening_names": ["SOXL", "MU"]}))
    sent = []
    send = lambda title, body, url="/", tag="": sent.append((title, body, url)) or 1
    blk = _ready_block()
    blk["long"][0].update(status="on", open=36.0, open_gap=36.0 / 40.87 - 1, grade_open="A")
    blk["long"][1].update(status="on", open=600.0, open_gap=-0.08, grade_open="C")    # under B: quiet
    blk["long"][2].update(status="on", open=90.0, open_gap=-0.1, grade_open="A")      # not in the evening push
    blk["short"][0].update(status="on", open=105.0, open_gap=0.05, grade_open="A")    # pinned
    got = gf.open_pushes(blk, "2026-11-06", "market", send, pinned=["PARA"], enabled=True)
    assert got == ["SOXL", "PARA"]
    assert sent[0][0] == "SOXL gap fade ON · A" and "2 straight down closes, gapped down" in sent[0][1]
    assert sent[1][2] == "/#/reversal/PARA"
    assert gf.open_pushes(blk, "2026-11-06", "market", send, pinned=["PARA"], enabled=True) == []   # once a name


def test_entry_line_says_off_playbook():
    e = {"n": 1, "entry": 10.0, "stop": 9.5, "t": "09:40"}
    on = watch.entry_line("MUU", e, {"score_open": 5, "gapfade": "on", "atr": 1.0}, "16+")
    off = watch.entry_line("MUU", e, {"score_open": 5, "gapfade": "no_gap", "atr": 1.0}, "16+")
    assert on.startswith("MUU 5/6") and off.startswith("OFF-PLAYBOOK MUU 5/6")


# ---------------------------------------------------------------- the market up-side build

def _long_frame(series: dict, start="2026-09-01"):
    days = pd.bdate_range(start, periods=len(next(iter(series.values()))))
    rows = []
    for t, closes in series.items():
        for d, c in zip(days, closes):
            rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": t, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 2_000_000})
    return pd.DataFrame(rows), days


def test_build_up_candidates_keeps_stretched_up_runs():
    flat = [20.0] * 25
    run = flat[:-3] + [22.0, 25.0, 29.0]                       # 3 up closes, far over the 9EMA
    mild = flat[:-2] + [20.1, 20.2]                            # 2 up closes, barely stretched
    down = flat[:-2] + [19.0, 18.0]
    long, days = _long_frame({"RUN": run, "MILD": mild, "DOWN": down})
    session = (days[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    out = breadth.build_up_candidates(session, long)
    assert set(out) == {"RUN"}
    c = out["RUN"]
    assert c["up"] == 3 and c["prior_close"] == 29.0 and c["ext9"] > 0.10 and c["adv_dollars"] > 0


def test_build_writes_the_short_side_even_when_the_long_side_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(breadth, "METHOD_DIR", tmp_path)
    session = "2026-11-06"
    (tmp_path / f"breadth_candidates_{session}.json").write_text(json.dumps(
        {"session": session, "missing_days": [], "candidates": {}}))
    long, _ = _long_frame({"RUN": [20.0] * 22 + [22.0, 25.0, 29.0]})
    monkeypatch.setattr(breadth, "window", lambda s: sorted(long["date"].unique()))
    monkeypatch.setattr(breadth, "update_store", lambda days: [])
    monkeypatch.setattr(breadth, "load_store", lambda days: long)
    monkeypatch.setattr(breadth, "split_tickers", lambda since: set())
    monkeypatch.setattr(breadth, "etf_tickers", lambda: set())
    breadth.build(session)
    up = breadth.load_short_candidates(session)
    assert set(up["candidates"]) == {"RUN"}
    assert json.loads((tmp_path / f"breadth_candidates_{session}.json").read_text())["candidates"] == {}   # untouched


# ---------------------------------------------------------------- a refresh pass

def test_evening_pass_lists_soxl_setting_up(env):
    mp = env["monkeypatch"]
    falling = [120.0] * 18 + [110.0, 100.0]                    # two straight down closes into tomorrow

    def daily(t, session):
        return pd.DataFrame({"close": falling if t == "SOXL" else [100.0] * 20})
    mp.setattr(rf, "completed_daily", daily)
    mp.setattr(rf.pq, "get_intraday", lambda *a: None)
    snap = rf.run(now=at("17:45"))
    g = snap["gapfade"]
    assert g["session"] == "2026-10-16" and g["mode"] == "evening" and g["built"] is False
    assert [it["ticker"] for it in g["long"]] == ["SOXL"]
    soxl = g["long"][0]
    assert soxl["status"] == "setting_up" and soxl["streak"] == 2 and soxl["a_ready"]
    assert soxl["b_price"] == 96.0 and soxl["a_price"] == 92.0
    assert g["short"] == []
    statics = json.loads((env["dir"] / "method" / "statics_2026-10-16.json").read_text())
    assert statics["SOXL"]["down_streak"] == 2 and statics["MXL"]["down_streak"] == 0


def test_market_pass_tags_rows_on_and_off(env):
    mp = env["monkeypatch"]
    falling = [120.0] * 18 + [110.0, 100.0]
    mp.setattr(rf, "completed_daily", lambda t, s: pd.DataFrame({"close": falling if t == "SOXL" else [100.0] * 20}))
    from tests.unit.test_method import _bars
    mp.setattr(rf.pq, "get_intraday", lambda t, *a: _bars("2026-10-15", [
        ("09:30", 94, 95, 93, 94) if t == "SOXL" else ("09:30", 98, 99, 97, 98),
        ("09:31", 94, 95, 93, 94.5) if t == "SOXL" else ("09:31", 98, 99, 97, 98)]))
    snap = rf.run(now=at("09:50"))
    rows = {r["ticker"]: r for r in snap["bounce"]["rows"]}
    assert rows["SOXL"]["gapfade"] == "on" and rows["MXL"]["gapfade"] == "off"
    soxl = snap["gapfade"]["long"][0]
    assert soxl["status"] == "on" and soxl["open"] == 94.0 and soxl["grade_open"] == "B"


def test_names_that_stopped_trading_drop_off_both_lists():
    falling = [30.0] * 22 + [28.0, 26.0, 24.0]
    rising = [20.0] * 22 + [22.0, 25.0, 29.0]
    long, days = _long_frame({"LIVE": falling, "GONE": falling, "UP": rising, "UPGONE": rising})
    last = long["date"].max()
    long = long[~((long["ticker"].isin(["GONE", "UPGONE"])) & (long["date"] == last))]   # no bar on the newest day
    assert breadth.prefilter(long) == ["LIVE"]
    assert breadth.prefilter_up(long) == ["UP"]


def test_a_missing_open_is_read_from_the_snapshot_any_time_of_day(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(rf.breadth, "snapshot_prices", lambda day: calls.append(day) or {"PARA": {"price": 104.0, "open": 103.0}})
    monkeypatch.setattr(rf, "completed_daily", lambda t, s: pd.DataFrame({"close": [90.0, 95.0, 100.0]}))
    rows = [{"ticker": "PARA", "prior_close": 100.0, "pct_from_9ema": 0.2, "atr_pct": 0.05, "price": 104.0}]
    blk = rf.gapfade_block("2026-11-09", "market", at("14:00", day="2026-11-09"), [], rows, {}, {}, {}, True,
                           [], ["PARA"], [], [], {}, None, None, False)
    assert calls == ["2026-11-09"]
    para = blk["short"][0]
    assert para["open"] == 103.0 and para["status"] == "on" and rows[0]["gapfade"] == "on"
