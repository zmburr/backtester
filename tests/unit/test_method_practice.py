"""Method's Practice arena: labels, the deck and how it's dealt, your trade days from the feed,
one day's payload (triggers, the rules' R on your sizing, quick-read cards), the log and stats,
the /api/practice routes, and the play engine's node tests."""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from method import core
from method import practice as pr

FIX = Path(__file__).parent / "fixtures" / "practice_days.json"
DAYS = json.loads(FIX.read_text(encoding="utf-8"))


def review_of(day: dict) -> dict:
    """A review()-shaped dict from a fixture payload: its candles and the rules' raw trades."""
    trades = [{k: v for k, v in t.items() if k not in ("r_live", "risk_live")} for t in day["rules"]]
    return {"ticker": day["ticker"], "date": day["date"], "side": day["side"], "cap": day.get("cap"),
            "candles": day["candles"], "daily": [], "prior_close": day["prior_close"], "atr": day["atr"],
            "session": day.get("session") or {}, "at_open": day.get("at_open"), "trades": trades}


def row_of(day: dict, source: str = "book", setup: str | None = None) -> dict:
    return {"key": day["key"], "side": day["side"], "ticker": day["ticker"], "date": day["date"], "source": source,
            "setup": setup or day.get("setup"), "year": int(day["date"][:4]),
            "setup_class": pr.setup_class(day["side"], setup or day.get("setup"), source)}


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("side,setup,source,want", [
    ("bounce", "GapFade_weakstock", "book", "Gap-fade"),
    ("bounce", "IntradayCapitch_strongstock", "book", "Intraday flush"),
    ("bounce", None, "flush", "Intraday flush"),
    ("bounce", "GapFade_strongstock", "pool", "Gap-fade"),
    ("bounce", "capitulation", "yours", "Gap-fade"),
    ("bounce", None, "yours", None),
    ("reversal", "3DGapFade", "book", "3-day gap-fade"),
    ("reversal", "2DGapFade", "pool", "2-day gap-fade"),
    ("reversal", "GapDownTrendBreak", "book", "Gap-down trend break"),
    ("reversal", "2DBreakoutIB", "book", "Other"),
    ("reversal", "generic", "pool", None),
])
def test_setup_class(side, setup, source, want):
    assert pr.setup_class(side, setup, source) == want


def test_day_worth_follows_the_breadth_rule():
    assert [pr.day_worth(n) for n in (None, 0, 10, 11, 15, 16, 40)] == [None, "Pass", "Pass", "1R", "1R", "3R a try", "3R a try"]


# ---------------------------------------------------------------- the deck

def test_build_deck_dedupes_and_marks_days_you_traded():
    lib_rows = [{"key": "bounce:SOXL:2026-07-28", "side": "bounce", "ticker": "SOXL", "date": "2026-07-28", "cap": "ETF",
                 "setup": "GapFade_weakstock", "grade": "C", "ok": True},
                {"key": "bounce:NKLA:2020-06-09", "side": "bounce", "ticker": "NKLA", "date": "2020-06-09", "ok": False}]
    seed = [{"key": "reversal:AMC:2021-01-27", "side": "reversal", "ticker": "AMC", "date": "2021-01-27", "cap": None,
             "setup": "3DGapFade", "grade": None, "source": "pool"}]
    yours = [{"key": "bounce:SOXL:2026-07-28", "side": "bounce", "ticker": "SOXL", "date": "2026-07-28", "cap": None,
              "setup": "capitulation", "grade": None, "source": "yours"}]
    deck = pr.build_deck(lib_rows, seed, yours)
    assert [r["key"] for r in deck] == ["bounce:SOXL:2026-07-28", "reversal:AMC:2021-01-27"]   # a failed replay stays out
    soxl = deck[0]
    assert soxl["source"] == "book" and soxl["traded"] is True and soxl["grade"] == "C" and soxl["year"] == 2026
    assert deck[1]["setup_class"] == "3-day gap-fade" and deck[1]["traded"] is False


def _deck(n: int) -> list[dict]:
    return [{"key": f"bounce:T{i}:{2018 + i % 9}-01-02", "year": 2018 + i % 9} for i in range(n)]


def test_deal_never_repeats_inside_a_cycle(tmp_path):
    deck, state = _deck(25), tmp_path / "state.json"
    seen = []
    for _ in range(5):
        seen += pr.deal("play", 5, deck, state)
    assert len(seen) == 25 and len(set(seen)) == 25
    nxt = pr.deal("play", 5, deck, state)                     # the next cycle starts
    assert len(set(nxt)) == 5 and core.read_json(state)["play"]["cycle"] == 1
    assert pr.deal("reads", 3, deck, state)                   # each mode keeps its own place


def test_deal_adds_new_days_to_the_cycle(tmp_path):
    deck, state = _deck(6), tmp_path / "state.json"
    first = pr.deal("play", 4, deck, state)
    deck2 = deck + [{"key": "bounce:NEW:2026-11-05", "year": 2026}]
    rest = pr.deal("play", 3, deck2, state)
    assert set(first) | set(rest) == {r["key"] for r in deck2}


def test_old_days_come_first_more_often():
    deck = [{"key": f"k{i}", "year": 2018 if i < 50 else 2026} for i in range(100)]
    first = pr.cycle_order(deck, seed=3, this_year=2026)[:30]
    old = sum(1 for k in first if int(k[1:]) < 50)
    assert old >= 20


def test_feed_days_reads_your_price_action_days(tmp_path):
    feed = tmp_path / "trade_data.csv"
    pd.DataFrame([
        {"Date": "11/05/2026", "Symbol": "SOXS", "Start": "2026-11-05 09:33:41-05:00", "End": "2026-11-05 09:35:00-05:00",
         "Net P&L": -3580.78, "Max Size": -3000, "Avg Price at Max": 67.38, "Tags": "capitulation,bounce,A",
         "Avg Exit Price": 68.56, "Side": -1.0},
        {"Date": "11/05/2026", "Symbol": "SOXS", "Start": "2026-11-05 10:54:01-05:00", "End": "2026-11-05 11:05:32-05:00",
         "Net P&L": -5123.96, "Max Size": -15000, "Avg Price at Max": 78.82, "Tags": "capitulation,bounce,A",
         "Avg Exit Price": 79.15, "Side": -1.0},
        {"Date": "11/05/2026", "Symbol": "QQQ", "Start": "2026-11-05 11:35:07-05:00", "End": "2026-11-05 11:36:40-05:00",
         "Net P&L": 13841.9, "Max Size": 20000, "Avg Price at Max": 660.6, "Tags": "news,C", "Avg Exit Price": 661.3, "Side": 1.0},
        {"Date": "07/29/2026", "Symbol": "SNDK", "Start": "2026-07-29 09:40:00", "End": "2026-07-29 09:50:00",
         "Net P&L": -3000, "Max Size": 100, "Avg Price at Max": 1000, "Tags": "intradaycapitch", "Avg Exit Price": 970, "Side": 1.0},
        {"Date": "07/29/2026", "Symbol": "SNDK  260731C01000000", "Start": "", "End": "", "Net P&L": 10, "Max Size": 1,
         "Avg Price at Max": 1, "Tags": "capitulation,[opt]", "Avg Exit Price": 2, "Side": 1.0},
    ]).to_csv(feed, index=False)
    rows, fills = pr.feed_days(feed)
    keys = {r["key"]: r for r in rows}
    assert set(keys) == {"reversal:SOXS:2026-11-05", "bounce:SNDK:2026-07-29"}     # news and options stay out
    assert "bounce:QQQ:2026-11-05" in fills          # but a day you traded keeps its fills, for a book or pool day
    assert keys["bounce:SNDK:2026-07-29"]["setup"] == "intradaycapitch"
    f = fills["reversal:SOXS:2026-11-05"]
    assert len(f) == 2 and f[0]["pnl"] == pytest.approx(-3580.78)
    assert f[0]["start"] == int(pd.Timestamp("2026-11-05 09:33:41").timestamp())  # ET wall clock, like the candles


# ---------------------------------------------------------------- one day

@pytest.mark.parametrize("key", list(DAYS))
def test_day_payload_triggers_rules_and_cards(key):
    day = DAYS[key]
    out = pr.day_payload(row_of(day), review_of(day), None, {"2026-07-28": (14, "elevated")})
    assert out["ok"] is True
    assert out["triggers"] == day["triggers"]
    entries = {t["entry_t"] for t in out["rules"]}
    assert entries <= set(out["triggers"])                         # every rule entry is on a trigger bar
    assert out["rules_r"] == pytest.approx(day["rules_r"])
    trig = set(out["triggers"])
    for c in out["cards"]:
        assert c["answer"] in c["choices"]
        assert c["cut"] % 60 == 0
        if c["type"] == "trigger":
            assert (c["mark"] in trig) == (c["answer"] == "Yes")
    types = {c["type"] for c in out["cards"]}
    assert {"setup", "trigger", "stop", "works"} <= types
    if day["side"] == "bounce":
        size = next(c for c in out["cards"] if c["type"] == "size")
        assert size["answer"] == "1R"                              # 14 names: the 11-15 band


def test_bounce_triggers_match_the_live_rule():
    from scripts.backfill_entry_r import entry_signals
    day = DAYS["bounce:SOXL:2026-07-28"]
    two = pr.two_min(pr.candles_frame(day["candles"]))
    fires, _ = entry_signals(two, day["atr"])
    assert pr.bounce_triggers(two, day["atr"]) == [int((t - pd.Timedelta(minutes=2)).timestamp()) for t in fires]


def test_live_r_floors_the_risk():
    trades = [{"entry": 100.0, "stop": 99.0, "exit": 99.0, "kind": "stop", "r": -1.0, "half": None},
              {"entry": 100.0, "stop": 95.0, "exit": 110.0, "kind": "close", "r": 2.0, "half": None}]
    out = pr.live_r("bounce", trades, atr=8.0)                     # floor 4.0
    assert [round(t["r_live"], 4) for t in out] == [-0.25, 2.0]
    short = pr.live_r("reversal", [{"entry": 50.0, "stop": 52.0, "exit": 46.0, "kind": "tripwire",
                                    "half": {"t": 0, "price": 44.0}}], atr=4.0)
    assert short[0]["r_live"] == pytest.approx(0.5 * 6 / 2 + 0.5 * 4 / 2)


def test_a_day_without_bars_is_not_ok():
    day = DAYS["bounce:SOXL:2026-07-28"]
    out = pr.day_payload(row_of(day), {**review_of(day), "candles": []}, None, {})
    assert out["ok"] is False and out["cards"] == []


# ---------------------------------------------------------------- log and stats

def test_stats_rates_trends_and_the_will_it_work_base_rate(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(30):
        flags = [{"type": "early"}] if i < 20 else []          # early on the first 20 days, clean after
        pr.append_log({"mode": "play", "key": f"k{i}", "you_r": -1.0 if flags else 1.0, "rules_r": 0.5, "flags": flags}, log)
    for ans, chosen in [("Yes", "Yes"), ("No", "Yes"), ("No", "No"), ("No", "Yes")]:
        pr.append_log({"mode": "read", "key": "x", "card": "works", "answer": ans, "chosen": chosen, "correct": ans == chosen}, log)
    s = pr.stats(pr.read_log(log), last=10)
    assert s["plays"] == 30 and s["recent"] == 10 and s["clean_days"] == 10
    assert s["flags"]["early"]["days"] == 0 and s["trend"]["early"] == pytest.approx(-1.0)
    assert s["you_r"] == pytest.approx(10.0) and s["rules_r"] == pytest.approx(5.0)
    assert s["works"] == {"n": 4, "you": 0.5, "always_yes": 0.25, "always_no": 0.75}
    assert s["by_type"]["works"] == {"n": 4, "right": 2}


# ---------------------------------------------------------------- endpoints

@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    from method import library as lib
    from method import server as sv
    home = tmp_path / "practice"
    monkeypatch.setattr(pr, "HOME", home)
    monkeypatch.setattr(pr, "DECK", home / "deck.json")
    monkeypatch.setattr(pr, "STATE", home / "state.json")
    monkeypatch.setattr(pr, "LOG", home / "log.jsonl")
    day = DAYS["bounce:SOXL:2026-07-28"]
    deck = pr.build_deck([], [], [{**row_of(day, "yours", "capitulation"), "cap": "ETF", "grade": None}])
    core.write_json(pr.DECK, {"rows": deck})
    core.write_json(home / "fills.json", {day["key"]: [{"start": 1, "pnl": -3000.0}]})
    monkeypatch.setattr(lib, "cached_review", lambda side, tk, date, d, cap=None, row=None: review_of(day))
    monkeypatch.setitem(sv._practice_cache, "mtime", None)
    monkeypatch.setitem(sv._practice_cache, "states", {"2026-07-28": (14, "elevated")})
    monkeypatch.setitem(sv._practice_cache, "states_at", 1e18)
    app = sv.create_app()
    app.testing = True
    return app.test_client(), day


def test_practice_routes(arena):
    c, day = arena
    nxt = c.get("/api/practice/next?mode=play&n=3").get_json()
    assert nxt == {"keys": [day["key"]], "deck": 1}
    assert c.get("/api/practice/next?mode=nope").status_code == 400
    d = c.get(f"/api/practice/day?key={day['key']}").get_json()
    assert d["ok"] and d["fills"] == [{"start": 1, "pnl": -3000.0}] and d["day_n"] == 14 and d["traded"] is True
    assert c.get("/api/practice/day?key=bounce:ZZZ:2026-01-02").status_code == 404
    hdr = {"X-Method": "1"}
    r = c.post("/api/practice/log", json={"mode": "play", "key": day["key"], "you_r": -2.0, "rules_r": 0.1,
                                          "flags": [{"type": "early", "t": 1}], "junk": "x"}, headers=hdr).get_json()
    assert "junk" not in r and r["you_r"] == -2.0
    assert c.post("/api/practice/log", json={"mode": "play"}, headers=hdr).status_code == 400
    assert c.post("/api/practice/log", json={"mode": "read", "key": "k"}).status_code == 400      # no X-Method
    s = c.get("/api/practice/stats").get_json()
    assert s["plays"] == 1 and s["flags"]["early"]["days"] == 1


# ---------------------------------------------------------------- the play engine (node)

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_play_engine_node_suite():
    root = Path(__file__).resolve().parents[2]
    res = subprocess.run(["node", "--test", str(root / "tests" / "js" / "practice_engine.test.js")],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
