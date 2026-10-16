"""Method's market breadth: the study's count at a price, the nightly candidates and store,
the live count (Trillium over Polygon, the open's lock), the once-a-day push, the edge
table lookups, the PC's extra names and the refresh pass's market rows (Polygon stubbed)."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from method import breadth as B
from method import core
from method import pc_pusher as pp
from method import refresh as rf

ET = "US/Eastern"


def at(hhmm, day="2026-10-15"):
    return pd.Timestamp(f"{day} {hhmm}", tz=ET).to_pydatetime()


def cand(pc=100.0, flag=92.0, relaxed=96.0, counted=True, **kw):
    return {"ticker": kw.get("ticker", "X"), "prior_close": pc, "cap": "Medium", "setup": "GapFade_strongstock",
            "counted": counted, "static_score": 3, "static_relaxed": 3, "gap_th": flag / pc - 1,
            "flag_price": flag, "relaxed_price": relaxed, "failing": [], "adv_dollars": 5e7}


@pytest.fixture
def bdir(tmp_path, monkeypatch):
    d = tmp_path / "method"
    monkeypatch.setattr(B, "METHOD_DIR", d)
    monkeypatch.setattr(B, "EDGE_FILE", tmp_path / "edge.json")
    core.write_json(tmp_path / "edge.json", {"states": {
        "normal": {"n": 733, "mean_r": 0.14, "win": 0.39, "days": 1080,
                   "by_score": {"4": {"n": 490, "mean_r": 0.21, "win": 0.4}, "5": {"n": 204, "mean_r": -0.01, "win": 0.36},
                                "6": {"n": 39, "mean_r": -0.03, "win": 0.41}, "low": {"n": 491, "mean_r": -0.07, "win": 0.3}}},
        "breadth": {"n": 295, "mean_r": 1.2, "win": 0.57, "days": 17,
                    "by_score": {"4": {"n": 167, "mean_r": 0.91, "win": 0.53}, "5": {"n": 97, "mean_r": 1.48, "win": 0.57},
                                 "6": {"n": 31, "mean_r": 1.89, "win": 0.74},
                                 "low": {"n": 26, "mean_r": 1.5, "win": 0.6, "shrunk": True}}}},
        "relaxed": {"breadth": {"n": 221, "mean_r": 1.05, "win": 0.59}}})
    return d


# ---------------------------------------------------------------- the count at a price

def test_a_name_flags_at_or_below_its_flag_price_and_only_on_a_gap_down():
    c = cand()
    assert B.flag_of(c, 92.0) == "strict" and B.flag_of(c, 80.0) == "strict"
    assert B.flag_of(c, 94.0) == "relaxed"                      # between the strict and relaxed prices
    assert B.flag_of(c, 97.0) is None and B.flag_of(c, None) is None
    four = cand(flag=100.0, relaxed=100.0)                      # already 4/6: any gap down
    assert B.flag_of(four, 99.99) == "strict" and B.flag_of(four, 100.0) is None


def test_count_is_strict_flags_in_the_counted_universe():
    strict = lambda **kw: cand(relaxed=None, **kw)
    cands = {"A": strict(), "B": strict(), "C": strict(counted=False), "D": strict(),
             "E": dict(cand(), flag_price=None, relaxed_price=95.0)}
    res = B.count(cands, {"A": 90.0, "B": 93.0, "C": 85.0, "D": 95.0, "E": 94.0})
    assert res["n"] == 1 and set(res["names"]) == {"A", "C"}     # C flags but isn't counted
    assert res["relaxed"] == ["E"] and res["state"] == "normal"
    assert [x["ticker"] for x in res["near"]] == ["B", "D"]      # closest first, within 5%
    assert res["near"][0]["away"] == pytest.approx(93 / 92 - 1)
    many = {f"N{i}": cand() for i in range(11)}
    assert B.count(many, {t: 90.0 for t in many})["state"] == "breadth"
    assert B.state_for(6) == "elevated" and B.state_for(5) == "normal" and B.state_for(None) is None


# ---------------------------------------------------------------- candidates

def _selloff(n=260, drop=(0.92, 0.9, 0.88, 0.85), start=100.0, rng_seed=1):
    """A steady stock that then sells off hard on rising ranges: dates as strings."""
    rng = np.random.default_rng(rng_seed)
    days = pd.bdate_range("2025-06-02", periods=n + len(drop)).strftime("%Y-%m-%d")
    closes = list(start * (1 + rng.normal(0.001, 0.01, n)).cumprod())
    for f in drop:
        closes.append(closes[-1] * f)
    closes = np.array(closes)
    opens = np.r_[closes[0], closes[:-1]]
    rngs = np.r_[np.full(n, 0.02), np.full(len(drop), 0.09)]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * (1 + rngs / 2),
                         "low": np.minimum(opens, closes) * (1 - rngs / 2), "close": closes,
                         "volume": 2e6}, index=days)


def test_the_flag_price_is_where_the_real_checklist_reaches_4_of_6():
    from analyzers.bounce_scorer import BouncePretrade
    from scanners.bounce_backscanner import BounceBackscanner
    hist = _selloff()
    session = (pd.Timestamp(hist.index[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    c = B.compute_candidate("SELL", session, hist)
    assert c and c["counted"] and c["flag_price"] is not None and c["flag_price"] < c["prior_close"]

    def score_at(price):                        # the backscanner's own scoring at that open
        bar = pd.DataFrame([{"open": price, "high": price, "low": price, "close": price, "volume": 0.0}], index=[session])
        m = BounceBackscanner.compute_bounce_metrics(None, "SELL", session, pd.concat([hist, bar]))
        cap = BounceBackscanner.estimate_cap("SELL", m["current_price"], m["avg_daily_vol"])
        return BouncePretrade().validate("SELL", m, cap=cap).score
    assert score_at(c["flag_price"] * 0.999) >= 4
    if c["static_score"] == 3:
        assert score_at(min(c["flag_price"] * 1.01, c["prior_close"] * 0.999)) == 3


def test_names_that_cant_flag_are_not_candidates():
    flat = _selloff(drop=())                     # no selloff: not 2 down days / 5% off the high
    assert B.compute_candidate("FLAT", "2026-07-01", flat) is None
    thin = _selloff().assign(volume=1000.0)      # $ volume far under both floors
    assert B.compute_candidate("THIN", "2026-07-01", thin) is None


def test_prefilter_keeps_sellers_and_drops_flat_and_dotted_names():
    sell, flat = _selloff(), _selloff(drop=())
    long = pd.concat([sell.assign(ticker="SELL"), flat.assign(ticker="FLAT"), sell.assign(ticker="BRK.B")])
    long = long.rename_axis("date").reset_index()
    assert B.prefilter(long) == ["SELL"]


def test_the_store_fetches_only_missing_days_and_prunes(tmp_path):
    store = tmp_path / "grouped"
    store.mkdir()
    pd.DataFrame({"ticker": ["OLD"]}).to_pickle(store / "2026-01-02.pkl")
    pd.DataFrame({"ticker": ["HAVE"]}).to_pickle(store / "2026-10-13.pkl")
    fetched = []

    def fetch(d):
        fetched.append(d)
        return pd.DataFrame() if d == "2026-10-15" else pd.DataFrame({"ticker": ["X"], "open": [1.0], "high": [1.0],
                                                                       "low": [1.0], "close": [1.0], "volume": [1.0]})
    missing = B.update_store(["2026-10-13", "2026-10-14", "2026-10-15"], fetch=fetch, store=store)
    assert fetched == ["2026-10-14", "2026-10-15"] and missing == ["2026-10-15"]
    assert sorted(p.stem for p in store.glob("*.pkl")) == ["2026-10-13", "2026-10-14"]
    assert list(B.load_store(["2026-10-13", "2026-10-14"], store=store).date) == ["2026-10-13", "2026-10-14"]


def test_split_names_use_adjusted_history(monkeypatch):
    sell = _selloff()
    raw = sell.copy()
    raw.loc[raw.index[:200], ["open", "high", "low", "close"]] *= 10      # a 1:10 reverse split, unadjusted
    long = raw.assign(ticker="SPLT").rename_axis("date").reset_index()
    session = (pd.Timestamp(sell.index[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    plain = B.build_candidates(session, long)
    fixed = B.build_candidates(session, long, {"SPLT"}, adjusted=lambda t, s: sell)
    assert "SPLT" in fixed
    assert plain.get("SPLT", {}).get("flag_price") != fixed["SPLT"]["flag_price"]


# ---------------------------------------------------------------- the live count

def _built(bdir, session="2026-10-15", **cands):
    core.write_json(bdir / f"breadth_candidates_{session}.json",
                    {"session": session, "built_at": "2026-10-14T17:45:00-04:00", "last_day": "2026-10-14",
                     "missing_days": [], "candidates": cands})


def test_premarket_count_prefers_trillium_and_reuses_polygon_on_a_post(bdir):
    _built(bdir, A=cand(), B=cand())
    poly = {"A": {"price": 95.0, "ts": 1}, "B": {"price": 90.0, "ts": 1}, "ZZ": {"price": 1.0, "ts": 1}}
    mb = B.evaluate("2026-10-15", "premarket", at("08:00"), {"A": {"price": 91.0}}, snap_prices=poly)
    assert mb["status"] == "ok" and mb["n"] == 2 and mb["trillium"] == 1 and mb["basis"] == "now"
    assert set(mb["poly"]) == {"A", "B"} and mb["edge"]["mean_r"] == 0.14
    again = B.evaluate("2026-10-15", "premarket", at("08:01"), {}, prev=mb)     # server re-count: no Polygon
    assert again["n"] == 1 and again["detail"]["A"]["price"] == 95.0
    assert "poly" not in B.public(again)
    assert B.evaluate("2026-10-16", "premarket", at("08:00"), {})["status"] == "not built"


def test_the_open_count_locks_and_stays(bdir):
    _built(bdir, A=cand(), B=cand())
    poly = {"A": {"price": 99.0, "ts": 1, "open": 91.0}, "B": {"price": 90.0, "ts": 1}}
    mb = B.evaluate("2026-10-15", "market", at("09:32"), {}, snap_prices=poly)
    assert not mb["locked"] and mb["basis"] == "now" and mb["n"] == 2        # B has no open yet
    poly["B"]["open"] = 99.0
    mb = B.evaluate("2026-10-15", "market", at("09:46"), {}, prev=mb, snap_prices=poly)
    assert mb["locked"] and mb["basis"] == "open" and mb["names"] == ["A"]   # A opened at 91, B at 99
    later = B.evaluate("2026-10-15", "market", at("10:30"), {"B": {"price": 80.0, "open": 85.0}}, prev=mb, snap_prices={})
    assert later["names"] == ["A"] and later["n"] == 1                       # the open's count is fixed
    assert B.watch_names(later) == []


def test_one_push_a_day_and_only_once_a_phone_got_it(bdir):
    _built(bdir, **{f"N{i}": cand() for i in range(12)})
    mb = B.evaluate("2026-10-15", "premarket", at("08:00"), {}, snap_prices={f"N{i}": {"price": 90.0} for i in range(12)})
    assert mb["n"] == 12 and mb["state"] == "breadth"
    assert B.breadth_push(mb, lambda *a, **k: 0) is False and "pushed" not in mb        # no phone: not recorded
    sent = []
    assert B.breadth_push(mb, lambda title, body, url, tag: sent.append((title, body, tag)) or 1)
    assert sent[0][0] == "Breadth day · 12 names" and "+1.2R a name, 57% win" in sent[0][1]
    assert sent[0][2] == "breadth-2026-10-15"
    nxt = B.evaluate("2026-10-15", "premarket", at("08:05"), {}, prev=mb)
    assert nxt["pushed"] and B.breadth_push(nxt, lambda *a, **k: sent.append(1) or 1) is False
    assert len(sent) == 1
    small = B.count({"A": cand()}, {"A": 90.0}) | {"status": "ok", "session": "2026-10-16"}
    assert B.breadth_push(small, lambda *a, **k: 1) is False


def test_edge_lookup_by_state_and_score(bdir):
    t = B.edge_table()
    assert B.edge_for(t, "breadth", 5)["mean_r"] == 1.48
    assert B.edge_for(t, "breadth", 7)["mean_r"] == 1.89 and B.edge_for(t, "breadth", 2)["shrunk"]
    assert B.edge_for(t, "breadth", None)["mean_r"] == 1.2
    assert B.edge_for(t, "breadth", 4, relaxed=True)["mean_r"] == 1.05
    assert B.edge_for(t, "elevated", 4) is None and B.edge_for(t, None, 4) is None


def test_the_committed_edge_table_has_every_state():
    t = json.loads((B.DATA_DIR / "bounce_day_edge.json").read_text())
    assert set(t["states"]) == {"normal", "elevated", "breadth"}
    assert t["states"]["breadth"]["mean_r"] > t["states"]["normal"]["mean_r"]
    assert all(set(s["by_score"]) == {"4", "5", "6", "low"} for s in t["states"].values())


def test_board_and_watch_names():
    mb = {"status": "ok", "state": "normal", "names": ["A"], "relaxed": ["R"], "near": [{"ticker": "N"}]}
    assert B.board_names(mb) == ["A"] and B.watch_names(mb) == ["A", "N"]
    assert B.board_names(dict(mb, state="breadth")) == ["A", "R"]
    assert B.board_names({"status": "not built"}) == []


def test_the_pc_prices_near_names_until_the_open_locks():
    snap = {"bounce": {"rows": [{"ticker": "SOXL"}]}, "reversal": {"rows": []},
            "market_breadth": {"status": "ok", "names": ["CRWV"], "near": [{"ticker": "ROKU"}]}}
    assert pp.pick_names(snap)[1] == ["SOXL", "CRWV", "ROKU"]
    snap["market_breadth"]["locked"] = True
    assert pp.pick_names(snap)[1] == ["SOXL"]


# ---------------------------------------------------------------- the refresh pass

def test_refresh_adds_market_rows_with_their_edge(bdir, tmp_path, monkeypatch):
    from tests.unit.test_method import LADDER, STATIC, _bars, fake_grade_at
    sig = tmp_path / "priority_signals"
    sig.mkdir()
    monkeypatch.setattr(rf, "METHOD_DIR", bdir)
    monkeypatch.setattr(rf, "SNAPSHOT", bdir / "snapshot.json")
    monkeypatch.setattr(rf, "LIVE", bdir / "live.json")
    monkeypatch.setattr(rf, "MARKET_BREADTH", bdir / "market_breadth.json")
    monkeypatch.setattr(rf, "SIGNALS_DIR", sig)
    monkeypatch.setattr(rf, "load_watchlist", lambda: ["SOXL"])
    monkeypatch.setattr(rf, "_load_json", lambda p: {})
    monkeypatch.setattr(rf, "get_cap", lambda t, c: "Medium")
    monkeypatch.setattr(rf, "is_session_day", lambda d: True)
    monkeypatch.setattr(rf, "next_trading_day", lambda d: d + datetime.timedelta(days=1))
    monkeypatch.setattr(rf, "completed_daily", lambda t, s: pd.DataFrame({"close": [100.0] * 20}))
    monkeypatch.setattr(rf, "static_from_hist", lambda hist: dict(STATIC))
    monkeypatch.setattr(rf, "go_gap_trigger", lambda t, s, cap, checker: -0.04)
    monkeypatch.setattr(rf, "ladder_payload", lambda s, cap, t: dict(LADDER))
    monkeypatch.setattr(rf, "grade_at", fake_grade_at)
    monkeypatch.setattr(rf, "historical_grade_stats", lambda: {})
    monkeypatch.setattr(rf, "lookup_bounce_odds", lambda score, cap: None)
    monkeypatch.setattr(rf, "reversal_pass", lambda *a: [])
    monkeypatch.setattr(rf.time, "sleep", lambda s: None)
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(rf.pq, "get_intraday", lambda t, d, m, s: _bars(d, [("08:20", 96, 96, 90, 91)]))
    _built(bdir, CRWV=cand(ticker="CRWV"), ROKU=cand(ticker="ROKU"))
    calls = []
    monkeypatch.setattr(rf.breadth, "snapshot_prices",
                        lambda day: calls.append(day) or {"CRWV": {"price": 90.0, "ts": 1}, "ROKU": {"price": 99.0, "ts": 1}})
    snap = rf.run(now=at("08:35"))
    rows = {r["ticker"]: r for r in snap["bounce"]["rows"]}
    assert set(rows) == {"SOXL", "CRWV"} and calls == ["2026-10-15"]
    assert rows["CRWV"]["market"] and rows["CRWV"]["breadth_flag"] == "strict" and rows["CRWV"]["counted"]
    assert "market" not in rows["SOXL"] and rows["SOXL"]["edge"]["mean_r"] is not None
    mb = snap["market_breadth"]
    assert mb["n"] == 1 and mb["names"] == ["CRWV"] and "poly" not in mb
    assert json.loads((bdir / "market_breadth.json").read_text())["poly"]["CRWV"]["price"] == 90.0
