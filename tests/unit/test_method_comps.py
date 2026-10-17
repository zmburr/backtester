"""Method's comps (method/comps.py): the node state uses bars up to the node only, the labels
read the path after it, and the match keeps one node a comp day, at most 3 days a date, inside
the clock window."""

import numpy as np
import pandas as pd
import pytest

from method import comps as C

ET = "US/Eastern"
DAY = "2025-03-10"


def bars_from(path, day=DAY, start="09:30", vol=1000.0):
    """1-min bars from (minute after 9:30, close) anchors, linear in between; each bar's high and
    low sit 0.02 off the open/close."""
    xs, ys = zip(*path)
    closes = np.interp(np.arange(xs[-1] + 1), xs, ys)
    opens = np.r_[closes[0], closes[:-1]]
    idx = pd.date_range(pd.Timestamp(f"{day} {start}", tz=ET), periods=len(closes), freq="1min")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 0.02,
                         "low": np.minimum(opens, closes) - 0.02, "close": closes,
                         "volume": vol}, index=idx)


# open 100, flush to 95 at 9:40, bounce to 98 by 10:30, fade, flat to the close
FLUSH = [(0, 100), (10, 95), (60, 98), (120, 97), (389, 97)]
ATR_PCT = 0.04          # ATR = 0.04 x the 9:30 open (100) = $4


def test_state_is_causal():
    full = bars_from(FLUSH)
    for t in (12, 30, 61, 150, 270):
        cut = full[full.index < pd.Timestamp(f"{DAY} 09:30", tz=ET) + pd.Timedelta(minutes=t)]
        a, b = C.state_at(C.rth_arrays(full), 4.0, t), C.state_at(C.rth_arrays(cut), 4.0, t)
        assert {k: a[k] for k in C.FEATS + ("close", "lod", "lod_m")} == {k: b[k] for k in C.FEATS + ("close", "lod", "lod_m")}


def test_state_fields():
    a = C.rth_arrays(bars_from(FLUSH))
    s = C.state_at(a, 4.0, 30)                                  # 10:00: the 9:59 bar's close
    assert s["lod"] == pytest.approx(95 - 0.02)
    assert s["lod_m"] == 10 and s["mins_since_low"] == 20 and s["low_pre10"] == 1.0
    assert s["close"] == pytest.approx(95 + 3 * 19 / 50)
    assert s["bounce_atr"] == pytest.approx((s["close"] - s["lod"]) / 4.0)
    assert s["flush_atr"] == pytest.approx((100 - s["lod"]) / 4.0)
    assert C.state_at(a, 4.0, 0) is None                        # no bar has closed at 9:30
    assert C.state_at(a, 0, 30) is None


def test_premarket_bars_ignored_and_vwap():
    pm = bars_from([(0, 90), (29, 90)], start="09:00")         # a lower premarket
    full = pd.concat([pm, bars_from(FLUSH)])
    a = C.rth_arrays(full)
    assert a["m"][0] == 0 and a["l"].min() == pytest.approx(95 - 0.02)
    flat = bars_from([(0, 10), (9, 10)])
    flat["vwap"] = 10.0
    assert np.allclose(C.rth_arrays(flat)["vw"], 10.0)
    novol = bars_from(FLUSH).drop(columns="volume")
    v = C.rth_arrays(novol)["vw"]
    assert np.isfinite(v).all()


def test_labels_up_low_neither():
    a = C.rth_arrays(bars_from(FLUSH))
    up = C.labels_at(a, 4.0, C.state_at(a, 4.0, 20))           # 9:50, off the low: +0.5 ATR ($2) comes first
    assert up["outcome"] == "up"
    s = C.state_at(a, 4.0, 60)                                  # 10:30 at 98: +$2 never comes, no new low
    lab = C.labels_at(a, 4.0, s)
    assert lab["outcome"] == "neither"
    assert lab["fwd60"] == pytest.approx((a["c"][119] - s["close"]) / 4.0)
    assert lab["close_vs"] == pytest.approx((97 - s["close"]) / 4.0)
    dead = C.rth_arrays(bars_from([(0, 100), (10, 95), (30, 96), (60, 93), (389, 93)]))
    assert C.labels_at(dead, 4.0, C.state_at(dead, 4.0, 30))["outcome"] == "low"


def test_same_bar_counts_as_low_first():
    b = bars_from([(0, 100), (10, 95), (30, 96), (389, 96)])
    i = 40                                                      # one wide bar: new low and +0.5 ATR
    b.iloc[i, b.columns.get_loc("low")] = 94.0
    b.iloc[i, b.columns.get_loc("high")] = 99.0
    a = C.rth_arrays(b)
    assert C.labels_at(a, 4.0, C.state_at(a, 4.0, 30))["outcome"] == "low"


def test_day_nodes_grid():
    nodes = C.day_nodes(bars_from(FLUSH), ATR_PCT)
    assert len(nodes) == (C.NODE_TO - C.NODE_FROM) // C.STEP + 1
    assert nodes[0]["t"] == C.NODE_FROM and nodes[-1]["t"] == C.NODE_TO
    assert C.day_nodes(bars_from(FLUSH), None) == []


def synthetic(days_per_date=6, dates=("2024-01-02", "2024-01-03", "2024-01-04"), ts=(26, 30, 34, 60)):
    """A small table: `days_per_date` days on each date, nodes at `ts`; day i's state grows with i,
    its fwd values are i/100 and its outcome alternates up / low."""
    days, nodes = [], {k: [] for k in ("day", "t", *C.FEATS, "outcome", "fwd15", "fwd30", "fwd60", "close_vs")}
    for d in dates:
        for j in range(days_per_date):
            i = len(days)
            days.append({"date": d, "ticker": f"T{i}", "day_type": "normal", "kind": "stock", "atr_pct": 0.05, "src": "ev:surfaced"})
            for t in ts:
                nodes["day"].append(i)
                nodes["t"].append(t)
                nodes["mins_since_low"].append(10.0 + j)
                nodes["bounce_atr"].append(0.1 * j)
                nodes["vwap_atr"].append(0.0)
                nodes["low_pre10"].append(1.0)
                nodes["flush_atr"].append(1.0)
                nodes["outcome"].append(1 if j % 2 == 0 else -1)
                for k in ("fwd15", "fwd30", "fwd60", "close_vs"):
                    nodes[k].append(i / 100)
    arr = {k: np.asarray(v, dtype=np.int32 if k == "day" else np.int16 if k == "t" else np.int8 if k == "outcome" else np.float32)
           for k, v in nodes.items()}
    return C.Comps(pd.DataFrame(days), arr)


def test_match_caps():
    cm = synthetic()
    s = {"t": 30, "mins_since_low": 10.0, "bounce_atr": 0.0, "vwap_atr": 0.0, "low_pre10": 1.0, "flush_atr": 1.0,
         "day_type": "normal", "kind": "stock"}
    m = C.match(s, cm, k=200)
    assert m["n"] == 9 and m["dates"] == 3                      # 3 dates x at most 3 days, one node each
    m2 = C.match(s, cm, k=200, per_date=10)
    assert m2["n"] == 18                                        # every day once, not once per node
    assert C.match(s, cm, k=4)["n"] == 4
    assert C.match({**s, "t": 200}, cm) is None                 # nothing within +-20 min
    assert C.match(s, cm, before="2024-01-03")["dates"] == 1
    assert C.match(s, cm, exclude_date="2024-01-02")["dates"] == 2


def test_match_picks_nearest_days():
    cm = synthetic(days_per_date=6, dates=("2024-01-02",))
    s = {"t": 30, "mins_since_low": 10.0, "bounce_atr": 0.0, "vwap_atr": 0.0, "low_pre10": 1.0, "flush_atr": 1.0}
    m = C.match(s, cm, k=200, per_date=3)
    # the 3 closest days are j = 0, 1, 2: outcomes up, low, up; fwd 0.00, 0.01, 0.02
    assert m["p_up"] == pytest.approx(2 / 3) and m["p_low_first"] == pytest.approx(1 / 3)
    assert m["cone"][0][0] == 15 and m["cone"][0][2] == pytest.approx(0.01, abs=1e-6)
    assert m["close_med"] == pytest.approx(0.01, abs=1e-6)
    assert m["enough"] is False


def test_cone_quantiles():
    cm = synthetic(days_per_date=40, dates=("2024-01-02",))
    sel = np.flatnonzero(cm.n["t"] == 30)
    res = C.summarize(cm, sel)
    vals = cm.n["fwd30"][sel]
    assert res["cone"][1] == pytest.approx([30, *np.percentile(vals, [25, 50, 75])])
    assert res["n"] == 40 and res["enough"] is True


def test_save_load_subset(tmp_path):
    cm = synthetic()
    p = tmp_path / "nodes.npz"
    cm.save(p)
    back = C.Comps.load(p)
    assert back.days.ticker.tolist() == cm.days.ticker.tolist()
    assert np.array_equal(back.n["day"], cm.n["day"])
    sub = cm.subset(cm.day_date < "2024-01-03")
    assert len(sub.days) == 6 and sub.n["day"].max() == 5 and len(sub.n["t"]) == 6 * 4


def test_day_type_of():
    assert [C.day_type_of(n) for n in (0, 5, 6, 10, 11, 15, 16, 49, None)] == \
        ["normal", "normal", "elevated", "elevated", "11-15", "11-15", "16+", "16+", "normal"]


def test_population_dedupes_and_drops_junk_atr(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame({"date": ["2025-01-02", "2025-01-02"], "ticker": ["AAA", "BBB"], "atr_pct": [24.8, 0.05],
                  "cap": ["Small", "Medium"], "cohort": ["surfaced", "nogo_control"]}).to_csv(d / "score_ev_study.csv", index=False)
    pd.DataFrame({"target_date": ["2025-01-02", "2025-01-03"], "ticker": ["AAA", "CCC"], "bucket": ["bounce", "reversal"],
                  "atr_pct": [0.07, 0.03], "cap": ["Small", "Large"], "cohort": ["control", "surfaced"]}).to_csv(d / "signal_outcomes.csv", index=False)
    pd.DataFrame({"date": ["2025-01-02"], "ticker": ["BBB"], "kind": ["strict"], "score": [5], "n_flag": [12], "state": ["breadth"],
                  "exit_r": [1.0], "r1": [1.0], "r2": [None]}).to_csv(d / "bounce_day_edge_history.csv", index=False)
    pd.DataFrame({"date": ["1/2/2025"], "ticker": [" DDD "], "atr_pct": [0.09], "cap": ["ETF"]}).to_csv(d / "bounce_data.csv", index=False)
    monkeypatch.setattr(C, "DATA", d)
    pop = C.population(None)
    got = {(r.date, r.ticker): (r.atr_pct, r.src) for r in pop.itertuples()}
    assert got == {("2025-01-02", "AAA"): (0.07, "scorecard:control"),     # the ev row's 24.8 is junk
                   ("2025-01-02", "BBB"): (0.05, "ev:nogo_control"),
                   ("2025-01-02", "DDD"): (0.09, "book")}
