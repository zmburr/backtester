"""checkpoint_study.rule_sim — the live exit rule, replayed on 5s bars.

Parity with the real orderPipe TradeManager is checked on real trades by
checkpoint_study/parity.py (30/30 exact on 2026-09-29); these pin the rule's
edge cases on synthetic bars.
"""
import numpy as np
import pandas as pd
import pytest

from checkpoint_study.rule_sim import DayBars, SimParams, simulate, stop_offset

T0 = pd.Timestamp("2026-03-02 10:00:00", tz="US/Eastern")


def minute_bars(fives, minutes, extremes_fn=None):
    """Plain (T-m, T] resample — same convention as checkpoint_features.minute_bars."""
    return (fives.resample(f"{minutes}min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["close"]))


def bars(closes, start=T0 - pd.Timedelta(minutes=10), wick=0.02):
    idx = pd.DatetimeIndex([start + pd.Timedelta(seconds=5 * (i + 1)) for i in range(len(closes))])
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick,
                         "close": c, "volume": 1000.0}, index=idx)


def day(closes, **kw):
    f = bars(closes, **kw)
    return DayBars.build(f, minute_bars)


def ramp(n_pre=120, n_post=12 * 40, start=50.0, step=0.01):
    """Flat before the anchor, then a steady climb."""
    return [start] * n_pre + [start + step * (i + 1) for i in range(n_post)]


def test_stop_offset_bands_match_orderpipe():
    assert stop_offset(9.99) == 0.01
    assert stop_offset(10) == 0.05
    assert stop_offset(99.99) == 0.05
    assert stop_offset(100) == 0.20


def test_steady_trend_survives_all_checkpoints():
    r = simulate(day(ramp()), anchor=T0, side=1, ref=49.9, r_ps=0.1,
                 p=SimParams(horizon_min=35))
    assert r["path"]["first_breach_ts"] is None
    for k in (6, 17, 30):
        assert r["k"][k] is not None
        assert r["k"][k]["hold_capped"] is True


def test_ref_stop_breach_inside_two_minutes():
    closes = [50.0] * 120 + [50.05] * 6 + [49.80] * 60      # drops through ref 49.9 - 0.05
    r = simulate(day(closes), anchor=T0, side=1, ref=49.9, r_ps=0.1)
    assert r["path"]["first_breach_phase"] == "ref"
    assert r["path"]["stop"] == pytest.approx(49.85)
    assert r["k"][6] is None


def test_stop_moves_to_first_close_when_ref_is_through_it():
    closes = [50.0] * 120 + [49.0] + [49.1] * 200
    r = simulate(day(closes), anchor=T0, side=1, ref=49.9, r_ps=0.1)
    assert r["path"]["stop"] == pytest.approx(49.0 - 0.05)


def test_trail_level_is_the_latest_COMPLETED_two_min_bar_strictly_before_t():
    d = day(ramp())
    r = simulate(d, anchor=T0, side=1, ref=49.9, r_ps=0.1, p=SimParams(horizon_min=10))
    t6 = r["k"][6]["t_k"]
    completed = d.twos[d.twos.index < t6]
    assert r["k"][6]["trail_level_k"] == pytest.approx(completed["low"].iloc[-1])


def test_trail_breach_is_close_strictly_through_the_level():
    up = ramp(n_post=12 * 8)
    top = up[-1]
    closes = up + [top - 0.30] * 60            # falls below the last 2-min low
    r = simulate(day(closes), anchor=T0, side=1, ref=49.9, r_ps=0.1)
    assert r["path"]["first_breach_phase"] == "trail"
    assert r["path"]["first_breach_ts"] > T0 + pd.Timedelta(minutes=8)


def test_short_is_the_mirror_of_long():
    long_c = ramp()
    short_c = [100.0 - c for c in long_c]
    rl = simulate(day(long_c), anchor=T0, side=1, ref=49.9, r_ps=0.1, p=SimParams(horizon_min=20))
    rs = simulate(day(short_c), anchor=T0, side=-1, ref=50.1, r_ps=0.1, p=SimParams(horizon_min=20))
    for k in (6, 17):
        assert rs["k"][k]["fwd_hold"] == pytest.approx(rl["k"][k]["fwd_hold"])
        assert rs["k"][k]["fwd_trail_1m"] == pytest.approx(rl["k"][k]["fwd_trail_1m"])


def test_trim_half_is_the_average_of_hold_and_exit_now():
    r = simulate(day(ramp()), anchor=T0, side=1, ref=49.9, r_ps=0.1, p=SimParams(horizon_min=20))
    row = r["k"][6]
    assert row["fwd_trim_half"] == pytest.approx(0.5 * row["fwd_hold"] + 0.5 * row["fwd_exit_now"])


def test_one_min_trail_is_never_looser_than_the_two_min_rule():
    rng = np.random.default_rng(3)
    closes = list(50 + np.cumsum(rng.normal(0.002, 0.02, 12 * 60)))
    r = simulate(day(closes), anchor=T0, side=1, ref=49.5, r_ps=0.1)
    for row in r["k"].values():
        if row is not None:
            assert row["trail_1m_exit_min"] <= row["hold_exit_min"] + 1e-9


def test_no_look_ahead_future_bars_do_not_change_the_checkpoint():
    base = ramp(n_post=12 * 7)
    feats = lambda f, **kw: {"n_bars": len(f), "last": float(f["close"].iloc[-1])}
    a = simulate(day(base + [10.0] * 400), anchor=T0, side=1, ref=49.9, r_ps=0.1,
                 features_fn=feats, entry=50.0)["k"][6]
    b = simulate(day(base + [90.0] * 400), anchor=T0, side=1, ref=49.9, r_ps=0.1,
                 features_fn=feats, entry=50.0)["k"][6]
    for key in ("t_k", "px_k", "trail_level_k", "n_bars", "last"):
        assert a[key] == b[key]
