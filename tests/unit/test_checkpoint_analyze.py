"""checkpoint_study.analyze — the frozen protocol's mechanics, on synthetic panels.

The real panel is never used here: these check that a planted edge is found,
that noise is killed, and that noise never reaches (or peeks at) the test set.
"""
import numpy as np
import pandas as pd
import pytest

from checkpoint_study import analyze as A


def panel(n_days=400, per_day=3, planted=0.0, seed=0):
    rng = np.random.default_rng(seed)
    # ~260 train days (<= 2025-06-30) and ~140 test days at the default size
    dates = pd.date_range("2024-07-01", periods=n_days, freq="B").strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for _ in range(per_day):
            gb = rng.exponential(1.0)
            hold = rng.normal(0.0, 1.0)
            edge = -hold + (planted if gb > 1.2 else 0.0)
            rows.append({"date": d, "k": 6, "split": "train" if d <= "2025-06-30" else "test",
                         "open_R": rng.normal(2, 1), "giveback_R": gb, "stall_min": rng.exponential(1),
                         "cushion_R": rng.normal(1, 1), "vol_decay": rng.lognormal(0, .5),
                         "edge_exit_now": edge, "edge_trail_1m": edge * 0.5, "edge_switch": -edge * 0.2,
                         "r_ps": 0.2, "max_size": 1000})
    return pd.DataFrame(rows)


def test_holm_step_down():
    assert A.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert A.holm([0.5]) == [0.5]


def test_cluster_boot_resamples_dates_not_rows():
    x = np.array([1.0] * 10 + [-1.0])
    d = np.array(["a"] * 10 + ["b"])          # one big date, one small: 2 clusters
    s = A.cluster_boot(x, d, b=2000)
    assert s["mean"] == pytest.approx(9 / 11)
    assert s["lo"] < 0                        # 2 clusters can't give a tight CI


def test_ineligible_when_too_few_survivors():
    p = panel(n_days=60)
    assert A.eligible(p) == []
    assert A.run_protocol({"quick": p})["candidates"] == []


def test_noise_is_killed_at_stage_a_or_b():
    res = A.run_protocol({"quick": panel(planted=0.0, seed=1)})
    assert res["passed"] == []


def test_planted_edge_is_found_in_the_right_cell():
    res = A.run_protocol({"quick": panel(planted=1.5, seed=2)})
    assert res["passed"], "a +1.5R edge in the top giveback tercile should pass"
    top = res["passed"][0]["cell"]
    assert top.feature == "giveback_R" and top.bucket == 2


def test_test_set_untouched_when_stage_a_finds_nothing(monkeypatch):
    p = panel(planted=0.0, seed=3)
    p.loc[p["split"] == "train", ["edge_exit_now", "edge_trail_1m", "edge_switch"]] = -1.0   # nothing beats hold
    called = []
    monkeypatch.setattr(A, "stage_b", lambda *a, **k: called.append(1) or [])
    res = A.run_protocol({"quick": p})
    assert res["candidates"] == [] and called == []


def test_placebo_is_a_true_null_even_when_a_real_edge_exists():
    """Shuffling features alone would let every bucket inherit a real overall
    edge; demeaning makes any pass a genuine false positive."""
    assert A.placebo({"quick": panel(planted=1.5, seed=2)}, n=3) == 0


def test_tercile_edges_merge_ties():
    e = A.tercile_edges(pd.Series([0.0] * 90 + [1.0, 2.0, 3.0]))
    assert e[0] == -np.inf and e[-1] == np.inf and len(e) == 3


def test_holm_spans_both_rules():
    """A second rule doubles the Stage-A candidates; the Holm correction must
    cover all of them, not each rule separately."""
    quick, delayed = panel(planted=1.5, seed=2), panel(planted=1.5, seed=4)
    res = A.run_protocol({"quick": quick, "delayed": delayed})
    rules = {r["cell"].rule for r in res["candidates"]}
    assert rules == {"quick", "delayed"}
    ps = [r["test_p"] for r in res["candidates"]]
    assert [r["test_p_holm"] for r in res["candidates"]] == A.holm(ps)
    assert all(r["cell"].name.startswith(f"[{r['cell'].rule}]") for r in res["candidates"])
