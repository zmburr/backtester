"""Frozen-protocol analysis (PLAN.md §"Frozen protocol"). Stage A selects on
TRAIN; test outcomes are only computed for Stage-A candidates.

    orderPipe\\venv\\Scripts\\python -m checkpoint_study.analyze
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from checkpoint_study import config

FEATURES = ("open_R", "giveback_R", "stall_min", "cushion_R", "vol_decay")
POLICIES = ("exit_now", "trail_1m")
ELIG_TRAIN, ELIG_TEST = 150, 90
A_MIN_N, B_MIN_N, MIN_EDGE, MAX_CANDIDATES = 50, 30, 0.10, 6
ALPHA = 0.05
B = 10_000
SEED = 20260929
N_PLACEBO, PLACEBO_MAX_HITS = 20, 1
QUARTER_MIN_N, QUARTER_SHARE = 5, 0.75
ONE_R_DOLLARS = 3000.0
VARIANTS = ("floor_010", "floor_025", "atr_0", "atr_1", "anchor_m30", "anchor_p30", "no_filter",
            "ref_logged", "latency_30", "horizon_60", "horizon_240", "trail_close")


@dataclass(frozen=True)
class Cell:
    k: int
    policy: str
    feature: Optional[str] = None       # None = unconditional
    bucket: Optional[int] = None
    lo: float = -np.inf                 # (lo, hi] in the feature, from TRAIN quantiles
    hi: float = np.inf

    @property
    def name(self) -> str:
        if self.feature is None:
            return f"{self.k}m all -> {self.policy}"
        return f"{self.k}m {self.feature} ({self.lo:.3g}, {self.hi:.3g}] -> {self.policy}"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = df["k"] == self.k
        if self.feature is not None:
            v = df[self.feature]
            m &= (v > self.lo) & (v <= self.hi)
        return m


def cluster_boot(x: np.ndarray, dates: np.ndarray, seed: int = SEED, b: int = B) -> dict:
    """Mean with a 95% percentile CI resampling DATES; one-sided p = P(boot mean <= 0)."""
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    codes, uniq = pd.factorize(dates)
    s = np.bincount(codes, weights=x)
    c = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(b, len(uniq)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return {"n": int(n), "mean": float(x.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "p": float((means <= 0).mean())}


def holm(pvals: list) -> list:
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def tercile_edges(train_vals: pd.Series) -> list:
    v = train_vals.replace([np.inf, -np.inf], np.nan).dropna()
    q = sorted(set(np.quantile(v, [1 / 3, 2 / 3]).tolist())) if len(v) else []
    return [-np.inf] + q + [np.inf]


def build_cells(train: pd.DataFrame, ks) -> list:
    cells = []
    for k in ks:
        tk = train[train["k"] == k]
        for pol in POLICIES:
            cells.append(Cell(k, pol))
            for f in FEATURES:
                e = tercile_edges(tk[f])
                for bkt in range(len(e) - 1):
                    cells.append(Cell(k, pol, f, bkt, e[bkt], e[bkt + 1]))
    return cells


def stat(df: pd.DataFrame, cell: Cell, seed: int = SEED) -> dict:
    sub = df[cell.mask(df)]
    x = sub[f"edge_{cell.policy}"].to_numpy(dtype=float)
    ok = ~np.isnan(x)
    return cluster_boot(x[ok], sub["date"].to_numpy()[ok], seed)


def eligible(panel: pd.DataFrame) -> list:
    n = panel.groupby(["k", "split"]).size()
    return [k for k in config.CHECKPOINTS
            if n.get((k, "train"), 0) >= ELIG_TRAIN and n.get((k, "test"), 0) >= ELIG_TEST]


def stage_a(train: pd.DataFrame, cells: list) -> tuple:
    rows = []
    for c in cells:
        s = stat(train, c)
        rows.append({"cell": c, **{f"train_{k}": v for k, v in s.items()}})
    cand = [r for r in rows if r["train_n"] >= A_MIN_N and r["train_mean"] >= MIN_EDGE
            and r["train_lo"] > 0]
    cand = sorted(cand, key=lambda r: -r["train_lo"])[:MAX_CANDIDATES]
    return rows, cand


def stage_b(test: pd.DataFrame, cand: list) -> list:
    out = []
    for r in cand:
        s = stat(test, r["cell"])
        out.append({**r, **{f"test_{k}": v for k, v in s.items()}})
    adj = holm([r["test_p"] for r in out]) if out else []
    for r, a in zip(out, adj):
        r["test_p_holm"] = a
        r["stage_b_pass"] = bool(r["test_n"] >= B_MIN_N and r["test_mean"] >= MIN_EDGE
                                 and r["test_lo"] > 0 and a < ALPHA)
    return out


def robustness(test: pd.DataFrame, r: dict, variant_panels: dict) -> dict:
    c: Cell = r["cell"]
    sub = test[c.mask(test)]
    x = sub[f"edge_{c.policy}"].dropna()
    checks = {"median_or_share": bool(x.median() > 0 or (x > 0).mean() > 0.5)}
    q = sub.assign(q=pd.PeriodIndex(pd.to_datetime(sub["date"]), freq="Q"))
    qm = q.groupby("q")[f"edge_{c.policy}"].agg(["mean", "size"])
    qm = qm[qm["size"] >= QUARTER_MIN_N]
    checks["quarters_positive"] = bool(len(qm) and (qm["mean"] > 0).mean() >= QUARTER_SHARE)
    acct = sub[f"edge_{c.policy}"] * sub["r_ps"] * sub["max_size"] / ONE_R_DOLLARS
    checks["account_R"] = bool(acct.mean() > 0)
    for name, vp in variant_panels.items():
        vtrain, vtest = vp[vp["split"] == "train"], vp[vp["split"] == "test"]
        vc = c
        if c.feature is not None:
            e = tercile_edges(vtrain[vtrain["k"] == c.k][c.feature])
            b = min(c.bucket, len(e) - 2)
            vc = Cell(c.k, c.policy, c.feature, b, e[b], e[b + 1])
        vx = vtest[vc.mask(vtest)][f"edge_{c.policy}"].dropna()
        checks[f"variant_{name}"] = bool(len(vx) and vx.mean() > 0)
    checks["all"] = all(checks.values())
    return checks


def run_protocol(panel: pd.DataFrame) -> dict:
    ks = eligible(panel)
    train, test = panel[panel["split"] == "train"], panel[panel["split"] == "test"]
    if not ks:
        return {"eligible_k": [], "cells": [], "candidates": [], "passed": []}
    cells = build_cells(train, ks)
    rows, cand = stage_a(train, cells)
    tested = stage_b(test, cand) if cand else []
    return {"eligible_k": ks, "cells": rows, "candidates": tested,
            "passed": [r for r in tested if r["stage_b_pass"]]}


def placebo(panel: pd.DataFrame, n: int = N_PLACEBO) -> int:
    """Runs of the full protocol under a TRUE null that produce any pass.

    Features are permuted within k (no feature carries information) AND each
    edge is demeaned within (k, split) (no action beats holding on average).
    Shuffling alone isn't a null when the unconditional edge is real — every
    random bucket inherits it — so any pass here is a pipeline false positive."""
    hits = 0
    for i in range(n):
        rng = np.random.default_rng(SEED + 1 + i)
        p = panel.copy()
        for k in p["k"].unique():
            m = (p["k"] == k).to_numpy()
            perm = rng.permutation(m.sum())
            p.loc[m, list(FEATURES)] = p.loc[m, list(FEATURES)].to_numpy()[perm]
        for pol in POLICIES:
            col = f"edge_{pol}"
            p[col] = p[col] - p.groupby(["k", "split"])[col].transform("mean")
        hits += bool(run_protocol(p)["passed"])
    return hits


def load(name: str) -> Optional[pd.DataFrame]:
    p = config.DATA_DIR / "panels" / f"{name}.pkl"
    return pd.read_pickle(p) if p.exists() else None


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def main() -> dict:
    panel = load("primary")
    res = run_protocol(panel)
    variants = {v: load(v) for v in VARIANTS}
    missing = [v for v, p in variants.items() if p is None]
    for r in res["passed"]:
        r["robust"] = robustness(panel[panel["split"] == "test"], r,
                                 {k: v for k, v in variants.items() if v is not None})
    shipped = [r for r in res["passed"] if r.get("robust", {}).get("all")]
    placebo_hits = placebo(panel) if res["candidates"] else 0
    verdict = {
        "git": git_hash(), "run": pd.Timestamp.now(tz="US/Eastern").isoformat(),
        "survivors": panel.groupby(["k", "split"]).size().rename("n").reset_index().to_dict("records"),
        "eligible_k": res["eligible_k"],
        "n_cells": len(res["cells"]), "n_candidates": len(res["candidates"]),
        "n_stage_b_pass": len(res["passed"]), "n_robust": len(shipped),
        "variants_missing": missing, "placebo_hits": placebo_hits,
        "passed": bool(shipped) and placebo_hits <= PLACEBO_MAX_HITS and not missing,
    }
    verdict["decision"] = ("SHIP to shadow mode" if verdict["passed"] else
                           "KILL — checkpoint stays a neutral timer")
    out = config.REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    flat = lambda r: {"cell": r["cell"].name, **{k: v for k, v in r.items()
                                                  if k not in ("cell", "robust")},
                      **{f"robust_{k}": v for k, v in (r.get("robust") or {}).items()}}
    pd.DataFrame([flat(r) for r in res["cells"]]).to_csv(out / "cells_train.csv", index=False)
    pd.DataFrame([flat(r) for r in res["candidates"]]).to_csv(out / "candidates.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    return verdict


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
