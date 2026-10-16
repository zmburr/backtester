"""
The bounce game plan's grade ladder (analyzers/bounce_ladder.py) in Method's
JSON shape. Grades are the game plan's: GO (5+/6) with intensity 75+ = A+,
60-75 = A, under 60 = B; 4/6 = B; 3/6 = C; 2/6 or less = no trade. The setup
type is fixed at the prior close, so the ladder is monotonic in price.
"""

from __future__ import annotations

from analyzers.bounce_ladder import GRADE_RANK, build_ladder, grade_zones, score_at  # noqa: F401


def ladder_payload(static: dict, cap: str, ticker: str) -> dict:
    """Every price where a criterion or the grade flips, the grade zones, and the ceiling."""
    lad = build_ladder(static, cap, ticker=ticker)
    return {
        "setup_type": lad.setup_type,
        "levels": [{"price": float(lv["price"]), "gap": float(lv["gap_pct"]), "grade": lv["grade"],
                    "score": int(lv["score"]), "intensity": lv["intensity"], "events": list(lv["events"])}
                   for lv in lad.levels],
        "zones": [{"grade": g, "top": float(top), "bottom": float(bottom)}
                  for g, top, bottom in grade_zones(lad)],
        "ceiling": {"score": int(lad.ceiling["score"]), "grade": lad.ceiling["grade"],
                    "blocked": list(lad.ceiling["blocked_desc"]),
                    "floor_price": float(lad.ceiling["floor_price"])},
    }


def next_better(levels: list[dict], price: float, grade: str) -> dict | None:
    """The first level below `price` whose grade beats `grade` (Ladder.next_level on the JSON)."""
    here = GRADE_RANK.get(grade, 0)
    for lv in levels:                       # high -> low
        if lv["price"] < price and GRADE_RANK.get(lv["grade"], 0) > here:
            return {"grade": lv["grade"], "price": lv["price"], "gap": lv["gap"],
                    "pct_away": lv["price"] / price - 1}
    return None


def grade_at(static: dict, cap: str, price: float, ticker: str) -> dict:
    """Score, intensity and grade at one price, exactly as the game plan and scanner score it."""
    return score_at(static, cap, price, ticker)
