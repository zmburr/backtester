"""Method's Hyperliquid watch (method/hl_watch.py): the hour's move, the three alerts and who
is left out of each, the cooldowns, and the loop's quiet hours, retries and saved state."""

import datetime
import json

import pytest

from method import hl_watch as hw

ET = hw.TZ
SAT = datetime.datetime(2026, 10, 17, 17, 0, tzinfo=ET).timestamp()       # markets shut
WED_RTH = datetime.datetime(2026, 10, 14, 10, 0, tzinfo=ET).timestamp()   # regular hours


def path(*pxs, end=SAT, step=300):
    """(ts, px) samples ending at `end`, `step` seconds apart."""
    return [(end - step * (len(pxs) - 1 - i), float(p)) for i, p in enumerate(pxs)]


def vols(*names, v=50e6):
    return {n: v for n in names}


def test_window_move_takes_the_larger_side():
    assert hw.window_move(path(100, 110, 104)) == pytest.approx(104 / 110 - 1)
    assert hw.window_move(path(100, 95, 103)) == pytest.approx(103 / 95 - 1)
    assert hw.window_move(path(100)) is None


def test_name_alert_only_for_liquid_us_listed_unleveraged_names():
    marks = {"RKLB": path(59.5, 58, 54.9), "KORU": path(10, 9.1), "MINIMAX": path(25, 23),
             "THIN": path(40, 37)}
    v = vols("RKLB", "KORU", "MINIMAX") | {"THIN": 1e6}
    alerts = hw.detect(marks, v, {}, SAT)
    assert [(a.kind, a.key) for a in alerts] == [("name", "name:RKLB:down")]
    assert alerts[0].title == "RKLB -7.7% in an hour"
    assert "59.50 to 54.90" in alerts[0].body


def test_breadth_market_alert_lists_the_names_and_covers_them():
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    marks = {n: path(100, 99, 94) for n in names} | {"KORU": path(10, 8.5)}
    alerts = hw.detect(marks, vols(*names, "KORU"), {}, SAT)
    assert [a.key for a in alerts] == ["market:down"]           # no name pushes, KORU not counted
    assert alerts[0].title == "6 names down 4%+ in an hour"
    sent = {}
    hw.record(alerts[0], sent, SAT)
    assert {"market:down", "name:AAA:down", "name:FFF:down"} <= set(sent)


def test_five_names_is_not_breadth():
    marks = {n: path(100, 94) for n in ["AAA", "BBB", "CCC", "DDD", "EEE"]}
    assert all(a.kind == "name" for a in hw.detect(marks, vols(*marks), {}, SAT))


def test_basket_market_alert_without_any_big_single_move():
    marks = {n: path(100, 99, 97.5) for n in hw.AI_BASKET}
    alerts = hw.detect(marks, vols(*marks), {}, SAT)
    assert [a.title for a in alerts] == ["AI stocks -2.5% in an hour"]


def test_index_alert():
    marks = {"XYZ100": path(29100, 28700), "SP500": path(7620, 7590)}
    alerts = hw.detect(marks, {}, {}, SAT)
    assert [(a.key, a.title) for a in alerts] == [("index:XYZ100:down", "XYZ100 -1.4% in an hour")]


def test_cooldown_holds_until_the_move_extends_or_three_hours_pass():
    sent = {}
    first = hw.detect({"RKLB": path(100, 94)}, vols("RKLB"), sent, SAT)
    hw.record(first[0], sent, SAT)
    assert hw.detect({"RKLB": path(100, 93)}, vols("RKLB"), sent, SAT + 600) == []
    deeper = hw.detect({"RKLB": path(100, 90.5)}, vols("RKLB"), sent, SAT + 600)  # 3%+ further
    assert [a.key for a in deeper] == ["name:RKLB:down"]
    later = SAT + hw.COOLDOWN_S + 1
    assert hw.detect({"RKLB": path(100, 94, end=later)}, vols("RKLB"), sent, later)


def test_names_stay_quiet_for_an_hour_after_a_market_alert():
    marks = {"IREN": path(25, 26.5)}
    sent = {"market:down": {"ts": SAT - 1800, "size": -0.033}}
    assert hw.detect(marks, vols("IREN"), sent, SAT) == []
    sent["market:down"]["ts"] = SAT - hw.WINDOW_S - 1
    assert [a.key for a in hw.detect(marks, vols("IREN"), sent, SAT)] == ["name:IREN:up"]


def test_step_is_quiet_in_regular_hours_and_forgets_the_window(tmp_path):
    w = hw.Watch(tmp_path, sender=lambda *a, **k: pytest.fail("pushed in regular hours"))
    w.marks = {"RKLB": path(100, end=WED_RTH - 60)}
    assert w.step({"RKLB": (90.0, 50e6)}, WED_RTH) == []
    assert w.marks == {}


def test_step_records_only_a_delivered_push_and_retries_after_a_wait(tmp_path):
    calls = []
    reach = [0]

    def sender(title, body, url, tag):
        calls.append((title, url, tag))
        return reach[0]

    w = hw.Watch(tmp_path, sender=sender)
    w.step({"RKLB": (100.0, 50e6)}, SAT)
    assert w.step({"RKLB": (94.0, 50e6)}, SAT + 60) == []         # no phone reached
    assert "name:RKLB:down" not in w.sent
    w.step({"RKLB": (94.0, 50e6)}, SAT + 90)                         # inside the retry wait
    assert len(calls) == 1
    reach[0] = 1
    fired = w.step({"RKLB": (94.0, 50e6)}, SAT + 60 + hw.RETRY_S)
    assert [a.key for a in fired] == ["name:RKLB:down"]
    assert calls[-1] == ("RKLB -6.0% in an hour", "/#/bounce", "name:RKLB:down")

    log = [json.loads(x) for x in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert [x["pushed"] for x in log] == [0, 1]
    again = hw.Watch(tmp_path)                                       # state survives a restart
    assert "name:RKLB:down" in again.sent and again.marks["RKLB"][-1][1] == 94.0


def test_marks_older_than_the_window_drop_off(tmp_path):
    w = hw.Watch(None)
    w.add({"RKLB": (100.0, 1.0)}, SAT)
    w.add({"RKLB": (99.0, 1.0), "MU": (800.0, 1.0)}, SAT + hw.WINDOW_S + 1)
    assert w.marks == {"RKLB": [(SAT + hw.WINDOW_S + 1, 99.0)], "MU": [(SAT + hw.WINDOW_S + 1, 800.0)]}


def test_rolling_volume_sums_the_trailing_day():
    day = 24 * 3600 * 1000
    candles = [{"t": 0, "c": "10", "v": "1"}, {"t": day // 2, "c": "20", "v": "1"},
               {"t": day, "c": "30", "v": "1"}]
    assert hw.rolling_volume(candles) == {0: (10.0, 10.0), day // 2: (20.0, 30.0), day: (30.0, 50.0)}
