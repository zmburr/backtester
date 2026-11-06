"""Method's Watching list (method/pins.py): the list, which events push, and when a print
is fresh enough to push at all."""

import datetime

import pytest

from method import pins

ET = datetime.timezone(datetime.timedelta(hours=-5))
NOW = datetime.datetime(2026, 11, 5, 21, 5, tzinfo=ET)


@pytest.fixture(autouse=True)
def mdir(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_DIR", str(tmp_path))
    return tmp_path


def row(t="MU", grade="C", score=3, price=672.0, gap=0.0, nxt=None, src="Y", age_min=5):
    return {"ticker": t, "grade": grade, "score": score, "max_score": 6, "price": price, "gap": gap,
            "next": nxt, "src": src, "src_label": {"Y": "Blue Ocean", "P": "after hours"}.get(src, src),
            "bar_time": (NOW - datetime.timedelta(minutes=age_min)).strftime("%H:%M"),
            "asof": (NOW - datetime.timedelta(minutes=age_min)).isoformat()}


class Phone:
    def __init__(self, reach=1):
        self.reach, self.sent = reach, []

    def __call__(self, title, body, url="/", tag="method"):
        self.sent.append((title, body, url, tag))
        return self.reach


def test_pins_add_remove_and_persist(mdir):
    assert pins.tickers() == []
    assert pins.set_pin("mu", True) == ["MU"]
    assert pins.set_pin("SNDK", True) == ["MU", "SNDK"]
    assert pins.set_pin("MU", True) == ["MU", "SNDK"]                    # no duplicate
    assert pins.set_pin("MU", False) == ["SNDK"]
    assert (mdir / "pins.json").exists() and pins.tickers() == ["SNDK"]
    for junk in ("", "mu sndk", "1ABC", "<script>"):
        with pytest.raises(ValueError):
            pins.set_pin(junk, True)


def test_grade_gap_and_near_events_push_once_each():
    phone = Phone()
    nxt = {"grade": "B", "price": 641.32, "gap": -0.047}
    # 21:05, MU -4% on Blue Ocean: past the -3% step and within 1% of B
    r = row(price=646.0, gap=-0.0399, nxt=nxt)
    assert pins.pin_alerts([r], "2026-11-06", "overnight", phone, NOW, ["MU"]) == ["MU"]
    title, body, url, tag = phone.sent[0]
    assert title == "MU 646.00 (-4.0%)"
    assert body == ("MU 646.00 (-4.0% vs the close, Blue Ocean 21:00) · C 3/6 · B at 641.32. "
                    "-3% vs the close; 0.7% from B.")
    assert url == "/#/bounce/MU" and tag == "pin-MU-2026-11-06"
    # the same print again: nothing new
    assert pins.pin_alerts([r], "2026-11-06", "overnight", phone, NOW, ["MU"]) == []
    # it drops into B and through -5%: one push naming both
    r2 = row(grade="B", score=4, price=638.0, gap=-0.051, nxt={"grade": "A", "price": 578.18, "gap": -0.141})
    assert pins.pin_alerts([r2], "2026-11-06", "overnight", phone, NOW, ["MU"]) == ["MU"]
    assert "Now a B" in phone.sent[1][1] and "-5% vs the close" in phone.sent[1][1]
    assert len(phone.sent) == 2


def test_a_jump_through_several_steps_is_one_alert_and_marks_them_all():
    phone = Phone()
    r = row(grade="A", score=6, price=570.0, gap=-0.153)                  # straight to -15%
    pins.pin_alerts([r], "2026-11-06", "overnight", phone, NOW, ["MU"])
    body = phone.sent[0][1]
    assert "-12% vs the close" in body and "-3%" not in body and "-8%" not in body
    assert "Now an A" in body
    r2 = row(grade="A", score=6, price=580.0, gap=-0.138)                 # bounces back to -13.8%
    assert pins.pin_alerts([r2], "2026-11-06", "overnight", phone, NOW, ["MU"]) == []


def test_gap_up_steps_push_too():
    phone = Phone()
    pins.pin_alerts([row(price=706.0, gap=0.05)], "2026-11-06", "overnight", phone, NOW, ["MU"])
    assert "+5% vs the close" in phone.sent[0][1]


def test_pins_alert_whatever_the_report_said_and_only_pinned_names():
    phone = Phone()
    rows = [row("MU", grade="B", score=4, gap=-0.06), row("AMD", grade="A+", score=6, gap=-0.2)]
    rows[0]["report_rec"] = None                                          # MU wasn't in the report
    assert pins.pin_alerts(rows, "2026-11-06", "overnight", phone, NOW, ["MU"]) == ["MU"]


def test_only_fresh_prints_alert():
    phone = Phone()
    big = dict(grade="A", score=6, gap=-0.15)
    # overnight, the evening price carried forward (src P) is not news
    assert pins.pin_alerts([row(src="P", **big)], "2026-11-06", "overnight", phone, NOW, ["MU"]) == []
    # a 45-minute-old print is stale anywhere
    assert pins.pin_alerts([row(age_min=45, **big)], "2026-11-06", "premarket", phone, NOW, ["MU"]) == []
    # a premarket Polygon bar ~15 min behind is fine
    assert pins.pin_alerts([row(src="P", age_min=16, **big)], "2026-11-06", "premarket", phone, NOW, ["MU"]) == ["MU"]
    # market hours belong to the entry watcher
    assert pins.pin_alerts([row(**big)], "2026-11-06", "market", Phone(), NOW, ["MU"]) == []


def test_nothing_is_recorded_unless_a_phone_got_it(mdir):
    r = row(grade="B", score=4, gap=-0.06)
    assert pins.pin_alerts([r], "2026-11-06", "overnight", Phone(reach=0), NOW, ["MU"]) == []
    assert not (mdir / "pin_alerts_2026-11-06.json").exists()
    assert pins.pin_alerts([r], "2026-11-06", "overnight", Phone(), NOW, ["MU"]) == ["MU"]   # retried


def test_each_session_starts_fresh():
    phone = Phone()
    r = row(grade="B", score=4, gap=-0.06)
    pins.pin_alerts([r], "2026-11-06", "overnight", phone, NOW, ["MU"])
    assert pins.pin_alerts([r], "2026-11-09", "overnight", phone, NOW, ["MU"]) == ["MU"]


def test_board_shows_pending_names_until_their_statics_build(mdir):
    pins.set_pin("MU", True)
    pins.set_pin("NEWCO", True)
    b = pins.board([row("MU", grade="B", score=4, gap=-0.06)])
    assert b[0]["ticker"] == "MU" and b[0]["grade"] == "B" and "ladder" not in b[0]
    assert b[1] == {"ticker": "NEWCO", "pending": True}


# ---------------------------------------------------------------- refresh + server

def test_grade_alerts_leave_pinned_names_to_the_pin_alerts(tmp_path, monkeypatch):
    from method import refresh as rf
    monkeypatch.setattr(rf, "METHOD_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(rf.push, "send", lambda title, body, url, tag: sent.append(title) or 1)
    rows = [{"ticker": "SOXL", "grade": "A", "report_rec": "GO", "price": 40.0, "gap": -0.09, "next": None}]
    assert rf.grade_alerts(rows, "2026-11-06", "overnight", skip=["SOXL"]) == []
    assert rf.grade_alerts(rows, "2026-11-06", "overnight") == ["SOXL"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    from method import server as sv
    app = sv.create_app()
    app.testing = True
    return app.test_client()


def test_pins_api_needs_the_method_header_and_validates(client):
    assert client.get("/api/pins").get_json() == {"pins": []}
    assert client.post("/api/pins", json={"ticker": "MU"}).status_code == 400          # no X-Method
    h = {"X-Method": "1"}
    assert client.post("/api/pins", json={"ticker": "mu", "on": True}, headers=h).get_json() == {
        "ok": True, "pins": ["MU"], "board": [{"ticker": "MU", "pending": True}]}       # no snapshot yet
    assert client.post("/api/pins", json={"ticker": "SNDK"}, headers=h).get_json()["pins"] == ["MU", "SNDK"]
    assert client.post("/api/pins", json={"ticker": "MU", "on": False}, headers=h).get_json()["pins"] == ["SNDK"]
    assert client.post("/api/pins", json={"ticker": "not a ticker"}, headers=h).status_code == 400


def test_page_and_static_files_revalidate_so_a_deploy_reaches_the_phone(client):
    for path in ("/", "/static/app.js", "/static/app.css"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["Cache-Control"] == "no-cache", path
