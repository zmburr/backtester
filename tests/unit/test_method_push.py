"""Method's push: subscriptions, sending (pywebpush stubbed), the login gate, the
"setup forming" delivery that replaces the email, and the grade-upgrade alerts."""

import datetime
import json
import sys
import types
from unittest.mock import Mock

import pytest

from method import push


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("METHOD_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def fake_vapid(monkeypatch, home):
    """A key file plus a py_vapid stub, so send() runs without the real library."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "vapid_private.pem").write_text("key")
    mod = types.ModuleType("py_vapid")
    mod.Vapid02 = types.SimpleNamespace(from_file=lambda p: "vapid")
    monkeypatch.setitem(sys.modules, "py_vapid", mod)
    calls = []
    monkeypatch.setattr(push, "_webpush", lambda sub, payload, vapid, claims: calls.append((sub["endpoint"], json.loads(payload))))
    return calls


SUB = {"endpoint": "https://web.push.apple.com/abc", "keys": {"p256dh": "p", "auth": "a"}}


def test_subscribe_dedupes_and_validates(home):
    assert push.count() == 0
    assert push.subscribe(SUB["endpoint"], SUB["keys"]) == 1
    assert push.subscribe(SUB["endpoint"], SUB["keys"]) == 1        # same endpoint: replaced
    with pytest.raises(ValueError):
        push.subscribe("http://insecure", SUB["keys"])
    with pytest.raises(ValueError):
        push.subscribe(SUB["endpoint"], {"p256dh": "p"})


def test_send_reaches_every_device_and_drops_gone_ones(home, fake_vapid, monkeypatch):
    push.subscribe(SUB["endpoint"], SUB["keys"])
    push.subscribe("https://web.push.apple.com/gone", SUB["keys"])
    assert push.send("Title", "Body", url="/#/bounce", tag="t") == 2
    assert fake_vapid[0][1] == {"title": "Title", "body": "Body", "url": "/#/bounce", "tag": "t"}

    class Gone(Exception):
        response = types.SimpleNamespace(status_code=410)

    def flaky(sub, payload, vapid, claims):
        if sub["endpoint"].endswith("gone"):
            raise Gone()
    monkeypatch.setattr(push, "_webpush", flaky)
    assert push.send("T", "B") == 1
    assert push.count() == 1                                         # the 410 one is dropped


def test_send_without_subscribers_or_keys_is_a_no_op(home):
    assert push.send("T", "B") == 0


# ---------------------------------------------------------------- server

@pytest.fixture
def client(tmp_path, monkeypatch, home):
    from method import server as sv
    monkeypatch.setattr(sv.push, "ensure_keys", lambda: "PUBKEY")
    app = sv.create_app()
    app.testing = True
    return app.test_client()


def test_gate_fails_closed_without_your_login(client, monkeypatch):
    monkeypatch.delenv("METHOD_DEV_LOGIN", raising=False)
    assert client.get("/api/health").status_code == 200              # health stays open
    assert client.get("/api/push").status_code == 403
    assert client.get("/api/push", headers={"Tailscale-User-Login": "someone@else.com"}).status_code == 403
    assert client.get("/api/push", headers={"Tailscale-User-Login": "ZMBURR@gmail.com"}).status_code == 200


def test_writes_need_the_method_header(client, monkeypatch):
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    assert client.post("/api/push/subscribe", json=SUB).status_code == 400
    r = client.post("/api/push/subscribe", json=SUB, headers={"X-Method": "1"})
    assert r.get_json() == {"ok": True, "subscriptions": 1}
    assert client.get("/api/push").get_json() == {"public_key": "PUBKEY", "subscriptions": 1}


def test_service_worker_is_served_at_the_root(client, monkeypatch):
    monkeypatch.setenv("METHOD_DEV_LOGIN", "zmburr@gmail.com")
    r = client.get("/sw.js")
    assert r.status_code == 200 and b"showNotification" in r.data
    assert r.headers["Service-Worker-Allowed"] == "/"


# ---------------------------------------------------------------- setup forming

def test_setup_forming_pushes_instead_of_email_when_a_phone_is_subscribed(home, fake_vapid, monkeypatch):
    from scripts import playbook_email as pe
    mail = Mock()
    monkeypatch.setattr(pe, "send_email", mail)
    rows = [{"ticker": "SOXL", "rec": "GO", "grade": "B", "better": [("A", 92.22)]},
            {"ticker": "MXL", "rec": "GO", "grade": "B", "better": []},
            {"ticker": "NBIS", "rec": "CAUTION", "grade": "C"}]
    target = datetime.date(2026, 10, 16)
    assert pe.deliver("bounce", "evening", target, "subj", "<html>", rows) == "email"   # nobody subscribed
    assert mail.call_count == 1
    push.subscribe(SUB["endpoint"], SUB["keys"])
    assert pe.deliver("bounce", "evening", target, "subj", "<html>", rows) == "push"
    assert mail.call_count == 1
    msg = fake_vapid[-1][1]
    assert msg["title"] == "Bounce setup for Fri 10/16"
    assert msg["body"] == "SOXL B (A at 92.22) · MXL B. Open Method for the plan."
    assert msg["url"] == "/#/bounce"


def test_a_failed_push_still_emails(home, monkeypatch):
    from scripts import playbook_email as pe
    mail = Mock()
    monkeypatch.setattr(pe, "send_email", mail)
    monkeypatch.setattr(push, "count", lambda: 1)
    monkeypatch.setattr(push, "send", Mock(side_effect=RuntimeError("apns down")))
    assert pe.deliver("reversal", "morning", datetime.date(2026, 10, 16), "s", "b", []) == "email"
    mail.assert_called_once()


# ---------------------------------------------------------------- grade upgrades

def _row(t, grade, rec="GO", price=91.0):
    return {"ticker": t, "grade": grade, "report_rec": rec, "price": price, "gap": -0.09,
            "next": {"grade": "A+", "price": 85.0} if grade == "A" else None}


def test_grade_alerts_once_per_grade_and_only_report_names(tmp_path, monkeypatch):
    from method import refresh as rf
    monkeypatch.setattr(rf, "METHOD_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(rf.push, "send", lambda title, body, url, tag: sent.append((title, body, url)) or 1)
    rows = [_row("SOXL", "A"), _row("MXL", "B"), _row("KEEL", "A+", rec=None)]
    assert rf.grade_alerts(rows, "2026-10-16", "premarket") == ["SOXL"]
    assert sent[0] == ("SOXL is an A", "SOXL at 91.00 (-9.0% vs the close) is an A. A+ at 85.00.", "/#/bounce/SOXL")
    assert rf.grade_alerts(rows, "2026-10-16", "premarket") == []                 # already told
    rows[0] = _row("SOXL", "A+", price=84.0)
    assert rf.grade_alerts(rows, "2026-10-16", "overnight") == ["SOXL"]           # upgraded again
    assert rf.grade_alerts([_row("SOXL", "A+")], "2026-10-16", "market") == []     # market hours: the watcher's job


def test_grade_alerts_record_nothing_when_no_phone_got_it(tmp_path, monkeypatch):
    from method import refresh as rf
    monkeypatch.setattr(rf, "METHOD_DIR", tmp_path)
    monkeypatch.setattr(rf.push, "send", lambda *a, **k: 0)
    assert rf.grade_alerts([_row("SOXL", "A")], "2026-10-16", "premarket") == []
    assert not (tmp_path / "grade_alerts_2026-10-16.json").exists()
