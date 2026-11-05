"""support.config.send_email: logs in as the sending account (not the inbox it mails to)
and says whether Gmail took the message. No network: smtplib is stubbed."""

import smtplib

import pytest

from support import config


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.logins, self.sent = [], []
        FakeSMTP.instances.append(self)

    def ehlo(self):
        pass

    def starttls(self):
        pass

    def login(self, user, password):
        if password != "abcdefghijklmnop":
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
        self.logins.append(user)

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        pass


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(config.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def test_sends_from_the_agent_account_to_the_inbox(smtp, monkeypatch):
    monkeypatch.setenv("GMAIL_PASSWORD", "abcd efgh ijkl mnop")      # as Google shows it
    assert config.send_email("zmburr@gmail.com", "s", "b") is True
    s = smtp.instances[-1]
    assert s.logins == [config.GMAIL_USER] and config.GMAIL_USER == "agentzachary6@gmail.com"
    assert s.sent[0]["From"] == config.GMAIL_USER and s.sent[0]["To"] == "zmburr@gmail.com"


def test_a_refused_login_or_no_password_returns_false(smtp, monkeypatch):
    monkeypatch.setenv("GMAIL_PASSWORD", "old-revoked-pass")
    assert config.send_email("zmburr@gmail.com", "s", "b") is False
    monkeypatch.delenv("GMAIL_PASSWORD")
    assert config.send_email("zmburr@gmail.com", "s", "b") is False
