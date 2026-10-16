"""
Web Push to Method on the phone's home screen (Aspen's pattern, aspen/services/push.py).

The VAPID key pair is made once and kept in METHOD_HOME (default ~/.method, outside
the repo the mini auto-commits): private key PEM chmod 600, public key text, and the
subscriptions as JSON. iOS only delivers to a home-screen app, permission is asked
from a tap, and every push must show a notification (the service worker always does),
or iOS drops the subscription.

Two alerts use it (user, 2026-10-15): "setup forming" (the playbook email's gate,
which now pushes instead of emailing when a phone is subscribed) and "grade upgrade"
(a report GO/CAUTION name reaching A or A+ overnight or premarket).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("method.push")

VAPID_SUB = "mailto:zmburr@gmail.com"


def home() -> Path:
    return Path(os.environ.get("METHOD_HOME") or Path.home() / ".method")


def _paths() -> tuple[Path, Path, Path]:
    h = home()
    return h / "vapid_private.pem", h / "vapid_public.txt", h / "push_subscriptions.json"


def ensure_keys() -> str:
    """Create the VAPID pair on first use; return the public key (base64url, uncompressed point)."""
    priv, pub, _ = _paths()
    if not priv.exists():
        from cryptography.hazmat.primitives import serialization
        from py_vapid import Vapid02
        priv.parent.mkdir(parents=True, exist_ok=True)
        v = Vapid02()
        v.generate_keys()
        v.save_key(str(priv))
        os.chmod(priv, 0o600)
        raw = v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        pub.write_text(base64.urlsafe_b64encode(raw).rstrip(b"=").decode())
    return pub.read_text().strip()


def _load_subs() -> list[dict]:
    try:
        return json.loads(_paths()[2].read_text())
    except (OSError, ValueError):
        return []


def _save_subs(subs: list[dict]) -> None:
    path = _paths()[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(subs, indent=1))
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def subscribe(endpoint: str, keys: dict) -> int:
    if not str(endpoint).startswith("https://") or not keys.get("p256dh") or not keys.get("auth"):
        raise ValueError("Bad push subscription.")
    subs = [s for s in _load_subs() if s["endpoint"] != endpoint]
    subs.append({"endpoint": endpoint, "keys": {"p256dh": keys["p256dh"], "auth": keys["auth"]}})
    _save_subs(subs)
    return len(subs)


def count() -> int:
    return len(_load_subs())


def _webpush(sub: dict, payload: str, vapid, claims: dict) -> None:
    from pywebpush import webpush
    webpush(sub, payload, vapid_private_key=vapid, vapid_claims=claims, ttl=3600, timeout=10)


def send(title: str, body: str, url: str = "/", tag: str = "method") -> int:
    """One notification to every subscribed device; returns how many it reached.
    Subscriptions the push service says are gone (404/410) are dropped."""
    subs = _load_subs()
    priv = _paths()[0]
    if not subs or not priv.exists():
        return 0
    from py_vapid import Vapid02
    vapid = Vapid02.from_file(str(priv))
    payload = json.dumps({"title": title, "body": body[:240], "url": url, "tag": tag})
    sent, keep = 0, []
    for s in subs:
        try:
            _webpush(s, payload, vapid, {"sub": VAPID_SUB})
            sent += 1
            keep.append(s)
        except Exception as exc:                      # pywebpush.WebPushException and transport errors
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (404, 410):
                log.info("push subscription gone (%s); dropped", status)
                continue
            log.warning("push to %s failed: %s", s["endpoint"][:40], exc)
            keep.append(s)
    if len(keep) != len(subs):
        _save_subs(keep)
    return sent
