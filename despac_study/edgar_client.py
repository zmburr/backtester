"""Polite SEC EDGAR client: full-text search, submissions JSON, filing text.

All GETs are throttled to EDGAR_MIN_INTERVAL and disk-cached under
data/despac_study/edgar_cache (immutable historical data, infinite TTL).
"""

import hashlib
import html as html_lib
import json
import logging
import re
import time

import requests

from despac_study.config import EDGAR_CACHE_DIR, EDGAR_MIN_INTERVAL, EDGAR_UA

logger = logging.getLogger(__name__)

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{}"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/{doc}"

_session = requests.Session()
_session.headers.update({"User-Agent": EDGAR_UA, "Accept-Encoding": "gzip, deflate"})
_last_request = [0.0]


def _throttled_get(url, params=None, timeout=30):
    wait = EDGAR_MIN_INTERVAL - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_request[0] = time.time()
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            return r
        except requests.RequestException as e:
            logger.warning("EDGAR GET failed (%s), attempt %d: %s", url, attempt, e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"EDGAR GET failed after retries: {url}")


def _disk_cache_get(key: str):
    path = EDGAR_CACHE_DIR / key
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _disk_cache_put(key: str, text: str):
    (EDGAR_CACHE_DIR / key).write_text(text, encoding="utf-8")


def fts_page(query: str, forms: str, startdt: str, enddt: str, frm: int) -> dict:
    """One page (10 hits) of EDGAR full-text search."""
    key = "fts_" + hashlib.md5(
        f"{query}|{forms}|{startdt}|{enddt}|{frm}".encode()
    ).hexdigest() + ".json"
    cached = _disk_cache_get(key)
    if cached is not None:
        return json.loads(cached)
    params = {"q": query, "forms": forms, "dateRange": "custom",
              "startdt": startdt, "enddt": enddt}
    if frm:
        params["from"] = frm
    r = _throttled_get(FTS_URL, params=params)
    r.raise_for_status()
    data = r.json()
    _disk_cache_put(key, json.dumps(data))
    return data


def fts_sweep(query: str, forms: str, startdt: str, enddt: str):
    """Yield every hit for a query in a date window, bisecting if the window
    exceeds the 10k pagination cap."""
    from despac_study.config import FTS_MAX_HITS

    first = fts_page(query, forms, startdt, enddt, 0)
    total = first.get("hits", {}).get("total", {}).get("value", 0)
    if total > FTS_MAX_HITS:
        import datetime as dt
        s = dt.date.fromisoformat(startdt)
        e = dt.date.fromisoformat(enddt)
        if (e - s).days <= 1:
            logger.warning("window %s..%s still >%d hits; taking first 10k",
                           startdt, enddt, FTS_MAX_HITS)
        else:
            mid = s + (e - s) / 2
            yield from fts_sweep(query, forms, startdt, mid.isoformat())
            yield from fts_sweep(query, forms,
                                 (mid + dt.timedelta(days=1)).isoformat(),
                                 enddt)
            return
    yield from first.get("hits", {}).get("hits", [])
    for frm in range(10, min(total, 9990), 10):
        page = fts_page(query, forms, startdt, enddt, frm)
        hits = page.get("hits", {}).get("hits", [])
        if not hits:
            break
        yield from hits


def get_submissions(cik) -> dict:
    """Company submissions JSON, with older-filing continuation pages merged
    into the 'recent' arrays."""
    cik10 = f"CIK{int(cik):010d}.json"
    key = f"sub_{int(cik):010d}.json"
    cached = _disk_cache_get(key)
    if cached is not None:
        return json.loads(cached)
    r = _throttled_get(SUBMISSIONS_URL.format(cik10))
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    for extra in data.get("filings", {}).get("files", []):
        r2 = _throttled_get(SUBMISSIONS_URL.format(extra["name"]))
        if r2.status_code != 200:
            continue
        older = r2.json()
        for col, vals in older.items():
            if col in recent and isinstance(vals, list):
                recent[col] = recent[col] + vals
    _disk_cache_put(key, json.dumps(data))
    return data


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")


def html_to_text(raw: str) -> str:
    txt = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    txt = re.sub(r"(?i)</(p|div|tr|td|th|br|li|h\d)>", "\n", txt)
    txt = _TAG_RE.sub(" ", txt)
    txt = html_lib.unescape(txt)
    txt = _WS_RE.sub(" ", txt)
    return re.sub(r"\n\s*\n+", "\n", txt)


def fetch_filing_text(cik, adsh: str, doc: str) -> str:
    """Primary document of a filing, HTML-stripped to text."""
    adsh_nodash = adsh.replace("-", "")
    key = f"doc_{int(cik)}_{adsh_nodash}_{hashlib.md5(doc.encode()).hexdigest()[:8]}.txt"
    cached = _disk_cache_get(key)
    if cached is not None:
        return cached
    url = ARCHIVES_URL.format(cik=int(cik), adsh=adsh_nodash, doc=doc)
    r = _throttled_get(url)
    if r.status_code != 200:
        # do NOT cache failures - a transient 403/429 would otherwise
        # permanently blank this filing's text
        logger.warning("filing doc %s -> HTTP %d", url, r.status_code)
        return ""
    text = html_to_text(r.text)
    _disk_cache_put(key, text)
    return text
