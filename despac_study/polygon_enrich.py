"""Stage 03: Polygon reference enrichment.

Per deal:
  - ticker-change events -> old SPAC ticker, ticker right after the flip,
    and the EXACT flip date (the tradeable "ticker changes tonight" moment)
  - ticker details -> SIC/sector info, listing/delisting status
  - pre-vote Class A shares outstanding -> redemption % denominator and
    post-redemption float estimate

Polygon's /vX/reference/tickers/{T}/events endpoint isn't wrapped anywhere in
the repo, so it's wrapped here with requests.

Run:  python -m despac_study.polygon_enrich
"""

import argparse
import logging
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv

from despac_study.cache import load_cached, save_to_cache
from despac_study.config import ENRICHED_CSV, PROJECT_ROOT, TIMELINE_CSV

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("POLYGON_API_KEY")
BASE = "https://api.polygon.io"
_session = requests.Session()


def _get_json(path: str, cache_parts, **params):
    cached = load_cached(*cache_parts, **params)
    if cached is not None:
        return cached
    params["apiKey"] = API_KEY
    try:
        r = _session.get(BASE + path, params=params, timeout=20)
        data = r.json() if r.status_code == 200 else {"status": f"HTTP{r.status_code}"}
    except requests.RequestException as e:
        logger.warning("polygon GET %s failed: %s", path, e)
        return None
    save_to_cache(data, *cache_parts, **{k: v for k, v in params.items() if k != "apiKey"})
    return data


def get_ticker_events(ticker: str):
    d = _get_json(f"/vX/reference/tickers/{ticker}/events", ("events", ticker),
                  types="ticker_change")
    if not d or d.get("status") not in ("OK",):
        return None
    evs = (d.get("results") or {}).get("events") or []
    evs = [{"date": e["date"], "ticker": e["ticker_change"]["ticker"]}
           for e in evs if e.get("type") == "ticker_change"]
    return sorted(evs, key=lambda e: e["date"])


def get_details(ticker: str, date: str = None):
    params = {"date": date} if date else {}
    d = _get_json(f"/v3/reference/tickers/{ticker}", ("details", ticker, date or "latest"),
                  **params)
    if not d or d.get("status") != "OK":
        return {}
    return d.get("results") or {}


def symbol_at(cik, date: str) -> str:
    """Common-share symbol an SEC registrant traded under on a given date."""
    d = _get_json("/v3/reference/tickers", ("symat", int(cik), date),
                  cik=f"{int(cik):010d}", date=date, limit=20)
    for r in (d or {}).get("results") or []:
        if r.get("type") == "CS" and "." not in r.get("ticker", ""):
            return r["ticker"]
    # some listings are typed ADRC/ordinary etc.; take any bare symbol
    for r in (d or {}).get("results") or []:
        if "." not in r.get("ticker", "") and r.get("type") not in ("WARRANT", "UNIT", "RIGHT"):
            return r["ticker"]
    return ""


_NAME_SUFFIX = re.compile(r"\b(inc|corp|corporation|ltd|limited|plc|llc|holdings?|group|company|co|technologies|technology)\b\.?,?", re.I)


def search_new_listing(name: str, date: str, exclude: str):
    """Polygon ticker search by company name; candidates for a target that
    listed under a brand-new CIK (double-dummy deals)."""
    q = _NAME_SUFFIX.sub("", str(name)).strip(" ,.&")
    q = " ".join(q.split()[:3])
    if len(q) < 3:
        return []
    d = _get_json("/v3/reference/tickers", ("namesearch", q.lower(), date),
                  search=q, date=date, market="stocks", limit=10)
    out = []
    for r in (d or {}).get("results") or []:
        t = r.get("ticker", "")
        if t and t != exclude and "." not in t and r.get("type") in ("CS", "ADRC", None):
            out.append(t)
    return out[:4]


def first_trading_day(ticker: str, lo: str, hi: str):
    """First daily bar printed under a symbol in a window = ticker-flip day."""
    d = _get_json(f"/v2/aggs/ticker/{ticker}/range/1/day/{lo}/{hi}",
                  ("firstbar", ticker, lo, hi), limit=50, adjusted="true")
    res = (d or {}).get("results") or []
    if not res:
        return None
    return pd.Timestamp(res[0]["t"], unit="ms", tz="US/Eastern").strftime("%Y-%m-%d")


def pick_common_ticker(raw) -> str:
    """From an EDGAR ticker list like 'BODI,BODYW' pick the common share."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    toks = [t.strip().upper() for t in raw.replace(";", ",").split(",") if t.strip()]
    toks = [t for t in toks if "." not in t and "-" not in t]
    if not toks:
        return ""
    plain = [t for t in toks if not (len(t) == 5 and t[-1] in "WUR")]
    pool = plain or toks
    return min(pool, key=len)


_SYMBOL_RE = re.compile(r"Trading\s+Symbol[^A-Z]{0,40}([A-Z]{1,5})\b")


def old_ticker_from_text(text: str) -> str:
    """Best-effort SPAC symbol from an 8-K cover page ('Trading Symbol(s)')."""
    if not text:
        return ""
    m = _SYMBOL_RE.search(text[:4000])
    return m.group(1) if m else ""


def enrich_deal(row) -> dict:
    out = {"cik": row["cik"]}
    close = str(row.get("close_date") or row.get("super8k_date") or "")
    new_t = pick_common_ticker(row.get("ticker_current")) or pick_common_ticker(row.get("tickers"))
    if not new_t:
        nt = row.get("new_ticker_edgar")
        if isinstance(nt, str) and nt:
            new_t = nt
    out["new_ticker"] = new_t
    if not new_t or not close:
        out["enrich_status"] = "no_ticker"
        return out

    lo = (pd.Timestamp(close) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(close) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    # authoritative: what symbol did this CIK trade under before/after close
    pre_date = (pd.Timestamp(close) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    post_date = (pd.Timestamp(close) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    old_ticker = symbol_at(row["cik"], pre_date) or None
    flip_ticker = symbol_at(row["cik"], post_date) or None
    if flip_ticker and old_ticker and flip_ticker == old_ticker:
        flip_ticker = symbol_at(row["cik"], (pd.Timestamp(close) + pd.Timedelta(days=25)).strftime("%Y-%m-%d")) or flip_ticker
    if not old_ticker:
        ot = row.get("old_ticker_edgar")
        if isinstance(ot, str) and ot and ot != new_t:
            old_ticker = ot
    if not flip_ticker:
        flip_ticker = new_t

    # flip date: explicit ticker_change event if recorded, else first bar
    # printed under the post-close symbol
    flip_date = None
    evs = get_ticker_events(new_t) or []
    in_win = [e for e in evs if lo <= e["date"] <= hi and e["ticker"] == flip_ticker]
    if in_win:
        flip_date = in_win[0]["date"]
        out["flip_source"] = "ticker_event"
    out["flip_date"] = flip_date
    out["flip_ticker"] = flip_ticker
    out["old_ticker"] = old_ticker
    out["n_ticker_events"] = len(evs)

    det = get_details(new_t)
    if not det:
        # delisted tickers 404 on the current-details endpoint; ask as-of a
        # date just after the flip window instead
        det = get_details(new_t, date=hi)
    out["polygon_active"] = det.get("active")
    out["list_date"] = det.get("list_date")
    out["sic_code"] = det.get("sic_code")
    out["sic_description"] = det.get("sic_description")
    out["polygon_name"] = det.get("name")
    out["market_cap_now"] = det.get("market_cap")
    out["delisted_utc"] = det.get("delisted_utc")
    # entity listed under a brand-new CIK/ticker (double-dummy structures):
    # no flip event, but the listing date near close is the tradeable date
    probe_lo = (pd.Timestamp(lo) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    if not flip_date and flip_ticker and flip_ticker != old_ticker:
        # no explicit event recorded: first bar printed under the post-close
        # symbol is the flip day - but only if the symbol has NO bars before
        # the window (fresh symbol, not a pre-existing listing)
        fb = first_trading_day(flip_ticker, probe_lo, hi)
        if fb and fb >= lo:
            out["flip_date"] = fb
            out["flip_source"] = "first_aggs_bar"
        elif det.get("list_date") and lo <= det["list_date"] <= hi:
            out["flip_date"] = det["list_date"]
            out["flip_source"] = "list_date"

    if not out.get("flip_date"):
        # target listed under a brand-new CIK (double-dummy, e.g. GGPI->PSNY):
        # the old CIK's symbol never changes. Search Polygon by target name and
        # accept only a symbol whose FIRST bar falls in the flip window.
        cands = []
        for nm in (row.get("target_name"), row.get("company_name"), row.get("polygon_name")):
            if isinstance(nm, str) and nm.strip():
                cands += search_new_listing(nm, post_date, old_ticker or "")
        seen = set()
        for t in [c for c in cands if not (c in seen or seen.add(c))]:
            fb = first_trading_day(t, probe_lo, hi)
            if fb and fb >= lo:
                out["flip_date"] = fb
                out["flip_ticker"] = flip_ticker = t
                out["flip_source"] = "name_search"
                break

    # pre-vote public share count -> redemption denominator
    vote = row.get("vote_date")
    if isinstance(vote, str) and vote and old_ticker:
        asof = (pd.Timestamp(vote) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        det_old = get_details(old_ticker, date=asof)
        so = det_old.get("share_class_shares_outstanding")
        out["public_shares_pre_vote"] = so
        red = row.get("redeemed_shares")
        if so and pd.notna(red):
            pct = 100.0 * float(red) / so
            if 0 <= pct <= 100:
                out["redemption_pct_final"] = round(pct, 1)
                out["post_redeem_float_est"] = int(so - red)
    out["enrich_status"] = "ok" if out.get("flip_date") else "no_flip_event"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tl = pd.read_csv(TIMELINE_CSV)
    if args.limit:
        tl = tl.head(args.limit)
    rows = []
    for i, row in tl.iterrows():
        rows.append(enrich_deal(row))
        if (i + 1) % 50 == 0:
            logger.info("enriched %d/%d", i + 1, len(tl))
    enr = pd.DataFrame(rows)
    merged = tl.merge(enr, on="cik", how="left")
    # prefer the regex-stated pct; fall back to the Polygon-denominator estimate
    if "redemption_pct_final" in merged.columns:
        merged["redemption_pct_best"] = merged["redemption_pct_final"].fillna(
            merged["redemption_pct"])
    else:
        merged["redemption_pct_best"] = merged["redemption_pct"]
    merged.to_csv(ENRICHED_CSV, index=False)
    n = len(merged)
    print(f"\nWrote {n} deals -> {ENRICHED_CSV}")
    print(f"  flip_date found:    {merged['flip_date'].notna().sum()}/{n}")
    print(f"  old_ticker found:   {merged['old_ticker'].notna().sum()}/{n}")
    print(f"  redemption_pct_best:{merged['redemption_pct_best'].notna().sum()}/{n}")


if __name__ == "__main__":
    main()
