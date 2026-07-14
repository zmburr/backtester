"""Stage 02: per-deal EDGAR timeline + redemption extraction.

For every CIK in despac_universe.csv:
  - announcement date  = first Form 425 (fallback: first S-4/DEFM14A, flagged)
  - vote date          = reportDate of the LAST Item-5.07 8-K before close
                         (earlier 5.07 8-Ks are usually extension votes)
  - redemption figures = sentence-scoped extraction from the vote 8-K and
                         super 8-K texts (shares redeemed, shares remaining,
                         stated %, trust $/share)
  - SPAC confirmation  = filed 425s / SIC 6770 at close / renamed near close

Run:  python -m despac_study.edgar_timeline [--limit N]
"""

import argparse
import logging
import re

import pandas as pd

from despac_study.config import TIMELINE_CSV, UNIVERSE_CSV
from despac_study.edgar_client import fetch_filing_text, get_submissions

logger = logging.getLogger(__name__)

ANN_FORMS_FALLBACK = ["S-4", "DEFM14A", "PREM14A"]

_NUM = r"([\d,]{6,})"
_PCT = r"(\d{1,2}(?:\.\d+)?)\s*%"

RE_NO_REDEEM = re.compile(r"\bno\s+(?:public\s+)?(?:shares|shareholders|stockholders)[^.]{0,60}redeem|did not\s+(?:exercise|redeem)|de\s*minimis", re.I)
RE_REDEEMED = [
    re.compile(r"holders of (?:an aggregate of )?" + _NUM + r"[^.]{0,220}?redee?m", re.I),
    re.compile(r"exercised (?:their|its) (?:right|rights) to redeem (?:an aggregate of )?" + _NUM, re.I),
    re.compile(r"elected to redeem (?:an aggregate of )?" + _NUM, re.I),
    re.compile(_NUM + r"[^.]{0,120}?(?:were|have been|had been|was)\s+(?:validly\s+)?(?:submitted|tendered)?\s*(?:for\s+)?redee?m", re.I),
    re.compile(r"redemption of (?:an aggregate of )?" + _NUM + r"[^.]{0,60}(?:shares|stock)", re.I),
    re.compile(r"redee?m(?:ed)?\s+(?:an aggregate of\s+)?" + _NUM + r"\s+(?:public\s+)?(?:shares|Class A)", re.I),
    re.compile(r"in connection with the redemption[s]? of (?:an aggregate of )?" + _NUM, re.I),
    re.compile(r"(?:properly|validly) (?:exercised|elected)[^.]{0,80}?" + _NUM, re.I),
]
RE_REMAIN = re.compile(r"(?:after giving effect[^.]{0,160}?|remain(?:ed|ing)?[^.]{0,60}?)" + _NUM, re.I)
RE_PCT = re.compile(_PCT + r"[^.]{0,140}?(?:public shares|Class A|ordinary shares|outstanding shares)|redee?m[^.]{0,160}?" + _PCT
                    + r"|representing (?:approximately |about )?" + _PCT, re.I)
RE_TRUST_PS = re.compile(r"\$\s*(1?\d{1,2}\.\d{1,4})\s*per\s+(?:public\s+)?share", re.I)
RE_AGG_DOLLARS = re.compile(r"aggregate (?:redemption )?(?:amount|payment|of)[^.]{0,60}?\$\s*([\d,]+(?:\.\d+)?)\s*(million)?", re.I)


_COVER_STOP = {"NA", "NONE", "N", "A", "NYSE", "LLC", "INC", "CORP", "CO", "II", "III", "IV"}


def cover_ticker(text: str) -> str:
    """Common-share symbol from an 8-K cover page. The HTML table flattens to
    one cell per line: ... Trading Symbol(s) ... <class desc> / SYMBOL /
    <exchange>. Collect candidate symbol-lines after 'Trading Symbol' and take
    the shortest bare one (units are XXXX.U / XXXXU, warrants XXXXW)."""
    if not text:
        return ""
    m0 = re.search(r"Trading\s+Symbol", text)
    if not m0:
        return ""
    region = text[m0.start(): m0.start() + 900]
    cands = []
    for line in region.splitlines():
        tok = line.strip()
        if re.fullmatch(r"[A-Z]{2,6}(?:\.(?:U|WS|W|R))?", tok) and tok.split(".")[0] not in _COVER_STOP:
            cands.append(tok)
    bare = [c for c in cands if "." not in c]
    pool = bare or cands
    if not pool:
        return ""
    return min(pool, key=len).split(".")[0]


def _to_int(s):
    try:
        return int(s.replace(",", ""))
    except Exception:
        return None


def extract_redemptions(text: str) -> dict:
    """Sentence-scoped redemption extraction from one filing's text."""
    out = {"redeemed_shares": None, "remaining_shares": None,
           "redemption_pct_stated": None, "trust_per_share": None}
    if not text:
        return out
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.;])\s+", text)
    for sent in sentences:
        low = sent.lower()
        if "redee" not in low and "redemption" not in low:
            continue
        if RE_NO_REDEEM.search(sent):
            if out["redeemed_shares"] is None:
                out["redeemed_shares"] = 0
            continue
        for rx in RE_REDEEMED:
            m = rx.search(sent)
            if m:
                val = _to_int(m.group(1))
                # keep the largest plausible count (final redemption beats
                # partial mentions); ignore absurd values
                if val and val < 2_000_000_000:
                    if not out["redeemed_shares"] or val > out["redeemed_shares"]:
                        out["redeemed_shares"] = val
        m = RE_REMAIN.search(sent)
        if m and "remain" in low and "public" in low:
            val = _to_int(m.group(1))
            if val and val < 2_000_000_000:
                out["remaining_shares"] = val
        m = RE_AGG_DOLLARS.search(sent)
        if m:
            amt = float(m.group(1).replace(",", ""))
            if m.group(2):
                amt *= 1e6
            if amt > 1e5:
                out.setdefault("_agg_dollars", amt)
        for m in RE_PCT.finditer(sent):
            val = next((g for g in m.groups() if g), None)
            if val:
                v = float(val)
                if 0 < v <= 100 and (out["redemption_pct_stated"] is None or v > out["redemption_pct_stated"]):
                    out["redemption_pct_stated"] = v
        m = RE_TRUST_PS.search(sent)
        if m:
            v = float(m.group(1))
            if 9.0 <= v <= 25.0:
                out["trust_per_share"] = v
    # derive share count from aggregate $ paid out when count wasn't stated
    agg = out.pop("_agg_dollars", None)
    if agg and not out["redeemed_shares"]:
        ps = out["trust_per_share"] or 10.0
        out["redeemed_shares"] = int(agg / ps)
    return out


def _filings_frame(subs: dict) -> pd.DataFrame:
    rec = subs.get("filings", {}).get("recent", {})
    if not rec.get("form"):
        return pd.DataFrame()
    df = pd.DataFrame({
        "form": rec.get("form", []),
        "filingDate": rec.get("filingDate", []),
        "reportDate": rec.get("reportDate", []),
        "adsh": rec.get("accessionNumber", []),
        "doc": rec.get("primaryDocument", []),
        "items": rec.get("items", []),
    })
    return df


def process_deal(row) -> dict:
    cik = int(row["cik"])
    close_date = str(row["close_date"]) or str(row["super8k_date"])
    out = {"cik": cik}
    try:
        subs = get_submissions(cik)
    except Exception as e:
        logger.warning("submissions failed for CIK %s: %s", cik, e)
        out["error"] = str(e)
        return out

    out["ticker_current"] = ",".join(subs.get("tickers") or [])
    out["exchange"] = ",".join([e for e in (subs.get("exchanges") or []) if e])
    out["sic_current"] = subs.get("sic", "")
    out["sic_desc"] = subs.get("sicDescription", "")
    out["entity_name"] = subs.get("name", "")
    formers = subs.get("formerNames") or []
    out["former_names"] = "; ".join(f.get("name", "") for f in formers)
    name_change_near_close = False
    for f in formers:
        chg = (f.get("to") or "")[:10]
        if chg and close_date and abs((pd.Timestamp(chg) - pd.Timestamp(close_date)).days) < 400:
            name_change_near_close = True
    fl = _filings_frame(subs)
    if fl.empty:
        out["error"] = "no filings"
        return out

    # constrain the 425 search to the SPAC lifecycle (<=30 months pre-close)
    # so serial filers' unrelated old deals don't fake an early announcement
    ann_floor = (pd.Timestamp(close_date) - pd.DateOffset(months=30)).strftime("%Y-%m-%d") if close_date else "2019-01-01"
    f425 = fl[(fl["form"] == "425") & (fl["filingDate"] <= close_date) & (fl["filingDate"] >= ann_floor)]
    ann_date, ann_source = None, None
    if not f425.empty:
        ann_date, ann_source = f425["filingDate"].min(), "425"
    else:
        fb = fl[fl["form"].isin(ANN_FORMS_FALLBACK) & (fl["filingDate"] <= close_date)]
        if not fb.empty:
            ann_date, ann_source = fb["filingDate"].min(), fb.loc[fb["filingDate"].idxmin(), "form"]
    out["ann_date"] = ann_date
    out["ann_source"] = ann_source

    is8k = fl["form"].str.startswith("8-K")
    f507 = fl[is8k & fl["items"].fillna("").str.contains("5.07", regex=False)]
    close_cutoff = (pd.Timestamp(close_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d") if close_date else None
    if close_cutoff is not None:
        f507 = f507[f507["filingDate"] <= close_cutoff]
    out["n_vote_8ks"] = len(f507)
    vote_row = None
    if not f507.empty:
        vote_row = f507.sort_values("filingDate").iloc[-1]
        out["vote_8k_date"] = vote_row["filingDate"]
        rd = vote_row["reportDate"]
        out["vote_date"] = rd if isinstance(rd, str) and rd else vote_row["filingDate"]

    # SPAC confirmation (is_spac finalized after texts are read: 425s alone
    # also cover conventional stock-for-stock M&A)
    out["has_425"] = bool(len(f425))
    out["sic_6770_at_close"] = str(row.get("sic_at_filing", "")) in ("6770", "6770.0")
    out["name_change_near_close"] = name_change_near_close

    # redemption extraction: super 8-K first, vote 8-K fills gaps
    texts = []
    sup = fl[fl["adsh"] == row["super8k_adsh"]]
    if not sup.empty:
        doc = sup.iloc[0]["doc"]
        if isinstance(doc, str) and doc:
            texts.append(("super8k", fetch_filing_text(cik, row["super8k_adsh"], doc)))
    if vote_row is not None and isinstance(vote_row["doc"], str) and vote_row["doc"]:
        texts.append(("vote8k", fetch_filing_text(cik, vote_row["adsh"], vote_row["doc"])))
    # redemption figures sometimes land in a separate 8-K between vote and close
    if vote_row is not None and close_cutoff:
        near = fl[is8k & (fl["filingDate"] >= vote_row["filingDate"])
                  & (fl["filingDate"] <= close_cutoff)
                  & (fl["adsh"] != row["super8k_adsh"])
                  & (fl["adsh"] != vote_row["adsh"])].head(3)
        for _, r8 in near.iterrows():
            if isinstance(r8["doc"], str) and r8["doc"]:
                texts.append(("near8k", fetch_filing_text(cik, r8["adsh"], r8["doc"])))
    # target company name from the super 8-K (needed to find the new listing
    # when the target lists under a brand-new CIK/ticker, e.g. Polestar)
    out["target_name"] = ""
    for label, txt in texts:
        if label == "super8k" and txt:
            flat = re.sub(r"\s+", " ", txt[:20000])
            m = re.search(
                r"(?:business combination|merger|combination)[^.]{0,120}? (?:with|between|of) "
                r"(?:the Company and )?([A-Z][A-Za-z0-9 .,&'’-]{2,60}?)"
                r"(?:\s*\(|,? (?:a|an) [A-Z]?[a-z]|\.)", flat)
            if m:
                out["target_name"] = m.group(1).strip(" ,.")
            break

    # old SPAC ticker: cover page of the latest 8-K filed STRICTLY before the
    # close (covers filed on/after close already show the new symbol)
    out["old_ticker_edgar"] = ""
    pre = fl[is8k & (fl["filingDate"] < close_date)].sort_values("filingDate").tail(2)
    for _, r8 in pre[::-1].iterrows():
        if isinstance(r8["doc"], str) and r8["doc"]:
            t = cover_ticker(fetch_filing_text(cik, r8["adsh"], r8["doc"]))
            if t:
                out["old_ticker_edgar"] = t
                break
    # new ticker from the super 8-K cover (survives delistings where EDGAR /
    # Polygon no longer map the company)
    out["new_ticker_edgar"] = ""
    for label, txt in texts:
        if label == "super8k":
            out["new_ticker_edgar"] = cover_ticker(txt)
            break

    red = {"redeemed_shares": None, "remaining_shares": None,
           "redemption_pct_stated": None, "trust_per_share": None}
    src = None
    for label, txt in texts:
        r = extract_redemptions(txt)
        for k, v in r.items():
            if red.get(k) is None and v is not None:
                red[k] = v
                if k in ("redeemed_shares", "redemption_pct_stated") and src is None:
                    src = label
    out.update(red)
    out["redemption_source"] = src

    # SPAC evidence: trust-account language, extracted redemption figures, or
    # public-share redemption phrasing. Biotech reverse mergers rename the
    # registrant too and conventional M&A files 425s - but neither redeems
    # public shares from a trust.
    flat_texts = [re.sub(r"\s+", " ", (t or "")).lower() for _, t in texts]
    out["mentions_trust"] = any(
        ("trust account" in t or "held in trust" in t) for t in flat_texts)
    spac_lang = any(re.search(
        r"redee?m[^.]{0,120}?(?:public share|public stockholder|public shareholder|"
        r"class a common|class a ordinary|ordinary shares sold in)", t) for t in flat_texts)
    evidence = (out["mentions_trust"] or spac_lang
                or red["redeemed_shares"] is not None
                or red["redemption_pct_stated"] is not None)
    out["is_spac"] = (out["sic_6770_at_close"]
                      or (evidence and (out["has_425"] or name_change_near_close)))
    if red["redemption_pct_stated"] is not None:
        out["redemption_pct"] = red["redemption_pct_stated"]
    elif red["redeemed_shares"] is not None and red["remaining_shares"]:
        tot = red["redeemed_shares"] + red["remaining_shares"]
        out["redemption_pct"] = round(100 * red["redeemed_shares"] / tot, 1) if tot else None
    else:
        out["redemption_pct"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    uni = pd.read_csv(UNIVERSE_CSV)
    if args.limit:
        uni = uni.head(args.limit)
    rows = []
    for i, row in uni.iterrows():
        rows.append(process_deal(row))
        if (i + 1) % 25 == 0:
            logger.info("processed %d/%d deals", i + 1, len(uni))
            pd.DataFrame(rows).to_csv(TIMELINE_CSV.with_suffix(".partial.csv"), index=False)
    tl = pd.DataFrame(rows)
    merged = uni.merge(tl, on="cik", how="left")
    merged.to_csv(TIMELINE_CSV, index=False)
    n = len(merged)
    print(f"\nWrote {n} deals -> {TIMELINE_CSV}")
    print(f"  is_spac:          {merged['is_spac'].sum()}/{n}")
    print(f"  ann_date found:   {merged['ann_date'].notna().sum()}/{n}")
    print(f"  vote_date found:  {merged['vote_date'].notna().sum() if 'vote_date' in merged else 0}/{n}")
    print(f"  redemption_pct:   {merged['redemption_pct'].notna().sum()}/{n}")


if __name__ == "__main__":
    main()
