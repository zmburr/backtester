"""Stage 02b: LLM fallback for deals where regex found no redemption figures.

Pulls the cached super-8-K / vote-8-K texts, extracts just the sentences
mentioning redemption, and asks the repo LLM router for structured figures.
Updates despac_timeline.csv in place (only rows that were missing data).

Run:  python -m despac_study.llm_redemptions [--limit N]
"""

import argparse
import asyncio
import json
import logging
import re

import pandas as pd

from despac_study.config import TIMELINE_CSV
from despac_study.edgar_client import fetch_filing_text, get_submissions

logger = logging.getLogger(__name__)

PROMPT = """From these excerpts of SEC filings for a SPAC merger, extract:
- redeemed_shares: total public shares redeemed in connection with the merger vote (integer, 0 if it says none/de minimis)
- redemption_pct: redemptions as % of public shares if stated (number, else null)
- trust_per_share: redemption price per share in dollars (number, else null)
Reply with ONLY a JSON object with those three keys.

Excerpts:
"""


def redemption_windows(text: str, max_chars: int = 3500) -> str:
    """Sentences around 'redeem' mentions, deduped, capped."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.;])\s+", text)
    keep, seen = [], set()
    for i, s in enumerate(sents):
        if re.search(r"redee?m|redemption", s, re.I) and re.search(r"\d", s):
            w = " ".join(sents[max(0, i - 1):i + 2])[:600]
            key = w[:80]
            if key not in seen:
                seen.add(key)
                keep.append(w)
        if sum(len(k) for k in keep) > max_chars:
            break
    return "\n---\n".join(keep)[:max_chars]


def deal_texts(row) -> str:
    cik = int(row["cik"])
    try:
        subs = get_submissions(cik)
    except Exception:
        return ""
    rec = subs.get("filings", {}).get("recent", {})
    fl = pd.DataFrame({
        "form": rec.get("form", []), "filingDate": rec.get("filingDate", []),
        "adsh": rec.get("accessionNumber", []), "doc": rec.get("primaryDocument", []),
    })
    chunks = []
    sup = fl[fl["adsh"] == row["super8k_adsh"]]
    if not sup.empty and isinstance(sup.iloc[0]["doc"], str):
        chunks.append(redemption_windows(
            fetch_filing_text(cik, row["super8k_adsh"], sup.iloc[0]["doc"])))
    if isinstance(row.get("vote_8k_date"), str):
        v = fl[(fl["form"].str.startswith("8-K")) & (fl["filingDate"] == row["vote_8k_date"])]
        if not v.empty and isinstance(v.iloc[0]["doc"], str):
            chunks.append(redemption_windows(
                fetch_filing_text(cik, v.iloc[0]["adsh"], v.iloc[0]["doc"])))
    return "\n---\n".join(c for c in chunks if c)[:6000]


async def run(df, todo_idx):
    from support.llm_client import llm
    n_ok = 0
    for i in todo_idx:
        row = df.loc[i]
        excerpts = deal_texts(row)
        if not excerpts.strip():
            continue
        try:
            resp = await llm.chat(
                [{"role": "user", "content": PROMPT + excerpts}], tier="fast_foundation")
            m = re.search(r"\{.*\}", resp or "", re.S)
            if not m:
                continue
            data = json.loads(m.group(0))
        except Exception as e:
            logger.warning("LLM failed cik=%s: %s", row["cik"], e)
            continue
        rs = data.get("redeemed_shares")
        if rs is not None and pd.isna(row.get("redeemed_shares")):
            try:
                df.loc[i, "redeemed_shares"] = int(rs)
                df.loc[i, "redemption_source"] = "llm"
                if row.get("has_425") or row.get("name_change_near_close"):
                    df.loc[i, "is_spac"] = True
                n_ok += 1
            except (TypeError, ValueError):
                pass
        rp = data.get("redemption_pct")
        if rp is not None and pd.isna(row.get("redemption_pct")):
            try:
                v = float(rp)
                if 0 <= v <= 100:
                    df.loc[i, "redemption_pct"] = v
            except (TypeError, ValueError):
                pass
        tp = data.get("trust_per_share")
        if tp is not None and pd.isna(row.get("trust_per_share")):
            try:
                v = float(tp)
                if 9 <= v <= 25:
                    df.loc[i, "trust_per_share"] = v
            except (TypeError, ValueError):
                pass
        if n_ok and n_ok % 25 == 0:
            logger.info("llm redemptions: %d filled", n_ok)
            df.to_csv(TIMELINE_CSV, index=False)
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = pd.read_csv(TIMELINE_CSV)
    # include borderline rows (425/rename but no text evidence yet): an LLM
    # redemption hit is SPAC evidence and promotes them into the study set
    cand = (df["is_spac"] == True) | (
        (df["has_425"] == True) | (df["name_change_near_close"] == True))
    todo = df[cand & df["redeemed_shares"].isna() & df["redemption_pct"].isna()].index
    if args.limit:
        todo = todo[:args.limit]
    logger.info("LLM pass on %d deals missing redemption figures", len(todo))
    n = asyncio.run(run(df, list(todo)))
    df.to_csv(TIMELINE_CSV, index=False)
    print(f"filled {n} deals; redeemed_shares coverage now "
          f"{df[df['is_spac'] == True]['redeemed_shares'].notna().sum()}"
          f"/{(df['is_spac'] == True).sum()}")


if __name__ == "__main__":
    main()
