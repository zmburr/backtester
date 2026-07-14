"""Stage 04: sector bucket + high-tech flag per deal.

Keyword rules over Polygon company description + SIC description first;
ambiguous names go to the repo LLM router (support.llm_client) in batches.

Run:  python -m despac_study.classify [--no-llm]
"""

import argparse
import asyncio
import json
import logging
import re

import pandas as pd

from despac_study.config import CLASSIFIED_CSV, ENRICHED_CSV, HIGHTECH_BUCKETS, SECTOR_BUCKETS
from despac_study.polygon_enrich import get_details

logger = logging.getLogger(__name__)

KEYWORDS = {
    "ev_mobility": ["electric vehicle", " ev ", "ev charging", "charging network", "battery", "batteries",
                    "lidar", "autonomous driv", "self-driving", "mobility", "automotive", "e-bike", "scooter",
                    "air taxi", "evtol", "urban air"],
    "space_defense": ["space", "satellite", "launch vehicle", "rocket", "defense", "aerospace", "drone",
                      "hypersonic", "orbital"],
    "crypto": ["bitcoin", "crypto", "blockchain", "digital asset", "web3", "nft"],
    "fintech": ["fintech", "payment", "payments", "neobank", "digital bank", "lending platform", "insurtech",
                "financial technology", "trading platform", "wealth management", "banking platform"],
    "software_ai": ["software", "saas", "artificial intelligence", " ai ", "machine learning", "data analytics",
                    "cloud", "cybersecurity", "semiconductor", "quantum", "internet platform", "e-commerce platform",
                    "marketplace platform", "developer"],
    "biotech_health": ["biopharma", "therapeutic", "clinical", "biotech", "pharmaceutical", "medical device",
                       "diagnostic", "healthcare", "health care", "telehealth", "genomic", "oncology", "drug"],
    "energy": ["solar", "renewable", "hydrogen", "fuel cell", "wind ", "oil ", "natural gas", "energy storage",
               "geothermal", "nuclear", "uranium", "carbon capture", "clean energy", "biofuel", "lithium",
               "mining", "minerals"],
    "industrial": ["manufactur", "industrial", "construction", "logistics", "supply chain", "shipping",
                   "trucking", "agricultur", "farming", "robotics", "3d print", "additive"],
    "consumer": ["consumer", "retail", "food", "beverage", "restaurant", "apparel", "fitness", "wellness",
                 "beauty", "pet ", "cannabis", "grocery", "brand"],
    "media_gaming": ["gaming", "esports", "game", "media", "entertainment", "streaming", "sports betting",
                     "casino", "music", "content", "social network", "advertising"],
    "real_estate": ["real estate", "reit", "property", "properties", "housing", "hospitality", "hotel"],
}

# SIC major-group hints (first 2 digits) used as a weak vote
SIC_HINTS = {
    "28": "biotech_health", "80": "biotech_health", "38": "biotech_health",
    "73": "software_ai", "35": "industrial", "36": "ev_mobility",
    "37": "ev_mobility", "13": "energy", "29": "energy", "10": "energy", "12": "energy",
    "48": "media_gaming", "78": "media_gaming", "79": "media_gaming",
    "60": "fintech", "61": "fintech", "62": "fintech", "63": "fintech",
    "65": "real_estate", "67": "real_estate",
    "58": "consumer", "54": "consumer", "56": "consumer", "20": "consumer",
    "42": "industrial", "44": "industrial", "16": "industrial", "34": "industrial",
}


def _blob(row, desc: str) -> str:
    bits = [str(row.get("company_name", "")), str(row.get("polygon_name", "")),
            str(row.get("sic_description", "")), str(row.get("sic_desc", "")), desc or ""]
    return (" ".join(bits)).lower()


def rule_classify(row, desc: str):
    blob = " " + re.sub(r"\s+", " ", _blob(row, desc)) + " "
    scores = {}
    for bucket, kws in KEYWORDS.items():
        s = sum(blob.count(kw) for kw in kws)
        if s:
            scores[bucket] = s
    sic = str(row.get("sic_code") or row.get("sic_current") or "")[:2]
    hint = SIC_HINTS.get(sic)
    if hint:
        scores[hint] = scores.get(hint, 0) + 1
    if not scores:
        return None, 0
    best = max(scores, key=scores.get)
    runner = sorted(scores.values(), reverse=True)
    margin = runner[0] - (runner[1] if len(runner) > 1 else 0)
    return best, margin


async def llm_classify(rows) -> dict:
    """rows: list of (cik, name, desc). Returns {cik: bucket}."""
    from support.llm_client import llm
    out = {}
    for i in range(0, len(rows), 25):
        chunk = rows[i:i + 25]
        listing = "\n".join(f"{c}: {n} -- {d[:200]}" for c, n, d in chunk)
        prompt = (
            "Classify each company into exactly one bucket from this list:\n"
            + ", ".join(SECTOR_BUCKETS) +
            "\nReturn ONLY a JSON object mapping id to bucket, e.g. {\"123\": \"fintech\"}.\n\n"
            + listing
        )
        try:
            resp = await llm.chat([{"role": "user", "content": prompt}], tier="fast_foundation")
            m = re.search(r"\{.*\}", resp or "", re.S)
            if m:
                for k, v in json.loads(m.group(0)).items():
                    if v in SECTOR_BUCKETS:
                        out[int(k)] = v
        except Exception as e:
            logger.warning("LLM classify chunk failed: %s", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = pd.read_csv(ENRICHED_CSV)
    descs, buckets, margins = [], [], []
    for _, row in df.iterrows():
        t = row.get("new_ticker")
        det = get_details(t) if isinstance(t, str) and t else {}
        if not det and isinstance(t, str) and t and isinstance(row.get("flip_date"), str):
            det = get_details(t, date=row["flip_date"])
        desc = det.get("description", "") if det else ""
        descs.append(desc)
        b, m = rule_classify(row, desc)
        buckets.append(b)
        margins.append(m)
    df["sector"] = buckets
    df["sector_margin"] = margins

    need_llm = df[(df["sector"].isna()) | (df["sector_margin"] < 2)]
    if not args.no_llm and len(need_llm):
        rows = [(int(r["cik"]), str(r.get("polygon_name") or r.get("company_name")), d)
                for (_, r), d in zip(need_llm.iterrows(), [descs[i] for i in need_llm.index])]
        logger.info("LLM classifying %d ambiguous deals...", len(rows))
        got = asyncio.run(llm_classify(rows))
        df.loc[df["cik"].isin(got.keys()), "sector"] = df.loc[
            df["cik"].isin(got.keys()), "cik"].map(got)
    df["sector"] = df["sector"].fillna("other")
    df["is_hightech"] = df["sector"].isin(HIGHTECH_BUCKETS)
    df.to_csv(CLASSIFIED_CSV, index=False)
    print(f"\nWrote {len(df)} -> {CLASSIFIED_CSV}")
    print(df["sector"].value_counts().to_string())
    print(f"is_hightech: {df['is_hightech'].sum()}/{len(df)}")


if __name__ == "__main__":
    main()
