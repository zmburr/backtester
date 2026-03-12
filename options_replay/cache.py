"""
Local pickle cache for Theta Data API responses.
Historical options data never changes, so TTL is infinite.
"""

import pickle
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "options_replay_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(symbol: str, date: str, data_type: str, **kwargs) -> str:
    """Build a cache filename from parameters."""
    parts = [symbol.upper(), date.replace("-", ""), data_type]
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v}")
    name = "_".join(str(p) for p in parts)
    # Truncate long names via hash suffix
    if len(name) > 180:
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        name = name[:170] + "_" + h
    return name + ".pkl"


def get_cache_path(symbol: str, date: str, data_type: str, **kwargs) -> Path:
    return CACHE_DIR / _cache_key(symbol, date, data_type, **kwargs)


def load_cached(symbol: str, date: str, data_type: str, **kwargs):
    """Load cached DataFrame/object. Returns None on miss."""
    path = get_cache_path(symbol, date, data_type, **kwargs)
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.debug("Cache hit: %s", path.name)
            return data
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", path.name, e)
            path.unlink(missing_ok=True)
    return None


def save_to_cache(data, symbol: str, date: str, data_type: str, **kwargs):
    """Save DataFrame/object to cache."""
    path = get_cache_path(symbol, date, data_type, **kwargs)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Cached: %s", path.name)
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", path.name, e)
