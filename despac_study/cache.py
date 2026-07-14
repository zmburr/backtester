"""Infinite-TTL pickle cache (historical data is immutable).

Same idiom as options_replay/cache.py, pointed at data/despac_study/cache.
"""

import hashlib
import logging
import pickle
from pathlib import Path

from despac_study.config import CACHE_DIR

logger = logging.getLogger(__name__)


def _cache_key(*parts, **kwargs) -> str:
    bits = [str(p) for p in parts]
    for k, v in sorted(kwargs.items()):
        bits.append(f"{k}={v}")
    name = "_".join(bits).replace("/", "-").replace(":", "")
    if len(name) > 180:
        h = hashlib.md5(name.encode()).hexdigest()[:10]
        name = name[:160] + "_" + h
    return name + ".pkl"


def cache_path(*parts, **kwargs) -> Path:
    return CACHE_DIR / _cache_key(*parts, **kwargs)


def load_cached(*parts, **kwargs):
    path = cache_path(*parts, **kwargs)
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("cache read failed %s: %s", path.name, e)
            path.unlink(missing_ok=True)
    return None


def save_to_cache(data, *parts, **kwargs):
    path = cache_path(*parts, **kwargs)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.warning("cache write failed %s: %s", path.name, e)
