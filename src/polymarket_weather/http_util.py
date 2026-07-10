"""http_util.py — the single GET-with-retry client shared by the fetchers (megaplan F5).

fetch_weather / fetch_ensemble / fetch_polymarket each carried a byte-identical (modulo a log
label) exponential-backoff retry helper, so a rate-limit or timeout fix had to be applied three
times and could drift. This is the one implementation; each fetcher keeps a thin `_get` wrapper
that supplies its label so its call sites are unchanged.
"""
import logging
import time

import requests

from config import RETRY_ATTEMPTS, RETRY_BACKOFF, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def get_json(url: str, params: dict | None = None, label: str = "API"):
    """GET *url* with exponential-backoff retry (config.RETRY_ATTEMPTS / RETRY_BACKOFF /
    REQUEST_TIMEOUT). Returns parsed JSON (dict or list), or None on a non-429 HTTP error or
    once retries are exhausted. Backs off and retries on HTTP 429 and transient request errors."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            if status == 429:
                wait = RETRY_BACKOFF ** (attempt + 2)
                logger.warning("%s rate-limit. Waiting %.1fs …", label, wait)
                time.sleep(wait)
            else:
                logger.error("HTTP %s from %s: %s", status, label, exc)
                return None
        except requests.exceptions.RequestException as exc:
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Request error (%s). Retry in %.1fs …", exc, wait)
            time.sleep(wait)
    logger.error("All %d attempts failed for %s", RETRY_ATTEMPTS, label)
    return None
