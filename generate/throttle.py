"""
Replicate calls, paced for a low-credit account.

Below $5 of credit Replicate allows 6 predictions a minute with a burst of 1
(Owen, 2026-09-19). A demo batch is 2 calls per building back to back, so
without pacing the second call of the first building already gets a 429.
Every call goes through run(): at least MIN_INTERVAL_S since the previous one,
and a 429 is retried with backoff instead of failing the building.
"""

from __future__ import annotations

import time

MIN_INTERVAL_S = 10.0
MAX_RETRIES = 5
BACKOFF_S = 12.0

_last_call = 0.0


def _is_rate_limited(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    return status == 429 or "429" in str(exc) or "throttled" in str(exc).lower()


def run(model: str, *, input: dict, sleep=time.sleep, clock=time.monotonic):
    """replicate.run with spacing and 429 retries. Files in `input` must be
    re-readable across retries — callers pass open handles, so rewind them."""
    import replicate

    global _last_call
    for attempt in range(MAX_RETRIES + 1):
        wait = MIN_INTERVAL_S - (clock() - _last_call)
        if wait > 0:
            sleep(wait)
        _last_call = clock()
        for v in input.values():
            for h in (v if isinstance(v, list) else [v]):
                if hasattr(h, "seek"):
                    h.seek(0)
        try:
            return replicate.run(model, input=input)
        except Exception as exc:  # noqa: BLE001 — only 429s are retried
            if not _is_rate_limited(exc) or attempt == MAX_RETRIES:
                raise
            delay = BACKOFF_S * (attempt + 1)
            print(f"  replicate: rate limited (429); retrying in {delay:.0f} s "
                  f"({attempt + 1}/{MAX_RETRIES})")
            sleep(delay)
