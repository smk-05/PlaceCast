"""Which perception implementation runs: the real models (default) or the non-ML stand-ins.

    PROCEDURA_PERCEPTION=real   Grounding DINO + SAM 2 segmentation, silhouette render-and-compare (default)
    PROCEDURA_PERCEPTION=stub   the central-box mask and the abstaining silhouette scorer; no GPU, no torch

Resolution order: the explicit `backend=` argument, then the environment variable, then "real". Every perception
entry point takes `backend=`, so a caller's own flag (pipeline.py's `--perception`) must be passed through it:
the environment variable is only the fallback for callers that say nothing. The default is real, so a machine
without the ML stack must say `stub` explicitly: silently substituting a stand-in would leave someone believing
the models ran. The stand-ins are a legitimate fallback (geo/disambiguate.py's filters carry the decision without
them), not placeholders.

segment.segment_building_evidence() reports the backend that actually ran, for logging.
"""
from __future__ import annotations

import os

ENV_VAR = "PROCEDURA_PERCEPTION"
REAL, STUB = "real", "stub"


def resolve_backend(backend: str | None = None) -> str:
    """`backend` if given, else $PROCEDURA_PERCEPTION, else "real". Unknown values raise."""
    chosen = (backend if backend is not None else os.environ.get(ENV_VAR, "")).strip().lower() or REAL
    if chosen not in (REAL, STUB):
        raise ValueError(f"unknown perception backend {chosen!r}; use {REAL!r} or {STUB!r} (${ENV_VAR})")
    return chosen
