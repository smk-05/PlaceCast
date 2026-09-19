"""Which perception implementation runs: the real models (default) or the non-ML stand-ins.

    PROCEDURA_PERCEPTION=real   Grounding DINO + SAM 2 segmentation, silhouette render-and-compare (default)
    PROCEDURA_PERCEPTION=stub   the central-box mask and the abstaining silhouette scorer; no GPU, no torch

Every perception entry point also takes `backend=` to override the environment for one call. The default
is real, so a machine without the ML stack must say `stub` explicitly: silently substituting a stand-in
would leave someone believing the models ran. The stand-ins are a legitimate fallback (geo/disambiguate.py's
filters carry the decision without them), not placeholders.

The pipeline's own `--perception` flag does not reach this module today (pipeline.py imports the same
functions on both branches); until it does, set the environment variable.
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
