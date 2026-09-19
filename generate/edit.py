"""
Image editing. Spec 5.1, FLUX.1 Kontext via Replicate.

THE PROMPT IS NOT FREE-FORM, and this is the design-level tension spec 10.1
names as the fundamental contradiction in the challenge: the task asks you to
*redesign* a building, then fit the result to the *real* footprint — but a
redesign that changes the silhouette no longer matches that footprint.

Resolution 1 (spec 10.1): constrain prompts to surface-level transformations.
That also happens to cover the post-apocalyptic weathering aesthetic Scorched
Nebraska actually needs, so it costs nothing.

    Good:      weathered concrete, broken windows, ivy overgrowth,
               scorched upper floors, rusted fixtures
    Dangerous: add a collapsed tower, extend the west wing, partially demolished

Resolution 3 is also implemented downstream: the silhouette divergence is
recorded as a first-class quantity in the placement record rather than treated
as error. In a game world, that is a feature.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests

# Official model; inputs (input_image, prompt, seed, output_format) verified
# against the live schema on 2026-09-19.
MODEL = "black-forest-labs/flux-kontext-pro"

# Spec 5.1: edits that mutate the silhouette break the fit. Rejected up front
# rather than discovered after a 60-second generation.
SILHOUETTE_MUTATING = re.compile(
    r"\b(collaps\w+|demolish\w+|add (a |an )?(tower|wing|floor|storey|story|extension)"
    r"|extend\w*|expand\w*|taller|shorter|rebuild|reshape|new roof shape)\b",
    re.IGNORECASE,
)

SAFE_SUFFIX = (
    "Preserve the building's exact silhouette, roofline, footprint and "
    "proportions. Change only surface materials and weathering."
)


def validate_prompt(prompt: str) -> tuple[bool, str]:
    """-> (ok, reason). Spec 5.1's guardrail, enforced before spending a call."""
    hit = SILHOUETTE_MUTATING.search(prompt)
    if hit:
        return False, (
            f"prompt contains a silhouette-mutating instruction ({hit.group(0)!r}). "
            "The mesh must still fit the authoritative footprint — constrain the "
            "edit to surface-level transformations (spec 5.1, 10.1)."
        )
    return True, ""


def edit_image(image_path: Path, prompt: str, out_path: Path,
               *, seed: int = 42, force: bool = False,
               strict: bool = True) -> Path:
    """Apply the redesign. Cached by out_path; pass force=True to regenerate.

    Determinism (spec 14): the seed is pinned and recorded. An irreproducible
    demo is a demo you cannot debug.
    """
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path

    ok, reason = validate_prompt(prompt)
    if not ok:
        if strict:
            raise ValueError(reason)
        prompt = f"{prompt}. {SAFE_SUFFIX}"

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Copy .env.example to .env and fill "
            "it in — see the hour-0 account list."
        )

    from generate import throttle

    with open(image_path, "rb") as f:
        output = throttle.run(
            MODEL,
            input={
                "input_image": f,
                "prompt": f"{prompt}. {SAFE_SUFFIX}",
                "seed": seed,
                "output_format": "png",
            },
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _download(output, out_path)
    return out_path


def _download(output, out_path: Path) -> None:
    """Replicate returns a URL, a file-like object, or a list of either."""
    if isinstance(output, list):
        output = output[0]

    if hasattr(output, "read"):
        out_path.write_bytes(output.read())
        return

    url = str(output)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
