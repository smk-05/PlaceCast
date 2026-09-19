"""Perception half — addendum sections A and C.

The modules here ship with working NON-ML stand-ins so the pipeline runs
end-to-end from minute one. The real implementations (Grounded SAM 2 for
segmentation, silhouette render-and-compare, UniDepth for metric depth) are
developed on a separate machine and dropped in behind `--perception=real`.

Every stand-in is a legitimate fallback path, not a placeholder to be deleted:
if the learned components never arrive, the pipeline still produces correct
placements via geo/disambiguate.py's Filters 1-3.
"""
