"""Shared test data. Addendum F.4: fixtures are the merge-conflict insurance.

The confidence model cannot train until the solver produces benchmark runs, and
that lands late (hour 30+). Rather than wait, the perception contributor builds
and tests confidence/features.py and confidence/train.py entirely against
fake_fits.py, then swaps the data source when the real benchmark arrives.

Same trick in reverse: geo/ is exercised against fake_photo_evidence.py long
before any segmentation model exists.
"""
