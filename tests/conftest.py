"""Test-suite defaults.

Perception runs the REAL models unless told otherwise (perception/backend.py), which needs a GPU and the ML
stack. The suite must run on the geometry machine, so it defaults to the non-ML stand-ins. Tests that exercise
real-backend code pass `backend="real"` explicitly (with the models stubbed out) and never depend on this.
"""
import os

os.environ.setdefault("PROCEDURA_PERCEPTION", "stub")
os.environ.pop("PROCEDURA_LEARNED_GATE", None)  # the gate tests set what they need
os.environ.pop("PROCEDURA_GATE_MODEL", None)
