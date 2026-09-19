"""Replicate pacing: spacing between calls, and 429s retried, not fatal."""

from __future__ import annotations

import io
import sys
import types

import pytest

from generate import throttle


class _Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def sleep(self, s):
        self.slept.append(s)
        self.t += s

    def __call__(self):
        return self.t


class _E(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status = status


def _fake_replicate(monkeypatch, outcomes):
    calls = []

    def run(model, input):
        calls.append(input["images"][0].read())
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    monkeypatch.setitem(sys.modules, "replicate", types.SimpleNamespace(run=run))
    return calls


def test_429_is_retried_and_file_is_rewound(monkeypatch):
    calls = _fake_replicate(monkeypatch, [_E(429), "ok"])
    clock = _Clock()
    monkeypatch.setattr(throttle, "_last_call", 0.0)
    out = throttle.run("m", input={"images": [io.BytesIO(b"img")]},
                       sleep=clock.sleep, clock=clock)
    assert out == "ok"
    assert calls == [b"img", b"img"]          # second attempt re-read the file
    assert sum(clock.slept) >= throttle.BACKOFF_S


def test_other_errors_are_not_retried(monkeypatch):
    _fake_replicate(monkeypatch, [_E(422)])
    clock = _Clock()
    with pytest.raises(_E):
        throttle.run("m", input={"images": [io.BytesIO(b"x")]},
                     sleep=clock.sleep, clock=clock)


def test_back_to_back_calls_are_spaced(monkeypatch):
    _fake_replicate(monkeypatch, ["a", "b"])
    clock = _Clock()
    monkeypatch.setattr(throttle, "_last_call", 0.0)
    throttle.run("m", input={"images": [io.BytesIO(b"x")]}, sleep=clock.sleep, clock=clock)
    throttle.run("m", input={"images": [io.BytesIO(b"x")]}, sleep=clock.sleep, clock=clock)
    assert clock.slept and clock.slept[-1] == pytest.approx(throttle.MIN_INTERVAL_S)
