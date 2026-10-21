# -*- coding: utf-8 -*-
"""Apply text patches to main.tex on LF-normalised text, write back as CRLF."""
import io
import sys


def load(p="main.tex"):
    raw = io.open(p, encoding="utf-8", newline="").read()
    return raw.replace("\r\n", "\n")


def save(t, p="main.tex"):
    with io.open(p, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(t)
    raw = io.open(p, encoding="utf-8", newline="").read()
    assert "\r\n" in raw
    lone = raw.count("\n") - raw.count("\r\n")
    print("saved; lone LF lines =", lone)
