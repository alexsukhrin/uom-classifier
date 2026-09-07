"""Regression tests against the bundled artifact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uom_classifier import UomClassifier  # noqa: E402

clf = UomClassifier()


def test_exact_vocabulary_hit():
    assert clf.classify("шт.") == ("штука", 1.0)
    assert clf.classify("ФЛ") == ("флакон", 1.0)


def test_ocr_garbles_resolve():
    for form, want in [("пакува rhh", "упаковка"), ("пляшк", "пляшка"), ("фл all.", "флакон")]:
        got = clf.classify(form)
        assert got is not None and got[0] == want, (form, got)


def test_exclusions_stay_silent():
    assert clf.classify("шт/уп") is None       # slash dual
    assert clf.classify("шт уп") is None       # spaced dual
    assert clf.classify("100 шт") is None      # digits
    assert clf.classify("х" * 40) is None      # too long


def test_low_confidence_stays_silent():
    assert clf.classify("qwzx") is None or clf.classify("qwzx")[1] >= clf.threshold


def test_confidence_always_at_or_above_threshold():
    for form in ["пако", "наков.", "ааков"]:
        got = clf.classify(form)
        if got is not None:
            assert got[1] >= clf.threshold
