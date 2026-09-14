"""v0.4 (artifact v3): product-name context, the __keep__ class, the metric.

Weight-independent: mechanics are checked on synthetic artifacts, feature
parity with the vendored copy in the downstream service on a pinned feature
list (the same list is pinned in the service's tests).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uom_classifier import UomClassifier  # noqa: E402
from uom_classifier.classifier import (  # noqa: E402
    KEEP_CLASS,
    extract_context,
    feature_vector,
)
from uom_classifier.train import carries_quantity, form_key, outcome  # noqa: E402

PARITY_NAME = "АМОКСИЛ таблетки по 500 мг №20 у блістері"
PARITY_FEATURES = [
    "w:амокс",
    "w:табле",
    "w:бліст",
    "q:tablet",
    "q:blister",
    "q:count_no",
    "q:per_n",
    "q:mass_mg",
]


def _clf(**extra):
    art = {
        "version": 3,
        "classes": ["л", "штука", KEEP_CLASS],
        "threshold": 0.6,
        "context_scale": 0.5,
        "bias": [0.0, 2.0, 0.0],
        "weights": {"q:vol_l": [12.0, 0.0, 0.0], "^zz$": [0.0, 0.0, 30.0]},
        "vocabulary": {},
        "exact": {"lit": "штука"},
    }
    art.update(extra)
    return UomClassifier(artifact=art)


def test_context_features_are_pinned():
    assert extract_context(PARITY_NAME) == PARITY_FEATURES
    assert extract_context(None) == []
    assert extract_context("<b></b>") == []


def test_context_is_scaled_below_the_form():
    vec = feature_vector("іпт", "Ксилол, каністра 5 л", 0.5)
    ctx = [v for f, v in vec.items() if f.startswith(("w:", "q:"))]
    assert abs(sum(v * v for v in ctx) ** 0.5 - 0.5) < 1e-9
    assert not any(f.startswith(("w:", "q:")) for f in feature_vector("іпт", "x 5 л", 0.0))


def test_context_disambiguates_the_model_path():
    clf = _clf()
    assert clf.classify("іпт", "Ксилол, каністра 5 л")[0] == "л"
    assert clf.classify("іпт")[0] == "штука"


def test_context_never_overrides_exact():
    assert _clf().classify("lit", "Ксилол, каністра 5 л") == ("штука", 1.0)


def test_keep_class_is_silence():
    assert _clf().classify("zz", "будь-що") is None


def test_exclusions_hold_with_context():
    for form in ("шт/уп", "100 шт", "дослідження", "блістер"):
        assert _clf().classify(form, "розчин 5 л") is None


def test_form_key_groups_spellings():
    assert form_key("шт/ уп") == form_key("шт/уп")
    assert form_key("<b>lit</b>") == form_key("lit.") == "lit"


def test_metric_outcomes():
    assert outcome("KEEP", None) == "correct"
    assert outcome("штука", None) == "miss"
    assert outcome("штука", "штука") == "correct"
    assert outcome("KEEP", "штука") == "wrong"
    assert outcome("блістер", None) == "miss"  # dictionary-only canon: silence is a miss


def test_amount_guard_unchanged():
    assert carries_quantity("тис. доз") and carries_quantity("100 шт")
    assert not carries_quantity("д03")
