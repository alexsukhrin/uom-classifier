"""Regression tests against the bundled artifact."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uom_classifier import UomClassifier  # noqa: E402
from uom_classifier.classifier import clean, exact_key  # noqa: E402

clf = UomClassifier()
DATASET = json.loads(
    (Path(__file__).parent.parent / "data" / "uom_training_pairs.json").read_text(
        encoding="utf-8"
    )
)


def test_artifact_is_v2_with_exact_table():
    assert clf.version == 3  # v0.4: v2 format + product-name context + __keep__
    assert clf.exact, "v2 artifact must ship the exact table"


def test_exact_vocabulary_hit():
    assert clf.classify("шт.") == ("штука", 1.0)
    assert clf.classify("ФЛ") == ("флакон", 1.0)


def test_ocr_garbles_resolve():
    for form, want in [("пакува rhh", "упаковка"), ("пляшк", "пляшка"), ("фл all.", "флакон")]:
        got = clf.classify(form)
        assert got is not None and got[0] == want, (form, got)


@pytest.mark.parametrize(
    ("form", "want"),
    [
        ("lilt.", "штука"),  # short OCR garble the softmax alone is unsure of
        ("<b>lut</b>", "штука"),  # markup + garble
        ("д03", "доза"),  # digits would exclude it from the model
        ("пл.", "пляшка"),
        ("паком", "упаковка"),
    ],
)
def test_labeled_campaign_forms_are_exact(form, want):
    assert clf.classify(form) == (want, 1.0)


def test_html_is_stripped():
    assert clean("<b>шт</b>") == "шт"
    assert clean("  <i>фл</i>   all. ") == "фл all."
    assert clf.classify("<b>шт</b>") == ("штука", 1.0)


def test_exact_table_holds_only_canonical_labels():
    assert set(clf.exact.values()) <= set(clf.classes)


def test_exact_table_never_resolves_duals_or_new_units():
    assert clf.lookup("шт/уп") is None
    assert clf.lookup("шт/паков") is None
    assert clf.lookup("блістер") is None  # legit unit outside the canon


@pytest.mark.parametrize(
    "form", ["блістер", "Блістер.", "банка", "каністра", "каністр", "каністри"]
)
def test_dictionary_only_units_keep_the_model_silent(form):
    # v0.3.0: canonical downstream, but the model has no class for them.
    assert clf.is_classifiable(form) is False
    assert clf.classify(form) is None
    assert set(clf.classes) & {"блістер", "банка", "каністра"} == set()


@pytest.mark.parametrize("form", ["тис. доз", "фл. 40мл", "100 шт"])
def test_exact_table_never_drops_an_amount(form):
    # «5 тис. доз» must not silently become «5 доз».
    assert clf.lookup(form) is None
    assert clf.classify(form) is None


def test_carries_quantity_rule():
    from uom_classifier.train import carries_quantity

    assert carries_quantity("тис. доз")
    assert carries_quantity("фл. 40мл")
    assert carries_quantity("100 шт")
    assert not carries_quantity("д03")  # OCR glyphs inside a letter token
    assert not carries_quantity("lilt.")


def test_exact_keys_match_lookup_normalization():
    for key in clf.exact:
        assert key == exact_key(key)


def test_vocabulary_wins_over_exact():
    overlap = set(clf.vocabulary) & set(clf.exact)
    assert not overlap


def test_dataset_exact_labels_are_canonical():
    assert set(DATASET["exact"].values()) <= set(DATASET["canons"])


@pytest.mark.parametrize(
    "form",
    [
        "шт/уп",  # slash dual
        "таб/капс",
        "шт уп",  # spaced dual
        "100 шт",  # digits
        "х" * 40,  # too long
        "т",  # single letter: a tonne, not «штука»
        "nan",
        "NULL",
        "дослідження",  # service marker
        "порошок",  # dosage form, not a unit
        "розчин",
        "набір паков",  # real unit outside the canon
        "тонна",
        "",
    ],
)
def test_exclusions_stay_silent(form):
    assert clf.is_classifiable(form) is False
    assert clf.classify(form) is None


def test_low_confidence_stays_silent():
    assert clf.classify("qwzx") is None or clf.classify("qwzx")[1] >= clf.threshold


def test_confidence_always_at_or_above_threshold():
    for form in ["пако", "наков.", "ааков"]:
        got = clf.classify(form)
        if got is not None:
            assert got[1] >= clf.threshold
