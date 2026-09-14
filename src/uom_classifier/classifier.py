"""Dependency-free inference for the UoM classifier.

The model is a tiny softmax regression over character n-grams (1..4),
trained offline (see :mod:`uom_classifier.train`). Inference is pure
Python — a sparse per-n-gram weight lookup plus a softmax — and takes
microseconds per call, so it can run inside any pipeline without numpy,
torch or network access.

Usage::

    from uom_classifier import UomClassifier

    clf = UomClassifier()                    # bundled artifact
    clf = UomClassifier("path/to/model.json")  # custom artifact

    clf.classify("пакува rhh")   # -> ("упаковка", 0.95)
    clf.classify("lilt.")        # -> ("штука", 1.0)   exact labeled form
    clf.classify("<b>шт</b>")    # -> ("штука", 1.0)   HTML stripped first
    clf.classify("шт/уп")        # -> None (dual descriptor, excluded)
    clf.classify("порошок")      # -> None (dosage form, not a unit)
    clf.classify("garbage")      # -> None (below confidence threshold)

``None`` always means "leave the value as it is" — the classifier stays
silent rather than guessing.

Order of resolution:

1. **Exact lookup** — the vocabulary of common spellings plus the labeled
   forms of the production campaign (``exact``). A form a human-reviewed
   campaign already resolved is answered deterministically: a softmax over
   22 classes is often *less* than threshold-confident on its own training
   examples when they are short («lilt.», «пл.»).
2. **Exclusions** (:func:`is_classifiable`) — forms the model must not have
   an opinion about.
3. **Model** — only above its confidence threshold.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_BUNDLED_ARTIFACT = Path(__file__).parent / "data" / "uom_classifier.json"

NGRAM_MIN = 1
NGRAM_MAX = 4
MAX_LEN = 32
# «т» is a tonne, not «штука» — single-letter forms are out of scope.
MIN_LEN = 2

# OCR tables sometimes carry markup from the PDF text layer: «<b>lit</b>».
# The slash in a closing tag would otherwise look like a dual descriptor.
_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_SPACE_RE = re.compile(r"\s+")

# Systematic model misses found by an LLM-judge eval on live data (73
# predictions): these are other entities, not corrupted units.
SERVICE_MARKERS = frozenset({"дослідження"})
PLACEHOLDERS = frozenset({"nan", "null", "none", "n/a", "-", "—"})
DOSAGE_FORMS = frozenset(
    {
        "порошок",
        "розчин",
        "мазь",
        "гель",
        "крем",
        "спрей",
        "сироп",
        "суспензія",
        "емульсія",
        "краплі",
        "аерозоль",
        "ліофілізат",
    }
)
# Real units that are simply outside the model's classes. Blister, jar and
# canister are canonical in the downstream dictionary (v0.3.0), but the model
# has no class for them — any opinion it had would be guaranteed wrong.
FOREIGN_UNITS = frozenset(
    {
        "набір", "комплект", "тонна",
        "блістер", "блістери", "банка", "банки",
        "каністра", "каністр", "каністри",
    }
)  # fmt: skip


def clean(raw: str | None) -> str:
    """Strip HTML tags and collapse whitespace."""
    return _SPACE_RE.sub(" ", _TAG_RE.sub("", raw or "")).strip()


def exact_key(raw: str | None) -> str:
    """Key of the exact lookup tables: cleaned and casefolded."""
    return clean(raw).casefold()


def extract_ngrams(raw: str) -> list[str]:
    """Character n-grams (1..4) over ``^form$`` — casefolded, de-spaced."""
    text = "^" + "".join((raw or "").casefold().split()) + "$"
    grams: list[str] = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return grams


def is_spaced_dual(text: str, vocabulary: dict[str, str]) -> bool:
    """Two+ space-separated tokens resolving to DIFFERENT canons («шт уп»)."""
    tokens = text.casefold().replace(".", "").split()
    if len(tokens) < 2:
        return False
    canons = {vocabulary[t] for t in tokens if t in vocabulary}
    return len(canons) >= 2


def is_classifiable(raw: str | None, vocabulary: dict[str, str]) -> bool:
    """Whether the model is allowed to have an opinion about ``raw``.

    Deliberate exclusions (the model must stay silent):

    * slash duals («шт/уп») and spaced duals («шт уп») — they encode TWO
      units at once (package + unit) and must not be collapsed;
    * strings containing digits («100 шт») — quantity descriptors;
    * overly long or single-letter strings;
    * placeholders («nan»), service markers («дослідження»), dosage forms
      («порошок») and real units outside the canon («набір»).
    """
    text = clean(raw)
    if not text or len(text) > MAX_LEN or len(text) < MIN_LEN:
        return False
    if "/" in text:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    folded = text.casefold()
    if folded in SERVICE_MARKERS or folded in PLACEHOLDERS:
        return False
    tokens = folded.replace(".", "").split()
    if any(t in DOSAGE_FORMS or t in FOREIGN_UNITS for t in tokens):
        return False
    return not is_spaced_dual(text, vocabulary)


class UomClassifier:
    """Loads a trained artifact and classifies raw UoM strings."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        path = Path(artifact_path) if artifact_path else _BUNDLED_ARTIFACT
        self._artifact = json.loads(path.read_text(encoding="utf-8"))
        self.version: int = int(self._artifact.get("version", 1))
        self.classes: list[str] = self._artifact["classes"]
        self.threshold: float = float(self._artifact["threshold"])
        self._bias: list[float] = self._artifact["bias"]
        self._weights: dict[str, list[float]] = self._artifact["weights"]
        # Vocabulary of common spellings — exact lookup and spaced-dual check.
        self.vocabulary: dict[str, str] = self._artifact.get("vocabulary", {})
        # Labeled campaign forms (v2+) — exact lookup only.
        self.exact: dict[str, str] = self._artifact.get("exact", {})

    # -- guards ----------------------------------------------------------

    def is_classifiable(self, raw: str) -> bool:
        return is_classifiable(raw, self.vocabulary)

    # -- inference -------------------------------------------------------

    def lookup(self, raw: str) -> str | None:
        """Exact lookup (cleaned, casefolded). Always beats the model."""
        key = exact_key(raw)
        return self.vocabulary.get(key) or self.exact.get(key)

    def scores(self, raw: str) -> list[float]:
        """Softmax probabilities over ``self.classes`` (no threshold)."""
        counts: dict[str, int] = {}
        for g in extract_ngrams(clean(raw)):
            counts[g] = counts.get(g, 0) + 1
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        logits = list(self._bias)
        for g, cnt in counts.items():
            row = self._weights.get(g)
            if row is None:
                continue
            scale = cnt / norm
            for i, w in enumerate(row):
                logits[i] += w * scale
        m = max(logits)
        exps = [math.exp(s - m) for s in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def classify(self, raw: str) -> tuple[str, float] | None:
        """(canonical_unit, confidence) or None.

        Exact lookup first; the model only speaks on lookup misses, on
        classifiable forms, and above its confidence threshold.
        """
        exact = self.lookup(raw)
        if exact is not None:
            return exact, 1.0
        if not self.is_classifiable(raw):
            return None
        probs = self.scores(raw)
        best = max(range(len(self.classes)), key=probs.__getitem__)
        if probs[best] < self.threshold:
            return None
        return self.classes[best], probs[best]
