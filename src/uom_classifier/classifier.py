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
    clf.classify("шт/уп")        # -> None (dual descriptor, excluded)
    clf.classify("garbage")      # -> None (below confidence threshold)

``None`` always means "leave the value as it is" — the classifier stays
silent rather than guessing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_BUNDLED_ARTIFACT = Path(__file__).parent / "data" / "uom_classifier.json"

NGRAM_MIN = 1
NGRAM_MAX = 4
MAX_LEN = 32


def extract_ngrams(raw: str) -> list[str]:
    """Character n-grams (1..4) over ``^form$`` — casefolded, de-spaced."""
    text = "^" + "".join((raw or "").casefold().split()) + "$"
    grams: list[str] = []
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return grams


class UomClassifier:
    """Loads a trained artifact and classifies raw UoM strings."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        path = Path(artifact_path) if artifact_path else _BUNDLED_ARTIFACT
        self._artifact = json.loads(path.read_text(encoding="utf-8"))
        self.classes: list[str] = self._artifact["classes"]
        self.threshold: float = float(self._artifact["threshold"])
        self._bias: list[float] = self._artifact["bias"]
        self._weights: dict[str, list[float]] = self._artifact["weights"]
        # Vocabulary shipped with the artifact — used both for exact lookup
        # and for detecting multi-token "dual" descriptors ("шт уп").
        self.vocabulary: dict[str, str] = self._artifact.get("vocabulary", {})

    # -- guards ----------------------------------------------------------

    def is_classifiable(self, raw: str) -> bool:
        """Whether the model is allowed to have an opinion about ``raw``.

        Deliberate exclusions (the model must stay silent):

        * slash duals («шт/уп») and spaced duals («шт уп») — they encode
          TWO units at once (package + unit) and must not be collapsed;
        * strings containing digits («100 шт») — quantity descriptors;
        * overly long strings — not a unit of measure.
        """
        text = (raw or "").strip()
        if not text or len(text) > MAX_LEN:
            return False
        if "/" in text:
            return False
        if any(ch.isdigit() for ch in text):
            return False
        if self._is_spaced_dual(text):
            return False
        return True

    def _is_spaced_dual(self, text: str) -> bool:
        tokens = text.casefold().replace(".", "").split()
        if len(tokens) < 2:
            return False
        canons = {self.vocabulary[t] for t in tokens if t in self.vocabulary}
        return len(canons) >= 2

    # -- inference -------------------------------------------------------

    def lookup(self, raw: str) -> str | None:
        """Exact vocabulary lookup (casefolded). Always beats the model."""
        return self.vocabulary.get((raw or "").strip().casefold())

    def scores(self, raw: str) -> list[float]:
        """Softmax probabilities over ``self.classes`` (no threshold)."""
        counts: dict[str, int] = {}
        for g in extract_ngrams(raw):
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

        The exact vocabulary is consulted first; the model only speaks on
        vocabulary misses and only above its confidence threshold.
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
