"""Dependency-free inference for the UoM classifier.

The model is a tiny softmax regression over character n-grams (1..4) of
the unit string, plus — since v0.4 (artifact v3) — prefixed features of the
**product name** of the same specification row. Inference is pure Python —
a sparse per-feature weight lookup plus a softmax — and takes microseconds
per call, so it can run inside any pipeline without numpy, torch or network
access.

Usage::

    from uom_classifier import UomClassifier

    clf = UomClassifier()                    # bundled artifact
    clf = UomClassifier("path/to/model.json")  # custom artifact

    clf.classify("пакува rhh")   # -> ("упаковка", 0.95)
    clf.classify("іпт", "Ксилол, каністра 5 л")  # context disambiguates
    clf.classify("lilt.")        # -> ("штука", 1.0)   exact labeled form
    clf.classify("<b>шт</b>")    # -> ("штука", 1.0)   HTML stripped first
    clf.classify("шт/уп")        # -> None (dual descriptor, excluded)
    clf.classify("порошок")      # -> None (dosage form, not a unit)
    clf.classify("garbage")      # -> None (keep class or below threshold)

``None`` always means "leave the value as it is" — the classifier stays
silent rather than guessing.

Order of resolution:

1. **Exact lookup** — the vocabulary of common spellings plus the reviewed
   labeled forms (``exact``). Context never overrides it.
2. **Exclusions** (:func:`is_classifiable`) — forms the model must not have
   an opinion about.
3. **Model** — form n-grams + product-name features. Since v0.4 the model
   has an explicit ``__keep__`` class (unreadable garbage, units outside the
   canon): predicting it — or any class below the confidence threshold —
   returns ``None``.
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
KEEP_CLASS = "__keep__"

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

# --- product-name context (artifact v3) ----------------------------------
# Words of the product name become «w:<first 5 letters>» features (a crude
# stem: «таблетки»/«таблеток» share «w:табле»); a curated list of cues about
# the dosage form and packaging becomes «q:<cue>» features. The context
# vector is L2-normalized on its own and scaled by the artifact's
# ``context_scale`` (< 1), so the unit string keeps the dominant voice.
_WORD_RE = re.compile(r"[a-zа-яіїєґ']+")
_STEM = 5
_MIN_WORD = 3
_CUES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("tablet", r"табл|таблет|tabl"),
        ("capsule", r"капс|caps"),
        ("ampoule", r"ампул|\bамп\b|amp"),
        ("vial", r"флакон|\bфл\b|флак|vial"),
        ("bottle", r"пляш|bottle"),
        ("syringe", r"шприц|syring"),
        ("pen", r"шприц-ручк|ручк"),
        ("tube", r"\bтуб|tube"),
        ("sachet", r"саше|sachet"),
        ("bag", r"пакет|мішок|мішк"),
        ("blister", r"блістер|чарунков|конвалют"),
        ("carton", r"пачк|коробк|упаков|упак\b|\bуп\b"),
        ("count_no", r"№\s*\d|\bn\s*\d|\bх\s*\d|x\s*\d"),
        ("per_n", r"\bпо\s+\d"),
        ("vol_ml", r"\d\s*мл\b|\d\s*ml\b"),
        ("vol_l", r"\d\s*л\b|\d\s*l\b|літр"),
        ("mass_mg", r"\d\s*мг\b|\d\s*mg\b"),
        ("mass_g", r"\d\s*г\b|\d\s*g\b|грам"),
        ("mass_kg", r"\d\s*кг\b|\d\s*kg\b|кілогр"),
        ("iu", r"\bмо\b|\bод\b|\biu\b"),
        ("dose", r"\bдоз"),
        ("solution", r"розчин|р-н|суспенз|сироп|крапл|емульс|концентрат"),
        ("soft", r"мазь|гель|крем|лінімент|паста"),
        ("powder", r"порош|ліофіл|гранул"),
        ("reagent", r"реаген|реактив|набір|тест|\bкит|kit|калібрат|контрол"),
        ("chem_grade", r"\bчда\b|\bхч\b|\bч\.?д\.?а|гост|\bосч\b"),
        ("glove", r"рукавич|бахіл|пара\b|пар\b"),
        ("device", r"катетер|голк|бинт|пластир|марл|серветк|зонд|канюл|маск"),
        ("canister", r"каністр"),
        ("jar", r"банк"),
    )
)


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


def extract_context(product_name: str | None) -> list[str]:
    """Prefixed product-name features: ``w:<stem>`` words and ``q:<cue>`` cues.

    Empty for a missing name — the model then decides on the unit string
    alone (it is trained with context dropout for exactly this case).
    """
    text = clean(product_name).casefold()
    if not text:
        return []
    feats = [f"w:{w[:_STEM]}" for w in _WORD_RE.findall(text) if len(w) >= _MIN_WORD]
    feats.extend(f"q:{name}" for name, pattern in _CUES if pattern.search(text))
    return feats


# Unit-class canons other than «штука» — «шт/фл» names a vial counted in
# pieces, i.e. the specific unit. Package-class canons are NOT here: «шт/уп»,
# «штука/контейнер» encode the price basis (unit + package) and stay duals.
SPECIFIC_UNITS = frozenset(
    {"ампула", "флакон", "таблетка", "капсула", "доза", "шприц", "шприц-ручка", "саше", "пара", "пакет"}
)
_PAIR_SPLIT_RE = re.compile(r"[/()]")


_MULTIPLIERS = frozenset({"тис", "тис.", "тисяч", "тисяча", "млн", "млн."})


def carries_quantity(raw: str | None) -> bool:
    """A form that encodes an amount («100 шт», «фл. 40мл», «тис. доз»):
    a whitespace token starts with a digit or is a multiplier. Digits INSIDE
    a letter token («д03» = OCR «доз») are glyph confusion, not an amount."""
    return any(t[:1].isdigit() or t in _MULTIPLIERS for t in clean(raw).casefold().split())


def specific_unit_pair(raw: str | None, vocabulary: dict[str, str]) -> str | None:
    """«шт/фл», «штука/ амп», «шт. (фл.)» → the specific unit; «уп/упаковка»
    → упаковка (both halves the same canon); else None.

    Exactly two parts split on «/» or parentheses, no digits, each resolving
    (a vocabulary spelling or the canon itself) to a canon: the same one, or
    «штука» plus ONE specific unit-class canon."""
    text = clean(raw).casefold()
    if ("/" not in text and "(" not in text) or any(ch.isdigit() for ch in text):
        return None
    parts = [p.strip(" .,;:") for p in _PAIR_SPLIT_RE.split(text)]
    parts = [p for p in parts if p]
    if len(parts) != 2:
        return None
    canons = set()
    for p in parts:
        canon = vocabulary.get(p) or vocabulary.get(p + ".") or (
            p if p in SPECIFIC_UNITS or p == "штука" else None
        )
        if canon is None:
            return None
        canons.add(canon)
    if len(canons) == 1:
        return next(iter(canons))
    if "штука" not in canons or len(canons) != 2:
        return None
    other = next(iter(canons - {"штука"}))
    return other if other in SPECIFIC_UNITS else None


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
    # v0.4: only forms that CARRY an amount are excluded («100 шт», «тис.
    # доз»); a digit inside a letter token («д03», «уг1», «na6ip») is glyph
    # confusion, and the model (with its __keep__ class) may judge it.
    if carries_quantity(text):
        return False
    folded = text.casefold()
    if folded in SERVICE_MARKERS or folded in PLACEHOLDERS:
        return False
    tokens = folded.replace(".", "").split()
    if any(t in DOSAGE_FORMS or t in FOREIGN_UNITS for t in tokens):
        return False
    return not is_spaced_dual(text, vocabulary)


def _l2(counts: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in counts.values())) or 1.0


def feature_vector(
    raw: str, product_name: str | None, context_scale: float
) -> dict[str, float]:
    """Sparse feature vector: L2-normalized form n-grams + scaled context."""
    form: dict[str, float] = {}
    for g in extract_ngrams(clean(raw)):
        form[g] = form.get(g, 0.0) + 1.0
    norm = _l2(form)
    vec = {g: v / norm for g, v in form.items()}
    if context_scale > 0:
        ctx: dict[str, float] = {}
        for f in extract_context(product_name):
            ctx[f] = ctx.get(f, 0.0) + 1.0
        if ctx:
            cnorm = _l2(ctx)
            for f, v in ctx.items():
                vec[f] = context_scale * v / cnorm
    return vec


class UomClassifier:
    """Loads a trained artifact and classifies raw UoM strings."""

    def __init__(
        self, artifact_path: str | Path | None = None, *, artifact: dict | None = None
    ) -> None:
        if artifact is None:
            path = Path(artifact_path) if artifact_path else _BUNDLED_ARTIFACT
            artifact = json.loads(path.read_text(encoding="utf-8"))
        self._artifact = artifact
        self.version: int = int(self._artifact.get("version", 1))
        self.classes: list[str] = self._artifact["classes"]
        self.threshold: float = float(self._artifact["threshold"])
        self.context_scale: float = float(self._artifact.get("context_scale", 0.0))
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

    def scores(self, raw: str, product_name: str | None = None) -> list[float]:
        """Softmax probabilities over ``self.classes`` (no threshold)."""
        logits = list(self._bias)
        for f, value in feature_vector(raw, product_name, self.context_scale).items():
            row = self._weights.get(f)
            if row is None:
                continue
            for i, w in enumerate(row):
                logits[i] += w * value
        m = max(logits)
        exps = [math.exp(s - m) for s in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def classify(
        self, raw: str, product_name: str | None = None
    ) -> tuple[str, float] | None:
        """(canonical_unit, confidence) or None.

        Exact lookup first — the product name never overrides it; the model
        only speaks on lookup misses, on classifiable forms, when its best
        class is a unit (not ``__keep__``) and above its confidence threshold.
        """
        exact = self.lookup(raw)
        if exact is not None:
            return exact, 1.0
        pair = specific_unit_pair(raw, self.vocabulary)
        if pair is not None:
            return pair, 1.0
        if not self.is_classifiable(raw):
            return None
        probs = self.scores(raw, product_name)
        best = max(range(len(self.classes)), key=probs.__getitem__)
        if self.classes[best] == KEEP_CLASS or probs[best] < self.threshold:
            return None
        return self.classes[best], probs[best]
