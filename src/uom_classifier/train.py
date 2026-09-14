"""Offline training for the UoM classifier (numpy only).

v0.4 (artifact v3) pipeline:

dataset (vocabulary + reviewed forms + labeled production ROWS with the
product name of the row) → OCR-style augmentation of the unit string →
features (unit-string char n-grams + product-name context, see
:func:`uom_classifier.classifier.feature_vector`) → weighted softmax
regression with an explicit ``__keep__`` class (sparse, Adam) →
form-grouped K-fold CV over several seeds, each fold simulating the whole
resolution path (exact table built from the training folds only →
exclusions → model) → confidence threshold tuned on the out-of-fold
predictions (validation, never the gold set) → optional evaluation on a
frozen gold file held out by form key → JSON artifact read by
:mod:`uom_classifier.classifier` with zero dependencies.

Metric (the same for CV and gold): a row is **correct** when the answer is
its canonical label, or silence when the label is ``KEEP``; **wrong** when a
canon is answered that is not the label (any canon on a ``KEEP`` row); a
silence on a canonical label is a **miss**. ``accuracy = correct / rows``
and ``wrong_rate = wrong / rows``, weighted by production row counts.

Run::

    python -m uom_classifier.train --dataset data/uom_training_pairs.json \
        --gold data/gold.jsonl --out src/uom_classifier/data/uom_classifier.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .classifier import (
    KEEP_CLASS,
    UomClassifier,
    clean,
    exact_key,
    feature_vector,
    is_spaced_dual,
    specific_unit_pair,
)

logger = logging.getLogger("uom_classifier.train")

ARTIFACT_VERSION = 3

AUG_FORM_ONLY = 8  # augmented copies per form-only example (vocabulary, pairs)
AUG_ROW = 2  # augmented copies per labeled production row
CONTEXT_DROPOUT = 0.35  # share of row copies trained WITHOUT the product name
MAX_WRONG_RATE = 0.015  # threshold tuning constraint (validation)
SEED = 24948
SEEDS = (24948, 1, 2)
CONTEXT_SCALE = 0.5

# Cyrillic ↔ Latin homoglyphs — the dominant OCR corruption class.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "к": "k", "м": "m", "т": "t", "н": "h", "в": "b", "г": "r",
    "ш": "w", "п": "n", "л": "l",
}  # fmt: skip
_HOMOGLYPHS_REV = {v: k for k, v in _HOMOGLYPHS.items()}

_NUMERIC_TOKEN_RE = re.compile(r"^\d")
_MULTIPLIERS = frozenset({"тис", "тис.", "тисяч", "тисяча", "млн", "млн."})


def carries_quantity(key: str) -> bool:
    """A form that encodes an amount, not just a unit («100 шт», «фл. 40мл»,
    «тис. доз»). Resolving it to the bare unit would silently rescale the
    quantity — «5 тис. доз» must not become «5 доз». Digits INSIDE a letter
    token («д03» = OCR «доз») are a glyph confusion, not an amount."""
    tokens = key.split()
    return any(_NUMERIC_TOKEN_RE.match(t) or t in _MULTIPLIERS for t in tokens)


def form_key(raw: str | None) -> str:
    """Grouping key of a form for hold-out and CV folds: HTML stripped,
    casefolded, de-spaced, edge punctuation removed — «шт/ уп» ≡ «шт/уп»,
    «lit.» ≡ «<b>lit</b>». No form key is ever on both sides of a split."""
    text = "".join(clean(raw).casefold().split())
    return text.strip(".,;:!_-'\"") or text


def augment(form: str, rng: random.Random) -> str:
    """One OCR-style corruption: homoglyph swaps, dots, spaces, case."""
    chars = list(form)
    for _ in range(max(1, len(chars) // 4)):
        i = rng.randrange(len(chars))
        ch = chars[i].lower()
        if ch in _HOMOGLYPHS and rng.random() < 0.7:
            chars[i] = _HOMOGLYPHS[ch]
        elif ch in _HOMOGLYPHS_REV and rng.random() < 0.7:
            chars[i] = _HOMOGLYPHS_REV[ch]
    out = "".join(chars)
    roll = rng.random()
    if roll < 0.25:
        out += "."
    elif roll < 0.4 and len(out) > 2:
        i = rng.randrange(1, len(out))
        out = out[:i] + " " + out[i:]
    if rng.random() < 0.3:
        out = out.upper()
    return out


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One labeled example. ``weight`` = production rows it stands for
    (0 for form-only examples — they train but never count in metrics)."""

    form: str
    name: str
    label: str  # canon or "KEEP"
    weight: float
    source: str

    @property
    def key(self) -> str:
        return form_key(self.form)


@dataclass
class Dataset:
    classes: list[str]  # model classes: canons + __keep__
    vocabulary: dict[str, str]
    exact: dict[str, str]  # reviewed form → canon (raw keys)
    rows: list[Row]  # labeled production rows with context (metric rows)
    form_only: list[Row]  # pairs without context (train only)
    context_only: list[Row]  # dictionary-labeled context rows (train only)


def model_label(label: str, classes: list[str]) -> str:
    """Labels outside the model's classes (``KEEP``, dictionary-only canons
    «блістер»/«банка»/«каністра») train the ``__keep__`` class: the model has
    no class for them, so silence is the only answer it can give."""
    return label if label in classes else KEEP_CLASS


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = sorted(raw["canons"]) + [KEEP_CLASS]
    vocabulary = {k.casefold(): v for k, v in raw["vocabulary"].items()}
    rows = [
        Row(r["form"], r.get("name") or "", r["label"], float(r.get("weight", 1.0)), r["source"])
        for r in raw.get("rows", [])
    ]
    form_only = [Row(f, "", lbl, 0.0, "pairs") for f, lbl in raw["pairs"].items()]
    context_only = [
        Row(r["form"], r.get("name") or "", r["label"], 0.0, "dictionary")
        for r in raw.get("context_rows", [])
    ]
    return Dataset(classes, vocabulary, dict(raw.get("exact", {})), rows, form_only, context_only)


_PART_RE = re.compile(r"[/()]")


def is_structural_dual(key: str, vocabulary: dict[str, str]) -> bool:
    """A spaced dual, or a slash/paren form whose halves resolve (through the
    vocabulary) to two or more DIFFERENT canons. Unresolvable halves — an
    English translation («units», «packing») or a truncation — do not make a
    dual on their own; the reviewed label decides, and gold overrides it."""
    if is_spaced_dual(key, vocabulary):
        return True
    if "/" not in key and "(" not in key:
        return False
    canons = set()
    for part in _PART_RE.split(key):
        p = part.strip(" .,;:")
        canon = vocabulary.get(p) or vocabulary.get(p + ".") if p else None
        if canon:
            canons.add(canon)
    return len(canons) >= 2


def build_exact(
    raw_exact: dict[str, str],
    vocabulary: dict[str, str],
    classes: list[str],
    *,
    exclude_keys: set[str] = frozenset(),  # type: ignore[assignment]
) -> tuple[dict[str, str], int]:
    """Exact lookup table from reviewed forms: (table, conflicts_dropped).

    Keys go through :func:`exact_key` — the same normalization the
    classifier applies at lookup. A key that two raw forms map to DIFFERENT
    labels is ambiguous and dropped; keys already in the vocabulary are left
    to the vocabulary (it is consulted first); forms that carry an amount
    (:func:`carries_quantity`) are never resolved to a bare unit; forms whose
    grouping key is held out (``exclude_keys``) are left out.
    """
    labels: dict[str, set[str]] = {}
    for form, label in raw_exact.items():
        if label not in classes or label == KEEP_CLASS:
            continue
        key = exact_key(form)
        if not key or key in vocabulary or carries_quantity(key):
            continue
        # Structural duals never enter the table: «пач (штука)», «ампула
        # (пачка)», «шт. паков» name two different units (price basis, ADR-065).
        # Bilingual or same-canon halves («штук / units», «уп/упаковка») stay.
        if is_structural_dual(key, vocabulary):
            continue
        if form_key(form) in exclude_keys:
            continue
        labels.setdefault(key, set()).add(label)
    table = {k: next(iter(v)) for k, v in sorted(labels.items()) if len(v) == 1}
    return table, sum(1 for v in labels.values() if len(v) > 1)


# --------------------------------------------------------------------------
# features + sparse softmax
# --------------------------------------------------------------------------


class Sparse:
    """CSR-like container: row indices, feature indices, values."""

    def __init__(self, n_rows: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray):
        self.n_rows, self.rows, self.cols, self.vals = n_rows, rows, cols, vals
        # ``rows`` is non-decreasing (built example by example) → segment sums.
        self._starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]]) if len(rows) else rows
        self._seg_rows = rows[self._starts] if len(rows) else rows

    def matmul(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_rows, w.shape[1]), dtype=np.float64)
        if len(self.rows):
            out[self._seg_rows] = np.add.reduceat(
                self.vals[:, None] * w[self.cols], self._starts, axis=0
            )
        return out

    def rmatmul(self, g: np.ndarray, n_features: int) -> np.ndarray:
        contrib = self.vals[:, None] * g[self.rows]
        out = np.empty((n_features, g.shape[1]), dtype=np.float64)
        for c in range(g.shape[1]):
            out[:, c] = np.bincount(self.cols, weights=contrib[:, c], minlength=n_features)
        return out


def vectorize(
    examples: list[tuple[str, str]],
    feature_index: dict[str, int],
    *,
    context_scale: float,
    grow: bool,
) -> Sparse:
    rows, cols, vals = [], [], []
    for i, (form, name) in enumerate(examples):
        for f, v in feature_vector(form, name, context_scale).items():
            idx = feature_index.get(f)
            if idx is None:
                if not grow:
                    continue
                idx = feature_index[f] = len(feature_index)
            rows.append(i)
            cols.append(idx)
            vals.append(v)
    return Sparse(
        len(examples),
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.float64),
    )


def train_softmax(
    x: Sparse,
    y: np.ndarray,
    sample_w: np.ndarray,
    n_features: int,
    n_classes: int,
    *,
    epochs: int = 400,
    seed: int = SEED,
    l2: float = 1e-5,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted multinomial logistic regression, full-batch Adam."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=(n_features, n_classes))
    b = np.zeros(n_classes)
    sw = sample_w / sample_w.sum()
    onehot = np.zeros((x.n_rows, n_classes))
    onehot[np.arange(x.n_rows), y] = 1.0
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        logits = x.matmul(w) + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        grad = (p - onehot) * sw[:, None]
        gw = x.rmatmul(grad, n_features) + l2 * w
        gb = grad.sum(axis=0)
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw * gw
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb * gb
        c1, c2 = 1 - beta1**epoch, 1 - beta2**epoch
        w -= lr * (mw / c1) / (np.sqrt(vw / c2) + eps)
        b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        if epoch % 100 == 0:
            loss = -(sw * np.log(p[np.arange(x.n_rows), y] + 1e-12)).sum()
            logger.debug("epoch %d loss %.4f", epoch, float(loss))
    return w, b


# --------------------------------------------------------------------------
# training set assembly
# --------------------------------------------------------------------------


def training_examples(
    ds: Dataset,
    *,
    exclude_keys: set[str],
    rng: random.Random,
    context_scale: float,
    use_context_rows: bool = True,
) -> tuple[list[tuple[str, str]], list[str], list[float]]:
    """(examples, labels, weights) for one training run.

    Every source drops forms whose grouping key is in ``exclude_keys`` (the
    held-out fold, or the gold set). Row copies get augmented unit strings;
    ``CONTEXT_DROPOUT`` of them lose the product name so the model also
    works when the name is missing. Weights: a production row weighs its row
    count (capped at 20 — a dozen rows of «ki» must not drown the tail),
    form-only and dictionary examples weigh 1.
    """
    classes = ds.classes
    ex: list[tuple[str, str]] = []
    lab: list[str] = []
    wts: list[float] = []

    def add(form: str, name: str, label: str, weight: float) -> None:
        ex.append((form, name))
        lab.append(model_label(label, classes))
        wts.append(weight)

    row_keys = {r.key for r in ds.rows}
    base = [(f, lbl) for f, lbl in ds.vocabulary.items()] + [
        (c, c) for c in classes if c != KEEP_CLASS
    ]
    for form, label in base:
        add(form, "", label, 1.0)
        for _ in range(AUG_FORM_ONLY):
            add(augment(form, rng), "", label, 1.0)
    for r in ds.form_only:
        if r.key in exclude_keys or r.key in row_keys:
            continue  # production rows of the same form carry it with context
        add(clean(r.form), "", r.label, 1.0)
        for _ in range(AUG_FORM_ONLY):
            add(augment(clean(r.form), rng), "", r.label, 1.0)
    for r in ds.rows:
        if r.key in exclude_keys:
            continue
        w = min(r.weight, 20.0)
        form = clean(r.form)
        for copy in range(1 + AUG_ROW):
            f = form if copy == 0 else augment(form, rng)
            name = "" if (context_scale <= 0 or rng.random() < CONTEXT_DROPOUT) else r.name
            add(f, name, r.label, w)
    if use_context_rows and context_scale > 0:
        for r in ds.context_only:
            if r.key in exclude_keys:
                continue
            add(clean(r.form), r.name, r.label, 0.5)
    return ex, lab, wts


def fit(
    ds: Dataset,
    *,
    exclude_keys: set[str],
    seed: int,
    context_scale: float,
    epochs: int,
    use_context_rows: bool = True,
) -> dict[str, Any]:
    """Train one model; returns an (unthresholded) artifact dict."""
    rng = random.Random(seed)
    ex, lab, wts = training_examples(
        ds,
        exclude_keys=exclude_keys,
        rng=rng,
        context_scale=context_scale,
        use_context_rows=use_context_rows,
    )
    class_index = {c: i for i, c in enumerate(ds.classes)}
    fi: dict[str, int] = {}
    x = vectorize(ex, fi, context_scale=context_scale, grow=True)
    y = np.asarray([class_index[lbl] for lbl in lab])
    w, b = train_softmax(
        x, y, np.asarray(wts), len(fi), len(ds.classes), epochs=epochs, seed=seed
    )
    sparse: dict[str, list[float]] = {}
    for f, idx in fi.items():
        row = w[idx]
        if float(np.abs(row).max()) >= 0.02:
            sparse[f] = [round(float(v), 3) for v in row]
    exact, conflicts = build_exact(ds.exact, ds.vocabulary, ds.classes, exclude_keys=exclude_keys)
    return {
        "version": ARTIFACT_VERSION,
        "classes": ds.classes,
        "threshold": 0.5,
        "context_scale": context_scale,
        "bias": [round(float(v), 4) for v in b],
        "weights": sparse,
        "vocabulary": ds.vocabulary,
        "exact": exact,
        "_exact_conflicts": conflicts,
        "_features_total": len(fi),
    }


# --------------------------------------------------------------------------
# evaluation (through the real inference code)
# --------------------------------------------------------------------------


@dataclass
class Scored:
    row: Row
    path: str  # vocabulary | exact | excluded | keep | model
    label: str | None  # predicted canon (None = silence before thresholding)
    prob: float


def score_rows(artifact: dict[str, Any], rows: Iterable[Row]) -> list[Scored]:
    """Run the resolution path of :class:`UomClassifier` WITHOUT the
    threshold, recording which path answered — thresholds are applied later
    so one pass serves every candidate threshold."""
    clf = UomClassifier(artifact=artifact)
    out: list[Scored] = []
    for r in rows:
        key = exact_key(r.form)
        if key in clf.vocabulary:
            out.append(Scored(r, "vocabulary", clf.vocabulary[key], 1.0))
            continue
        if key in clf.exact:
            out.append(Scored(r, "exact", clf.exact[key], 1.0))
            continue
        pair = specific_unit_pair(r.form, clf.vocabulary)
        if pair is not None:
            out.append(Scored(r, "pair_rule", pair, 1.0))
            continue
        if not clf.is_classifiable(r.form):
            out.append(Scored(r, "excluded", None, 0.0))
            continue
        probs = clf.scores(r.form, r.name)
        best = max(range(len(clf.classes)), key=probs.__getitem__)
        if clf.classes[best] == KEEP_CLASS:
            out.append(Scored(r, "keep", None, probs[best]))
        else:
            out.append(Scored(r, "model", clf.classes[best], probs[best]))
    return out


def outcome(label: str, answer: str | None) -> str:
    """correct | wrong | miss — the metric of the module docstring.

    Silence is correct only on ``KEEP``; on any canon — dictionary-only ones
    («блістер») included — it is a miss, never counted as correct."""
    if answer is None:
        return "correct" if label == "KEEP" else "miss"
    return "correct" if answer == label else "wrong"


def answer_at(s: Scored, threshold: float) -> str | None:
    if s.path in ("vocabulary", "exact", "pair_rule"):
        return s.label
    if s.path == "model" and s.prob >= threshold:
        return s.label
    return None


def metrics(scored: list[Scored], threshold: float) -> dict[str, Any]:
    tot = cor = wrg = mis = 0.0
    by_path: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in scored:
        ans = answer_at(s, threshold)
        o = outcome(s.row.label, ans)
        w = s.row.weight
        tot += w
        cor += w * (o == "correct")
        wrg += w * (o == "wrong")
        mis += w * (o == "miss")
        path = s.path if (s.path != "model" or ans is not None) else "below_threshold"
        by_path[path][o] += w
    return {
        "rows": round(tot, 2),
        "accuracy": round(cor / tot, 4) if tot else 0.0,
        "wrong_rate": round(wrg / tot, 4) if tot else 0.0,
        "miss_rate": round(mis / tot, 4) if tot else 0.0,
        "by_path": {p: {k: round(v, 2) for k, v in d.items()} for p, d in sorted(by_path.items())},
    }


def pick_threshold(scored: list[Scored], *, max_wrong: float = MAX_WRONG_RATE) -> float:
    """Highest-accuracy threshold whose wrong-rate stays ≤ ``max_wrong``."""
    best, best_acc = 0.99, -1.0
    for cand in np.arange(0.30, 0.995, 0.005):
        m = metrics(scored, float(cand))
        if m["wrong_rate"] <= max_wrong and m["accuracy"] > best_acc + 1e-9:
            best, best_acc = float(cand), m["accuracy"]
    return round(best, 3)


def cross_validate(
    ds: Dataset,
    *,
    exclude_keys: set[str],
    seeds: Iterable[int],
    k_folds: int,
    context_scale: float,
    epochs: int,
    use_context_rows: bool = True,
) -> dict[int, list[Scored]]:
    """Form-grouped K-fold per seed → out-of-fold scored rows per seed."""
    metric_rows = [r for r in ds.rows if r.key not in exclude_keys and r.weight > 0]
    keys = sorted({r.key for r in metric_rows})
    result: dict[int, list[Scored]] = {}
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = keys[:]
        rng.shuffle(shuffled)
        fold_of = {k: i % k_folds for i, k in enumerate(shuffled)}
        oof: list[Scored] = []
        for fold in range(k_folds):
            held = {k for k, f in fold_of.items() if f == fold}
            art = fit(
                ds,
                exclude_keys=exclude_keys | held,
                seed=seed + fold,
                context_scale=context_scale,
                epochs=epochs,
                use_context_rows=use_context_rows,
            )
            oof.extend(score_rows(art, [r for r in metric_rows if r.key in held]))
        result[seed] = oof
        logger.info("cv seed %d: %d oof rows", seed, len(oof))
    return result


def load_gold(path: Path) -> tuple[list[Row], list[dict[str, Any]]]:
    """(headline rows, contested records). Contested rows (no majority of the
    three judges) are reported separately and excluded from the metric."""
    rows, contested = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        if g["label"] is None:
            contested.append(g)
            continue
        rows.append(Row(g["raw"], g.get("name") or "", g["label"], float(g["row_weight"]), "gold"))
    return rows, contested


def form_weighted(scored: list[Scored], gold_meta: dict[str, float], threshold: float) -> dict[str, float]:
    """Each form weighs its sampling weight; per-form score = share of its
    labeled rows that are correct / wrong."""
    per_form: dict[str, list[str]] = defaultdict(list)
    for s in scored:
        per_form[s.row.key].append(outcome(s.row.label, answer_at(s, threshold)))
    tot = cor = wrg = 0.0
    for key, outs in per_form.items():
        w = gold_meta.get(key, 1.0)
        tot += w
        cor += w * outs.count("correct") / len(outs)
        wrg += w * outs.count("wrong") / len(outs)
    return {"forms": len(per_form), "accuracy": round(cor / tot, 4), "wrong_rate": round(wrg / tot, 4)}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def train(
    dataset_path: Path,
    out_path: Path,
    *,
    gold_path: Path | None,
    k_folds: int = 5,
    seeds: tuple[int, ...] = SEEDS,
    context_scale: float = CONTEXT_SCALE,
    epochs: int = 400,
    ship_with_gold: bool = True,
) -> dict[str, Any]:
    ds = load_dataset(dataset_path)
    gold_rows: list[Row] = []
    contested: list[dict[str, Any]] = []
    gold_keys: set[str] = set()
    if gold_path is not None:
        gold_rows, contested = load_gold(gold_path)
        gold_keys = {r.key for r in gold_rows} | {form_key(c["raw"]) for c in contested}

    # 1. CV on the training data only (gold held out) → threshold.
    oof = cross_validate(
        ds,
        exclude_keys=gold_keys,
        seeds=seeds,
        k_folds=k_folds,
        context_scale=context_scale,
        epochs=epochs,
    )
    first = seeds[0]
    threshold = pick_threshold(oof[first])
    cv = {str(s): metrics(v, threshold) for s, v in oof.items()}
    logger.info("threshold %.3f (tuned on OOF of seed %d)", threshold, first)
    for s, m in cv.items():
        logger.info("cv seed %s: acc=%.4f wrong=%.4f", s, m["accuracy"], m["wrong_rate"])

    report: dict[str, Any] = {"threshold": threshold, "cv": cv, "cv_seeds": list(seeds)}

    # 2. Held-out model (gold keys excluded everywhere) → gold metric.
    if gold_rows:
        held = fit(ds, exclude_keys=gold_keys, seed=first, context_scale=context_scale, epochs=epochs)
        held["threshold"] = threshold
        # Saved next to the shipped artifact: the downstream service evaluates
        # THIS one on gold through its own resolution path (the honest number).
        held_path = out_path.with_name(out_path.stem + "_heldout.json")
        held_path.parent.mkdir(parents=True, exist_ok=True)
        held_path.write_text(
            json.dumps({k: v for k, v in held.items() if not k.startswith("_")}, ensure_ascii=False),
            encoding="utf-8",
        )
        scored = score_rows(held, gold_rows)
        form_w: dict[str, float] = {}
        for line in gold_path.read_text(encoding="utf-8").splitlines():  # type: ignore[union-attr]
            if line.strip():
                g = json.loads(line)
                form_w[g["key"]] = float(g["form_weight"])
        report["gold_heldout"] = {
            **metrics(scored, threshold),
            "form_weighted": form_weighted(scored, form_w, threshold),
            "contested_rows": len(contested),
        }
        logger.info("gold (held out): %s", json.dumps(report["gold_heldout"], ensure_ascii=False))

    # 3. Shipped model: every labeled row, gold included (its labels are the
    #    best we have) — the gold number above is the honest one.
    final_ds = ds
    if gold_rows and ship_with_gold:
        final_ds = Dataset(
            ds.classes, ds.vocabulary, dict(ds.exact), ds.rows + gold_rows, ds.form_only, ds.context_only
        )
        # Gold overrides the reviewed table: a key whose gold rows say KEEP or
        # another canon leaves the table; unanimous canonical gold forms join.
        gold_labels: dict[str, set[str]] = defaultdict(set)
        for r in gold_rows:
            gold_labels[r.key].add(r.label)
        # A key with a contested gold row (no judge majority) is left exactly
        # as the reviewed table had it — neither removed nor added: an exact
        # answer must be certain («yıı» → мл on one majority row vs упаковка).
        contested_keys = {form_key(c["raw"]) for c in contested}
        for form, label in list(final_ds.exact.items()):
            if form_key(form) in contested_keys:
                continue
            labels = gold_labels.get(form_key(form))
            if labels and labels != {label}:
                final_ds.exact.pop(form)
        final_ds.exact.update(
            {
                r.form: r.label
                for r in gold_exact_candidates(gold_rows, ds.classes)
                if r.key not in contested_keys
            }
        )
    artifact = fit(final_ds, exclude_keys=set(), seed=first, context_scale=context_scale, epochs=epochs)
    artifact["threshold"] = threshold
    artifact["trained_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    artifact["metrics"] = {
        **report,
        "exact_forms": len(artifact["exact"]),
        "exact_conflicts_dropped": artifact.pop("_exact_conflicts"),
        "features_kept": len(artifact["weights"]),
        "features_total": artifact.pop("_features_total"),
        "training_rows": len(final_ds.rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    logger.info("artifact → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return artifact["metrics"]


def gold_exact_candidates(rows: list[Row], classes: list[str]) -> list[Row]:
    """Gold forms whose labeled rows ALL agree on one model canon — safe for
    the exact table of the shipped artifact (context-dependent forms stay
    with the model)."""
    by_key: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_key[r.key].append(r)
    out = []
    for group in by_key.values():
        labels = {r.label for r in group}
        if len(labels) == 1 and next(iter(labels)) in classes and next(iter(labels)) != KEEP_CLASS:
            out.extend(group)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/uom_training_pairs.json")
    ap.add_argument("--gold", default="data/gold.jsonl")
    ap.add_argument("--out", default="src/uom_classifier/data/uom_classifier.json")
    ap.add_argument("--context-scale", type=float, default=CONTEXT_SCALE)
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    gold = Path(args.gold) if args.gold and Path(args.gold).exists() else None
    train(
        Path(args.dataset),
        Path(args.out),
        gold_path=gold,
        context_scale=args.context_scale,
        epochs=args.epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
