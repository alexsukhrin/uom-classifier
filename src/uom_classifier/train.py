"""Offline training for the UoM classifier (numpy only).

Pipeline: dataset (labeled pairs + vocabulary) → OCR-style augmentation →
char-n-gram features → softmax regression (full-batch GD) → K-fold CV for
stable metrics and the confidence threshold → sparse JSON artifact that
:mod:`uom_classifier.classifier` reads with zero dependencies.

Run::

    python -m uom_classifier.train --dataset data/uom_training_pairs.json \
        --out src/uom_classifier/data/uom_classifier.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from .classifier import extract_ngrams

logger = logging.getLogger("uom_classifier.train")

AUG_PER_FORM = 10
TARGET_PRECISION = 0.98
SEED = 24948

# Cyrillic ↔ Latin homoglyphs — the dominant OCR corruption class.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "к": "k", "м": "m", "т": "t", "н": "h", "в": "b", "г": "r",
    "ш": "w", "п": "n", "л": "l",
}  # fmt: skip
_HOMOGLYPHS_REV = {v: k for k, v in _HOMOGLYPHS.items()}


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


def load_dataset(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, str], list[str]]:
    """(campaign_pairs, base_pairs, vocabulary, classes)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = sorted(raw["canons"])
    vocabulary = {k.casefold(): v for k, v in raw["vocabulary"].items()}
    campaign = [(f, l) for f, l in raw["pairs"].items() if l in raw["canons"]]
    base = [(f, l) for f, l in vocabulary.items()]
    base.extend((c, c) for c in classes)
    return campaign, base, vocabulary, classes


def with_augmentation(pairs: list[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    out = list(pairs)
    for form, label in pairs:
        out.extend((augment(form, rng), label) for _ in range(AUG_PER_FORM))
    rng.shuffle(out)
    return out


def vectorize(
    pairs: list[tuple[str, str]],
    feature_index: dict[str, int],
    classes: list[str],
    *,
    grow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    class_index = {c: i for i, c in enumerate(classes)}
    rows: list[dict[int, float]] = []
    labels: list[int] = []
    for form, label in pairs:
        feats: dict[int, float] = {}
        for g in extract_ngrams(form):
            idx = feature_index.get(g)
            if idx is None:
                if not grow:
                    continue
                idx = len(feature_index)
                feature_index[g] = idx
            feats[idx] = feats.get(idx, 0.0) + 1.0
        rows.append(feats)
        labels.append(class_index[label])
    x = np.zeros((len(rows), len(feature_index)), dtype=np.float32)
    for i, feats in enumerate(rows):
        norm = np.sqrt(sum(v * v for v in feats.values())) or 1.0
        for j, v in feats.items():
            x[i, j] = v / norm
    return x, np.asarray(labels, dtype=np.int64)


def train_softmax(
    x: np.ndarray, y: np.ndarray, n_classes: int, *, epochs: int = 1500
) -> tuple[np.ndarray, np.ndarray]:
    n, f = x.shape
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.01, size=(f, n_classes)).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    lr, l2 = 2.0, 5e-5
    onehot = np.zeros((n, n_classes), dtype=np.float32)
    onehot[np.arange(n), y] = 1.0
    for epoch in range(epochs):
        logits = x @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        grad = (p - onehot) / n
        w -= lr * (x.T @ grad + l2 * w)
        b -= lr * grad.sum(axis=0)
        if epoch % 300 == 299:
            loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
            logger.info("epoch %d loss %.4f", epoch + 1, float(loss))
    return w, b


def predict(
    forms: list[str], w: np.ndarray, b: np.ndarray, feature_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(forms), w.shape[0]), dtype=np.float32)
    for i, form in enumerate(forms):
        feats: dict[int, float] = {}
        for g in extract_ngrams(form):
            idx = feature_index.get(g)
            if idx is not None:
                feats[idx] = feats.get(idx, 0.0) + 1.0
        norm = np.sqrt(sum(v * v for v in feats.values())) or 1.0
        for j, v in feats.items():
            x[i, j] = v / norm
    logits = x @ w + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p.argmax(axis=1), p.max(axis=1)


def pick_threshold(
    probs: np.ndarray, correct: np.ndarray, *, target_precision: float = TARGET_PRECISION
) -> float:
    """Lowest threshold with best coverage at precision ≥ target."""
    best, best_cov = 0.99, -1.0
    for cand in np.arange(0.5, 0.995, 0.005):
        mask = probs >= cand
        if not mask.any():
            continue
        precision = float(correct[mask].mean())
        coverage = float(mask.mean())
        if precision >= target_precision and coverage > best_cov:
            best, best_cov = float(cand), coverage
    return round(best, 3)


def train(dataset_path: Path, out_path: Path, *, k_folds: int = 5) -> dict[str, Any]:
    campaign, base, vocabulary, classes = load_dataset(dataset_path)
    rng = random.Random(SEED)
    rng.shuffle(campaign)
    folds = [campaign[i::k_folds] for i in range(k_folds)]

    fold_thresholds: list[float] = []
    cv_n = cv_cov = cv_cor = 0
    for k in range(k_folds):
        test = folds[k]
        train_pairs = [p for i, f in enumerate(folds) if i != k for p in f] + base
        train_pairs = with_augmentation(train_pairs, random.Random(SEED + k))
        fi: dict[str, int] = {}
        x, y = vectorize(train_pairs, fi, classes, grow=True)
        w, b = train_softmax(x, y, len(classes), epochs=800)
        forms = [f for f, _ in test]
        labels = np.asarray([classes.index(label) for _, label in test])
        pred, probs = predict(forms, w, b, fi)
        correct = pred == labels
        thr = pick_threshold(probs, correct)
        mask = probs >= thr
        fold_thresholds.append(thr)
        cv_n += len(test)
        cv_cov += int(mask.sum())
        cv_cor += int(correct[mask].sum())
        logger.info(
            "fold %d: n=%d thr=%.3f coverage=%.3f precision=%.3f",
            k, len(test), thr, float(mask.mean()),
            float(correct[mask].mean()) if mask.any() else 0.0,
        )
    threshold = round(float(np.median(fold_thresholds)), 3)
    cv_coverage = cv_cov / cv_n if cv_n else 0.0
    cv_precision = cv_cor / cv_cov if cv_cov else 0.0
    logger.info(
        "CV: coverage=%.3f precision=%.3f threshold=%.3f",
        cv_coverage, cv_precision, threshold,
    )

    final_pairs = with_augmentation(campaign + base, random.Random(SEED))
    feature_index: dict[str, int] = {}
    x, y = vectorize(final_pairs, feature_index, classes, grow=True)
    w, b = train_softmax(x, y, len(classes), epochs=1500)

    sparse: dict[str, list[float]] = {}
    for g, idx in feature_index.items():
        row = w[idx]
        if float(np.abs(row).max()) >= 0.02:
            sparse[g] = [round(float(v), 3) for v in row]

    artifact: dict[str, Any] = {
        "version": 1,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "classes": classes,
        "threshold": threshold,
        "bias": [round(float(v), 4) for v in b],
        "weights": sparse,
        "vocabulary": vocabulary,
        "metrics": {
            "cv_folds": k_folds,
            "cv_test_forms": cv_n,
            "cv_coverage": round(cv_coverage, 4),
            "cv_precision": round(cv_precision, 4),
            "campaign_pairs": len(campaign),
            "features_kept": len(sparse),
            "features_total": len(feature_index),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    logger.info("artifact → %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return artifact["metrics"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/uom_training_pairs.json")
    ap.add_argument("--out", default="src/uom_classifier/data/uom_classifier.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    train(Path(args.dataset), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
