# uom-classifier

Tiny, **dependency-free** classifier that maps noisy OCR'd Ukrainian
unit-of-measure strings to canonical units.

```python
from uom_classifier import UomClassifier

clf = UomClassifier()

clf.classify("пакува rhh")   # ('упаковка', 0.95)  ← OCR garble
clf.classify("фл all.")      # ('флакон', 0.92)
clf.classify("шт.")          # ('штука', 1.0)      ← exact vocabulary hit
clf.classify("шт/уп")        # None                ← dual descriptor, excluded
clf.classify("garbage")      # None                ← below confidence, stays silent
```

`None` always means *"leave the value as it is"* — the model prefers
silence over guessing.

## Why

Tables in scanned procurement PDFs come out of OCR with unit-of-measure
columns full of garbage: `ki`, `nrnr`, `пакува rhh`, `Kopo6:u:i`. An exact
vocabulary catches the common spellings, but the long tail of one-off OCR
corruptions never repeats exactly. This model covers that tail:

* **model**: softmax regression over character n-grams (1..4) — a ~1.5 MB
  JSON of sparse weights, inference in pure Python (microseconds per call,
  no numpy/torch at runtime);
* **training data**: 731 real OCR forms labeled in a production
  data-quality campaign on Ukrainian public procurement data (Prozorro) +
  a hand-curated vocabulary of 125 common spellings, augmented with
  synthetic OCR corruptions (Cyrillic↔Latin homoglyphs, dots, spaces,
  case);
* **honest metrics** (5-fold CV on campaign forms the model never saw):
  **coverage 53% @ precision 98.6%** — it confidently resolves half of the
  unseen junk and stays silent on the rest;
* **deliberate exclusions**: dual descriptors (`шт/уп`, `шт уп` — two
  units at once), strings with digits (`100 шт` is a quantity, not a
  unit), overlong strings.

Compared on the same folds against embedding kNN (BAAI/bge-m3): the
embedding approach reached only 25% coverage at comparable precision —
character-level signal beats semantics on OCR garbles.

## Install

```bash
pip install uom-classifier            # when published
pip install git+https://github.com/alexsukhrin/uom-classifier
```

## Retrain

```bash
pip install "uom-classifier[train]"   # numpy
python -m uom_classifier.train \
    --dataset data/uom_training_pairs.json \
    --out src/uom_classifier/data/uom_classifier.json
```

Or walk through the whole pipeline in
[`notebooks/train_uom_classifier.ipynb`](notebooks/train_uom_classifier.ipynb):
dataset → augmentation → features → training → CV → threshold → artifact.

To adapt to your own domain, extend `data/uom_training_pairs.json`
(`pairs`: raw form → canonical label; `vocabulary`: exact spellings;
`canons`: the label set) and retrain — the artifact format is
self-contained.

## Canonical units (22)

МО, ампула, г, доза, капсула, картридж, кг, контейнер, л, мг, мл, пакет,
пара, пляшка, саше, таблетка, туба, упаковка, флакон, шприц, шприц-ручка,
штука

## License

MIT
