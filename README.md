# uom-classifier

Tiny, **dependency-free** classifier that maps noisy OCR'd Ukrainian
unit-of-measure strings to canonical units.

```python
from uom_classifier import UomClassifier

clf = UomClassifier()

clf.classify("пакува rhh")   # ('упаковка', 0.95)  ← OCR garble, model
clf.classify("фл all.")      # ('флакон', 0.92)
clf.classify("шт.")          # ('штука', 1.0)      ← exact vocabulary hit
clf.classify("lilt.")        # ('штука', 1.0)      ← exact labeled form
clf.classify("<b>шт</b>")    # ('штука', 1.0)      ← HTML stripped first
clf.classify("шт/уп")        # None                ← dual descriptor, excluded
clf.classify("тис. доз")     # None                ← carries an amount, never a bare unit
clf.classify("порошок")      # None                ← dosage form, not a unit
clf.classify("garbage")      # None                ← below confidence, stays silent
```

`None` always means *"leave the value as it is"* — the model prefers
silence over guessing.

## How a string is resolved

1. **Clean** — HTML tags from the PDF text layer are stripped
   (`<b>lit</b>` → `lit`), whitespace collapsed.
2. **Exact lookup** — the vocabulary of common spellings, then the
   labeled forms of the production campaign (`exact`, 602 forms). A form a
   reviewed campaign already resolved is answered deterministically: a
   softmax over 22 classes is often *below threshold on its own training
   examples* when they are short (`lilt.`, `пл.`).
3. **Exclusions** — the model must stay silent (see below).
4. **Model** — softmax regression, only above its confidence threshold.

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
* **honest metrics** (5-fold CV on campaign forms the model never saw,
  restricted to forms the model is allowed to judge): **coverage 52.9% @
  precision 98.6%**, threshold 0.735 — it confidently resolves half of the
  unseen junk and stays silent on the rest;
* **on live data** (278 non-canonical unit rows from a week of production
  after training): v0.1 resolved 6.5%, v0.2 resolves **26.3%** — 54 rows
  by the exact table, 14 by the vocabulary after HTML stripping, 5 by the
  model; no row v0.1 resolved is lost or changed. The rest is mostly
  legitimate units outside the canon (`блістер`, `банка`), duals and
  low-confidence labels — not the model's job;
* **deliberate exclusions**: dual descriptors (`шт/уп`, `шт уп` — two
  units at once), strings with digits (`100 шт` is a quantity, not a
  unit), overlong or single-letter strings (`т` is a tonne), placeholders
  (`nan`, `null`), dosage forms (`порошок`, `розчин`, …), service markers
  (`дослідження`) and real units outside the canon (`набір`, `тонна`);
* **amounts are never dropped**: a labeled form that carries an amount
  (`100 шт`, `фл. 40мл`, `тис. доз`) is kept out of the exact table —
  resolving `5 тис. доз` to `доза` would silently rescale the quantity
  1000×. Digits *inside* a letter token (`д03` = OCR `доз`) are glyph
  confusion and stay.

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
(`pairs`: raw form → canonical label, the model's training set;
`vocabulary`: exact spellings; `exact`: reviewed labeled forms answered by
exact lookup; `canons`: the label set) and retrain — the artifact format
is self-contained. The notebook walks through the v0.1 pipeline; the
exact table and exclusions of v0.2 live in `train.py` / `classifier.py`.

## Changelog

**0.2.0**
- HTML tags are stripped before lookup and classification.
- Exact table of reviewed campaign forms in the artifact (`exact`, v2
  format); keys that map to different labels are dropped, forms carrying
  an amount are excluded.
- Exclusions extended (single letters, placeholders, dosage forms,
  service markers, units outside the canon) — found by an LLM-judge eval
  on live data.
- Training and CV use only forms the model is allowed to judge.
- Live residual coverage 6.5% → 26.3% with no regressions.

**0.1.0** — initial release.

## Canonical units (22)

МО, ампула, г, доза, капсула, картридж, кг, контейнер, л, мг, мл, пакет,
пара, пляшка, саше, таблетка, туба, упаковка, флакон, шприц, шприц-ручка,
штука

## License

MIT
