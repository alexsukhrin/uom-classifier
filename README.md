# uom-classifier

Tiny, **dependency-free** classifier that maps noisy OCR'd Ukrainian
unit-of-measure strings to canonical units — since v0.4 it also reads the
**product name** of the same specification row.

```python
from uom_classifier import UomClassifier

clf = UomClassifier()

clf.classify("пакува rhh")                      # ('упаковка', 0.9…)  ← OCR garble, model
clf.classify("пф", "Цефотрин пор. д/ін. 1г фл.")  # ('флакон', 0.8…)   ← name context helps
clf.classify("шт.")                             # ('штука', 1.0)      ← exact vocabulary hit
clf.classify("lilt.")                           # ('штука', 1.0)      ← exact reviewed form
clf.classify("шт/фл")                           # ('флакон', 1.0)     ← «штука» + one specific unit
clf.classify("<b>шт</b>")                       # ('штука', 1.0)      ← HTML stripped first
clf.classify("шт/уп")                           # None                ← unit + package dual, excluded
clf.classify("тис. доз")                        # None                ← carries an amount
clf.classify("порошок")                         # None                ← dosage form, not a unit
clf.classify("garbage")                         # None                ← __keep__ class / below threshold
```

`None` always means *"leave the value as it is"*. The product name is
optional (`classify(raw)` still works) and never overrides an exact answer.

## How a string is resolved

1. **Clean** — HTML tags from the PDF text layer are stripped, whitespace
   collapsed.
2. **Exact lookup** — the vocabulary of common spellings, then the reviewed
   labeled forms (`exact`: 599 keys in the artifact, built from 854 reviewed
   forms minus vocabulary overlaps, amount-carrying forms and duals).
   Slash/paren forms whose halves name two different units («пач (штука)»)
   never enter the table.
3. **Pair rule** — «штука» plus exactly one specific unit-class form
   («шт/фл», «шт. (амп.)») → that unit; two halves of the same canon
   («уп/упаковка») → that canon. Package pairs («шт/уп», «штука/контейнер»)
   stay duals: they carry the price basis downstream.
4. **Exclusions** — the model must stay silent: slash and spaced duals,
   forms that carry an amount (a token starting with a digit or a
   multiplier — digits *inside* a letter token such as «д03», «уг1» are
   glyph confusion and are judged), one-letter strings, placeholders, dosage
   forms, service markers, units outside the model's classes.
5. **Model** — softmax regression over unit-string char n-grams (1..4) plus
   product-name features, with an explicit `__keep__` class. It answers only
   when its best class is a unit and the confidence clears the threshold
   (0.78).

Product-name features: words as `w:<first 5 letters>` and ~30 curated cues
(`q:tablet`, `q:vol_l`, `q:count_no`, `q:reagent`, …). The context vector is
L2-normalized on its own and scaled by 0.5, so the unit string keeps the
dominant voice. Training uses context dropout (35% of row copies without the
name), so a missing name degrades to the form-only model.

## Data (v0.4)

Real rows of Ukrainian public-procurement specifications (Prozorro), mined
read-only from a production data-quality copy:

* **gold set (held out)** — 464 rows / 246 forms, stratified by production
  row volume (all forms sampled per volume tier: ≥20 rows 50%, 10–19 50%,
  4–9 40%, 2–3 35%, 1 row 20%; ≤6/4/3/2/1 rows per form), **held out by form
  key** from every training source (pairs, exact table, context rows).
  Labels: two independent judges (gpt-5.5, gpt-4o) saw the raw form, the
  product name and up to 4 sibling names — never an existing label; agreement
  on 255/464 (55%); the 209 disagreements went to a third judge (o3), who
  sided with gpt-5.5 in 132, gpt-4o in 34 and neither in 43. **Headline rows:
  421** (255 unanimous + 166 majority) — 243 canonical, 178 `KEEP`
  (unreadable 79, dual 57, not-a-unit 22, foreign unit 17, amount 15); **43
  contested rows** (no majority, 7.3% of weight) are excluded and reported.
  The file is frozen (`data/gold.jsonl`, sha256 `627240f6…`) before any
  training.
* **training rows** — 1,406 deduplicated (form, product name) rows (weight
  1,432): 436 rows labeled by the consensus of gpt-5.5 and o3, 477 more rows
  of forms whose consensus rows were unanimous, 406 rows with high-confidence
  campaign labels, 113 consensus rows of OCR forms from the extraction stage
  that never reached the item table. Judge agreement on training rows 74.3%
  (items) and 83.1% (extraction forms); disagreements are not trained on.
  481 rows are `KEEP`.
* 755 form-only pairs and 125 vocabulary spellings (as in v0.3), 4,617
  dictionary-labeled context rows (clean forms with product names).
* LLM budget: 2,747 calls in total (gold 1,143; training 1,604).

## Metric

A gold row is **correct** when the answer is its canonical label, or
silence when the label is `KEEP`; **wrong** when a canon is answered that is
not the label (any canon on a `KEEP` row); silence on a canonical label is a
**miss** (neither). `accuracy = correct / rows`, weighted by the production
row volume each gold row stands for; also reported form-weighted.

## Results

Evaluated through the downstream service's resolution path
(dictionary → this classifier with the product name).

| | Gold accuracy | Gold wrong | Form-weighted acc / wrong |
|---|---:|---:|---:|
| v0.3.0 as shipped (its exact table contains many gold forms — partly in-sample) | 79.6% | 9.85% | 76.2% / 12.0% |
| v0.3.0 recipe, gold forms held out | 66.3% | 4.04% | 62.3% / 4.8% |
| **v0.4.0, gold forms held out** | **72.7%** | **2.88%** | 65.7% / 4.25% |
| v0.4.0 shipped (trained on gold labels too — in-sample, not a metric) | 94.9% | 1.16% | — |

Held-out path breakdown (share of gold weight): pair rule 3.3% correct,
model 17.2% correct / 2.9% wrong, silence 51.5% correct / 24.4% miss,
dictionary after HTML stripping 0.7%.

Context ablation (same data v0.4-rc, gold held out): context 0.5 →
71.2% / 4.17%, no context → 70.1% / 4.71%; grouped CV +4–6 points on every
seed.

**Multi-seed CV** (5 folds grouped by form key, gold held out, the whole
resolution path simulated per fold, threshold tuned on the out-of-fold
predictions of the first seed with a wrong-rate cap of 1.5%):

| seed | accuracy | wrong |
|---|---:|---:|
| 24948 | 65.9% | 1.47% |
| 1 | 63.6% | 1.96% |
| 2 | 63.6% | 1.82% |

The 95% / ≤2% target on unseen forms is **not met**. What blocks it (held-out
gold, share of weight):

| class | share | of which majority-only labels |
|---|---:|---:|
| model silent on an unseen form (`__keep__` / below threshold) | 18.0% | 50% |
| canon label on a unit+package dual («штука/контейнер» → контейнер) — excluded by policy | 3.0% | 60% |
| model wrong (mostly «іт» → штука vs `KEEP`) | 2.9% | 47% |
| canon label on a form carrying an amount («мг (по 250 мг)») | 1.6% | 88% |
| other exclusions (one letter «ш», dosage form «спрей») | 1.0% | 43% |
| dictionary-only canon («бан» → банка; no model class) | 0.8% | 0% |

Even a perfect model under the current exclusions would reach **93.6%**
on held-out forms. The two judges themselves agree on 55% of gold rows.

On production traffic most forms recur: with the reviewed forms in the exact
table, the shipped artifact resolves 54.7% of a week of live non-canonical
rows (v0.3: 41.4%) and loses none of the gold rows v0.3 answered correctly.

## Install

```bash
pip install git+https://github.com/alexsukhrin/uom-classifier
```

## Retrain

```bash
pip install "uom-classifier[train]"   # numpy
PYTHONPATH=src python -m uom_classifier.train \
    --dataset data/uom_training_pairs.json --gold data/gold.jsonl \
    --out src/uom_classifier/data/uom_classifier.json
```

The trainer writes the shipped artifact and `<out>_heldout.json` (gold forms
excluded everywhere) — evaluate the latter for honest numbers.

Dataset format (`data/uom_training_pairs.json`): `canons`, `vocabulary`,
`pairs` (form → label, no context), `exact` (reviewed forms), `rows`
(form, product name, label, weight = production rows, source),
`context_rows` (dictionary-labeled clean forms with names). `KEEP` and units
outside the model's classes train the `__keep__` class.

## Changelog

**0.4.0**
- Product-name context features (artifact v3, `context_scale`), context
  dropout in training; `classify(raw, product_name=None)` stays backward
  compatible.
- Explicit `__keep__` class trained on judge-labeled garbage, duals and
  units outside the canon — the model learns to stay silent, not only the
  threshold.
- New data: 1,406 labeled production rows with product names (two-judge
  consensus), frozen held-out gold set (464 rows), exact table 831 → 854
  (48 campaign labels contradicted by the consensus dropped, 71 added).
- Pair rule («шт/фл» → флакон, «уп/упаковка» → упаковка); unit+package
  duals never enter the exact table.
- Digits inside a letter token no longer exclude a form — only forms that
  carry an amount do.
- Sparse full-batch Adam training; form-grouped multi-seed CV that
  simulates the whole resolution path; threshold tuned on validation only.

**0.3.0** — 24 forms promoted by an independent second judge; blister/jar/
canister stay out of the model.

**0.2.0** — HTML stripping, exact table of reviewed forms (artifact v2),
service exclusions.

**0.1.0** — initial release.

## Canonical units (22 model classes)

МО, ампула, г, доза, капсула, картридж, кг, контейнер, л, мг, мл, пакет,
пара, пляшка, саше, таблетка, туба, упаковка, флакон, шприц, шприц-ручка,
штука — plus `__keep__`.

## License

MIT
