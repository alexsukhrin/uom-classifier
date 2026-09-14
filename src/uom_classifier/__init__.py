"""uom-classifier — tiny Ukrainian unit-of-measure classifier for OCR noise.

>>> from uom_classifier import UomClassifier
>>> UomClassifier().classify("пакува rhh")
('упаковка', 0.9...)
>>> UomClassifier().classify("іпт", "Ксилол, каністра 5 л")  # product-name context (v0.4)
"""

from .classifier import KEEP_CLASS, UomClassifier, extract_context, extract_ngrams

__all__ = ["KEEP_CLASS", "UomClassifier", "extract_context", "extract_ngrams"]
__version__ = "0.4.0"
