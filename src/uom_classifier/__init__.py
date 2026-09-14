"""uom-classifier — tiny Ukrainian unit-of-measure classifier for OCR noise.

>>> from uom_classifier import UomClassifier
>>> UomClassifier().classify("пакува rhh")
('упаковка', 0.95...)
"""

from .classifier import UomClassifier, extract_ngrams

__all__ = ["UomClassifier", "extract_ngrams"]
__version__ = "0.2.0"
