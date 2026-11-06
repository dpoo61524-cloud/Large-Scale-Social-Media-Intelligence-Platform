"""Misinformation and Fake-News Detection Engine (LIAR Benchmark).

Provides dataset ingestion, stylometric feature extraction, calibrated
classification, and social media post credibility scoring.
"""

from src.misinformation.dataset_loader import LiarDatasetLoader
from src.misinformation.feature_extractor import LinguisticFeatureExtractor
from src.misinformation.classifier import MisinformationClassifier
from src.misinformation.misinformation_pipeline import run_misinformation_pipeline

__all__ = [
    "LiarDatasetLoader",
    "LinguisticFeatureExtractor",
    "MisinformationClassifier",
    "run_misinformation_pipeline",
]
