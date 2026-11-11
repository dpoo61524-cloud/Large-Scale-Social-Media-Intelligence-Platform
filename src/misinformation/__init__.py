"""Misinformation and Fake-News Detection Engine (LIAR Benchmark Framework).

Provides dataset ingestion, stylometric feature extraction, calibrated
classification, transparent evidence-based risk assessment, and social media post credibility scoring.

Methodological & Benchmark Documentation:
1. The embedded 24-statement dataset is an offline development fixture designed
   for deterministic testing, not the official 12,836-sample LIAR benchmark.
2. The official benchmark has not been evaluated because the complete dataset
   has not been supplied.
3. Cross-validation results on small development fixtures are development
   diagnostics, not production-quality performance estimates.
4. Keyword-based subjectivity features are English-oriented and context-dependent.
5. Model-estimated credibility scores represent statistical pattern matching,
   not independent factual verification.
6. Binary ('fake' vs 'real') and six-class fine-grained veracity tiers have
   distinct target label interpretations.
7. Subjective opinions, personal experiences, questions, emotional expression,
   and negative sentiment are protected from false misinformation labeling.
"""

from src.misinformation.classifier import MisinformationClassifier
from src.misinformation.dataset_loader import LiarDatasetLoader
from src.misinformation.detector import MisinformationDetector, determine_risk_tier
from src.misinformation.feature_extractor import LinguisticFeatureExtractor
from src.misinformation.misinformation_pipeline import (
    evaluate_cross_validated,
    load_and_classify_source,
    run_misinformation_pipeline,
)
from src.misinformation.misinformation_validator import MisinformationArtifactValidator

__all__ = [
    "LiarDatasetLoader",
    "LinguisticFeatureExtractor",
    "MisinformationArtifactValidator",
    "MisinformationClassifier",
    "MisinformationDetector",
    "determine_risk_tier",
    "evaluate_cross_validated",
    "load_and_classify_source",
    "run_misinformation_pipeline",
]
