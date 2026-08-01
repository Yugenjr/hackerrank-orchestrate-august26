"""
Configuration module for the Retrieval Engine and overarching pipeline.
Supports future optimization and ablation testing by externalizing hardcoded weights.
"""
from typing import Dict

class RetrievalConfig:
    # Fusion Weights
    WEIGHT_BM25 = 0.3
    WEIGHT_FAISS = 0.4
    WEIGHT_RELATIONSHIP = 0.1
    WEIGHT_TRUST = 0.1
    WEIGHT_RECENCY = 0.1

    # Temporal Decay
    DECAY_HALF_LIFE_DAYS = 30.0  # Score halves every 30 days
    MIN_DECAY_SCORE = 0.1

    # Clustering
    CLUSTER_EPS = 0.15  # DBSCAN epsilon for semantic deduplication
    CLUSTER_MIN_SAMPLES = 1

    # Adaptive Thresholding Base
    BASE_CONFIDENCE_THRESHOLD = 0.50
    BUSINESS_THRESHOLD_MODIFIER = -0.10  # More lenient for business matches
    SPARSE_HISTORY_MODIFIER = +0.10     # Stricter if history is sparse

    @classmethod
    def get_weights(cls) -> Dict[str, float]:
        return {
            "bm25": cls.WEIGHT_BM25,
            "faiss": cls.WEIGHT_FAISS,
            "relationship": cls.WEIGHT_RELATIONSHIP,
            "trust": cls.WEIGHT_TRUST,
            "recency": cls.WEIGHT_RECENCY
        }
