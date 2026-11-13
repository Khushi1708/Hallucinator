"""
Semantic Entropy Calculator Module.
Computes discrete semantic entropy over semantic cluster distributions,
uncertainty metrics, and hallucination risk indicators based on Farquhar et al. (Nature 2024).
"""

import math
from typing import Dict, Any


class SemanticEntropyCalculator:
    """
    Computes Shannon Semantic Entropy and Hallucination Risk Scores.
    """

    def __init__(self, low_threshold: float = 0.35, high_threshold: float = 0.65):
        """
        :param low_threshold: Normalized entropy below this is Low Risk.
        :param high_threshold: Normalized entropy above this is High Risk.
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def calculate(self, cluster_probabilities: Dict[int, float], total_samples: int) -> Dict[str, Any]:
        """
        Calculate raw entropy, normalized entropy, confidence score, and risk status.
        """
        if not cluster_probabilities or total_samples <= 1:
            return {
                "raw_entropy": 0.0,
                "normalized_entropy": 0.0,
                "max_possible_entropy": 0.0,
                "confidence_score": 100.0,
                "uncertainty_level": "Low",
                "risk_status": "Low Uncertainty (Consistent)",
                "is_hallucination_suspect": False,
            }

        # Shannon discrete entropy: H = - sum(p * ln(p))
        raw_entropy = 0.0
        for p in cluster_probabilities.values():
            if p > 0:
                raw_entropy -= p * math.log(p)

        # Max theoretical entropy for N uniform samples
        max_possible_entropy = math.log(total_samples) if total_samples > 1 else 1.0

        # Normalized entropy bounded in [0, 1]
        normalized_entropy = raw_entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0
        normalized_entropy = max(0.0, min(1.0, normalized_entropy))

        # Confidence percentage
        confidence_score = round((1.0 - normalized_entropy) * 100.0, 1)

        # Determine risk classification
        if normalized_entropy < self.low_threshold:
            uncertainty_level = "Low"
            risk_status = "High Confidence / Consistent Consensus"
            is_hallucination_suspect = False
        elif normalized_entropy < self.high_threshold:
            uncertainty_level = "Moderate"
            risk_status = "Moderate Uncertainty (Divergent Clusters)"
            is_hallucination_suspect = True
        else:
            uncertainty_level = "High"
            risk_status = "⚠ High Hallucination Risk (High Semantic Entropy)"
            is_hallucination_suspect = True

        return {
            "raw_entropy": round(raw_entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "max_possible_entropy": round(max_possible_entropy, 4),
            "confidence_score": confidence_score,
            "uncertainty_level": uncertainty_level,
            "risk_status": risk_status,
            "is_hallucination_suspect": is_hallucination_suspect,
        }
