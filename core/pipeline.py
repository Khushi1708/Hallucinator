"""
Semantic Entropy & Verification Master Pipeline.
Orchestrates the entire 7-step pipeline from raw question to verified hallucination audit.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .generator import ResponseGenerator
from .embedder import TextEmbedder
from .clustering import SemanticClusterer
from .entropy import SemanticEntropyCalculator
from .verifier import RAGVerifier


class SemanticEntropyPipeline:
    """
    End-to-end pipeline implementing Semantic Entropy calculation and RAG verification.
    """

    def __init__(
        self,
        generator_mode: str = "simulation",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        similarity_threshold: float = 0.75,
        entropy_low_threshold: float = 0.35,
        entropy_high_threshold: float = 0.65
    ):
        self.generator = ResponseGenerator(mode=generator_mode, model_name=model_name, api_key=api_key)
        self.embedder = TextEmbedder()
        self.clusterer = SemanticClusterer(similarity_threshold=similarity_threshold)
        self.entropy_calculator = SemanticEntropyCalculator(
            low_threshold=entropy_low_threshold,
            high_threshold=entropy_high_threshold
        )
        self.verifier = RAGVerifier(embedder=self.embedder)

    def run(
        self,
        question: str,
        n_samples: int = 8,
        temperature: float = 0.8,
        custom_samples: Optional[List[str]] = None,
        compare_models: bool = False,
        api_keys: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full 7-step workflow:
        Step 1: User Input
        Step 2: Generate Multiple LLM Responses
        Step 3: Convert Responses into Embeddings
        Step 4: Semantic Similarity & Clustering
        Step 5: Calculate Semantic Entropy
        Step 6: Verification Layer (RAG & Rules)
        Step 7: Final Synthesized Output
        """
        # Step 1: User Input
        q = question.strip()

        # Step 2: Generate Multiple LLM Responses
        if custom_samples and len(custom_samples) > 0:
            responses = custom_samples
        else:
            responses = self.generator.generate(q, n_samples=n_samples, temperature=temperature)

        # Step 3: Convert Responses into Embeddings
        embeddings = self.embedder.embed(responses)

        # Step 4: Semantic Similarity & Clustering
        clustering_result = self.clusterer.cluster(responses, embeddings)

        # Step 5: Calculate Semantic Entropy
        entropy_result = self.entropy_calculator.calculate(
            clustering_result["cluster_probabilities"],
            total_samples=len(responses)
        )

        # Identify dominant answer (Cluster 0 representative)
        dominant_cluster_id = 0 if clustering_result["cluster_representatives"] else None
        dominant_answer = clustering_result["cluster_representatives"].get(0, responses[0] if responses else "")

        # Step 6: Add Verification Layer (RAG / Rules)
        verification_result = self.verifier.verify_answer(
            question=q,
            dominant_answer=dominant_answer,
            entropy_metrics=entropy_result
        )

        # Generate live answers from all 3 AI models (ChatGPT, Gemini, Kimi)
        # Opt-in: this makes 3 extra paid API calls, so it is off by default.
        all_models_data = (
            self.generator.generate_all_three(q, n_per_model=1, temperature=temperature, api_keys=api_keys)
            if compare_models else {}
        )

        # Step 7: Synthesize Final Output
        final_summary = {
            "question": q,
            "n_samples": len(responses),
            "responses": responses,
            "clustering": clustering_result,
            "entropy": entropy_result,
            "dominant_answer": dominant_answer,
            "verification": verification_result,
            "all_models": all_models_data,
            "generator_warnings": list(self.generator.warnings),
            "overall_status": {
                "answer": dominant_answer,
                "confidence": f"{entropy_result['confidence_score']}%",
                "semantic_entropy": entropy_result["uncertainty_level"],
                "raw_entropy": entropy_result["raw_entropy"],
                "normalized_entropy": entropy_result["normalized_entropy"],
                "status": "⚠ Possible Hallucination Detected" if (entropy_result["is_hallucination_suspect"] or verification_result["is_hallucination"]) else "✓ Low Uncertainty / Factual",
                "verified_answer": verification_result.get("ground_truth_answer", dominant_answer),
                "evidence_source": verification_result.get("evidence_title", "N/A"),
                "evidence_url": verification_result.get("evidence_url", "")
            }
        }

        return final_summary

