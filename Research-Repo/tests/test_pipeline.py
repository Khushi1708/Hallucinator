"""
Unit Tests for Semantic Entropy & Verification System.
"""

import pytest
import numpy as np
from core.embedder import TextEmbedder
from core.clustering import SemanticClusterer
from core.entropy import SemanticEntropyCalculator
from core.verifier import FormalRuleVerifier, RAGVerifier
from core.pipeline import SemanticEntropyPipeline


def test_embedder():
    embedder = TextEmbedder()
    texts = ["Alexander Graham Bell invented the telephone.", "Berlin is the capital of Germany."]
    embeddings = embedder.embed(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_clustering_identical():
    clusterer = SemanticClusterer(similarity_threshold=0.7)
    embedder = TextEmbedder()
    responses = [
        "The capital of Germany is Berlin.",
        "Berlin is Germany's capital city.",
        "Berlin is the capital of Germany."
    ]
    embs = embedder.embed(responses)
    if embedder._use_fallback:
        pytest.skip("needs sentence-transformers; TF-IDF fallback can't match paraphrases")
    result = clusterer.cluster(responses, embs)
    assert len(result["clusters"]) == 1
    assert result["cluster_probabilities"][0] == 1.0


def test_clustering_divergent():
    clusterer = SemanticClusterer(similarity_threshold=0.6)
    embedder = TextEmbedder()
    responses = [
        "Alexander Graham Bell invented the telephone.",
        "The capital of Germany is Berlin.",
        "No humans have ever landed on Mars yet."
    ]
    embs = embedder.embed(responses)
    result = clusterer.cluster(responses, embs)
    assert len(result["clusters"]) >= 2


def test_entropy_zero():
    calc = SemanticEntropyCalculator()
    probs = {0: 1.0}
    res = calc.calculate(probs, total_samples=5)
    assert res["raw_entropy"] == 0.0
    assert res["normalized_entropy"] == 0.0
    assert res["confidence_score"] == 100.0
    assert res["uncertainty_level"] == "Low"


def test_entropy_high():
    calc = SemanticEntropyCalculator()
    # Uniform 4 clusters
    probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    res = calc.calculate(probs, total_samples=4)
    assert res["raw_entropy"] > 1.0
    assert res["normalized_entropy"] == 1.0
    assert res["uncertainty_level"] == "High"
    assert res["is_hallucination_suspect"] is True


def test_formal_rule_verifier():
    verifier = FormalRuleVerifier()
    
    # Consistent case
    res1 = verifier.verify("What is the capital of Germany?", "Berlin is the capital.")
    assert res1 is not None
    assert res1["is_consistent"] is True

    # Inconsistent / Hallucination case
    res2 = verifier.verify("What is the capital of Germany?", "Paris is the capital of Germany.")
    assert res2 is not None
    assert res2["is_consistent"] is False


def test_full_pipeline():
    pipeline = SemanticEntropyPipeline(generator_mode="simulation")
    result = pipeline.run("What is the capital of Germany?", n_samples=5)
    
    assert "question" in result
    assert "responses" in result
    assert "clustering" in result
    assert "entropy" in result
    assert "verification" in result
    assert "overall_status" in result
    assert result["overall_status"]["confidence"]


def test_pipeline_makes_no_extra_api_calls_by_default():
    pipeline = SemanticEntropyPipeline(generator_mode="simulation")
    result = pipeline.run("What is the capital of Germany?", n_samples=4)
    assert result["all_models"] == {}


def test_rag_path_does_not_crash_without_rule():
    pipeline = SemanticEntropyPipeline(generator_mode="simulation")
    result = pipeline.run("Who won the 2030 FIFA World Cup?", n_samples=6)
    assert 0.0 <= result["verification"]["evidence_similarity"] <= 1.0
