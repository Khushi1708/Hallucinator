"""
Command-Line Interface for Semantic Entropy & Verification System.
Usage:
    python main.py --question "Who invented the telephone?"
    python main.py --question "What is the capital of Germany?" --samples 8
"""

import sys
import argparse
import json

# Ensure stdout supports unicode on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.pipeline import SemanticEntropyPipeline


def main():
    parser = argparse.ArgumentParser(description="LLM Semantic Entropy Hallucination Detection & Verification CLI")
    parser.add_argument("--question", "-q", type=str, default="Who invented the telephone?", help="Question to analyze")
    parser.add_argument("--samples", "-n", type=int, default=8, help="Number of sampled LLM responses")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--mode", "-m", type=str, default="simulation", choices=["simulation", "openai", "chatgpt", "gemini", "kimi", "huggingface"], help="Generation backend")
    parser.add_argument("--model", type=str, default=None, help="Specific model name (e.g., gpt-4o-mini, gemini-flash-latest, moonshot-v1-8k)")
    parser.add_argument("--api-key", type=str, default=None, help="Custom API key")
    parser.add_argument("--threshold", type=float, default=0.75, help="Clustering similarity threshold")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("[*] Semantic Entropy & Verification Pipeline")
    print("=" * 70)
    print(f"Question: {args.question}")
    print(f"Samples:  {args.samples} (Mode: {args.mode}, Temp: {args.temperature})")
    print("=" * 70 + "\n")

    pipeline = SemanticEntropyPipeline(
        generator_mode=args.mode,
        model_name=args.model,
        api_key=args.api_key,
        similarity_threshold=args.threshold
    )

    result = pipeline.run(
        question=args.question,
        n_samples=args.samples,
        temperature=args.temperature
    )

    if args.json:
        # Filter out non-serializable objects (numpy arrays)
        serializable_result = {
            "question": result["question"],
            "responses": result["responses"],
            "entropy": result["entropy"],
            "verification": result["verification"],
            "overall_status": result["overall_status"]
        }
        print(json.dumps(serializable_result, indent=2))
        return

    # Formatted terminal display
    print("--- [Step 2] Sampled LLM Responses ---")
    for i, r in enumerate(result["responses"]):
        cluster_id = result["clustering"]["cluster_labels"][i]
        print(f"  [{i+1}] (Cluster {cluster_id+1}): {r}")

    print("\n--- [Step 4] Semantic Clusters ---")
    for cid, indices in result["clustering"]["clusters"].items():
        prob = result["clustering"]["cluster_probabilities"][cid]
        rep = result["clustering"]["cluster_representatives"][cid]
        print(f"  Cluster {cid+1} ({len(indices)} items, {prob*100:.1f}%): \"{rep}\"")

    ent = result["entropy"]
    print("\n--- [Step 5] Semantic Entropy & Uncertainty ---")
    print(f"  Raw Shannon Entropy:      {ent['raw_entropy']}")
    print(f"  Normalized Entropy:       {ent['normalized_entropy']} ({ent['uncertainty_level']} Uncertainty)")
    print(f"  Model Confidence:         {ent['confidence_score']}%")
    print(f"  Entropy Risk Status:      {ent['risk_status']}")

    ver = result["verification"]
    print("\n--- [Step 6] Verification Layer (RAG & Formal Logic) ---")
    print(f"  Engine:                   {ver['verification_source']}")
    print(f"  Status:                   {ver['verification_status']}")
    print(f"  Evidence Match:           {ver['evidence_similarity']*100:.1f}%")
    print(f"  Evidence Passage:         \"{ver['evidence_extract'][:120]}...\"")

    ov = result["overall_status"]
    print("\n" + "=" * 70)
    print("[*] FINAL SYSTEM OUTPUT [Step 7]")
    print("=" * 70)
    print(f"  Answer:                   \"{ov['answer']}\"")
    print(f"  Confidence:               {ov['confidence']}")
    print(f"  Semantic Entropy:         {ov['semantic_entropy']}")
    print(f"  Status:                   {ov['status']}")
    print(f"  Verified Ground Truth:    \"{ov['verified_answer']}\"")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
