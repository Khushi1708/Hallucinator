# 🧠 Semantic Entropy & Verification Layer for LLM Hallucination Detection

An end-to-end framework for detecting hallucinations and estimating uncertainty in Large Language Models using **Semantic Entropy (Farquhar et al., Nature 2024)** combined with a novel **RAG & Formal Rule Verification Layer**.

---

## 📌 Complete 7-Step Workflow

```
Step 1: User Input
   │  "Who invented the telephone?"
   ▼
Step 2: Generate Multiple LLM Responses (N = 5–10 samples at τ = 0.8)
   │  R1: "Alexander Graham Bell invented the telephone."
   │  R2: "Antonio Meucci developed an early telephone."
   │  R3: "Alexander Graham Bell was awarded the first telephone patent."
   ▼
Step 3: Convert Responses into Embeddings
   │  Dense numerical vectors via Sentence-Transformers / all-MiniLM-L6-v2
   ▼
Step 4: Semantic Similarity Calculation & Clustering
   │  Pairwise Cosine Similarity: S_ij = (v_i · v_j) / (||v_i|| ||v_j||)
   │  Agglomerative Clustering:
   │    • Cluster 1: Bell invented telephone (75%)
   │    • Cluster 2: Meucci invented telephone (25%)
   ▼
Step 5: Calculate Semantic Entropy
   │  Shannon Semantic Entropy: H(S) = - ∑ P(C_k) ln P(C_k)
   │  Normalized Entropy: SE_norm = H(S) / ln(N)
   │  Confidence = (1 - SE_norm) * 100%
   ▼
Step 6: Add Verification Layer (Novel Contribution)
   │  • RAG Verification: Live Wikipedia Knowledge Search & Cosine Cross-Check
   │  • Formal Verification: Deterministic relation checking against ground-truth facts
   ▼
Step 7: Final Synthesized Output & Audit
   │  Answer: "Alexander Graham Bell invented the telephone."
   │  Confidence: 78.5%
   │  Semantic Entropy: Moderate Uncertainty
   │  Status: ✓ Verified Factual / ⚠ Possible Hallucination
   │  Verified Ground Truth: "Alexander Graham Bell was awarded the first US patent..."
```

---

## 🛠️ Installation

```bash
# 1. Clone repository & enter directory
cd <your-project-folder>

# 2. Install required dependencies
pip install -r requirements.txt   # use requirements-dev.txt to also get pytest
```

---

## 🔑 API Configuration

Copy `.env.example` to `.env` and fill in only the keys you need. **Never commit `.env`** (it is in `.gitignore`).

```env
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
KIMI_API_KEY=your-kimi-key
KIMI_BASE_URL=https://api.moonshot.cn/v1
```

For a public deployment keep `ALLOW_SERVER_KEYS=false` so visitors supply their own keys. See [DEPLOY.md](DEPLOY.md).

Supported LLM generation backends:
1. **ChatGPT (OpenAI)**: `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4o`
2. **Google Gemini**: `gemini-flash-latest`, `gemini-2.5-flash` (via the `google-genai` SDK)
3. **Kimi AI (Moonshot)**: `moonshot-v1-8k`, `moonshot-v1-32k`
4. **Offline Benchmarks / Simulation Mode**: Instant deterministic runs without incurring API quota.
5. **Hugging Face Local Models**: `HuggingFaceTB/SmolLM-135M-Instruct`.

---

## 🚀 Running the Project

### 1. Interactive Streamlit Web Dashboard
Launch the visual dashboard with 2D cluster maps, heatmap matrices, entropy gauges, and RAG verification cards:
```bash
streamlit run app.py
```
*(You can switch between ChatGPT, Gemini, Kimi, and Simulation directly from the sidebar).*

### 2. Command-Line Interface (CLI)
```bash
# Run with ChatGPT (OpenAI)
python main.py --question "Who invented the telephone?" --mode chatgpt

# Run with Google Gemini
python main.py --question "What is the capital of Germany?" --mode gemini

# Run with Kimi AI (Moonshot)
python main.py --question "Who discovered penicillin?" --mode kimi

# Run with offline simulation benchmark
python main.py --question "What was the first country to land a human on Mars?" --mode simulation
```

### 3. Python API Integration
```python
from core.pipeline import SemanticEntropyPipeline

pipeline = SemanticEntropyPipeline(generator_mode="simulation")
result = pipeline.run("Who invented the telephone?", n_samples=8)

print("Dominant Answer:", result["dominant_answer"])
print("Confidence:", result["entropy"]["confidence_score"])
print("Verified Ground Truth:", result["verification"]["ground_truth_answer"])
```

---

## 🧪 Running Automated Tests

```bash
pytest tests/
# or
python -m pytest tests/test_pipeline.py -v
```

---

## 📐 Mathematical Formulation

### 1. Pairwise Semantic Similarity
Given normalized embedding vectors $\mathbf{v}_i, \mathbf{v}_j \in \mathbb{R}^d$:
$$\text{Sim}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\|_2 \|\mathbf{v}_j\|_2}$$

### 2. Cluster Probability Distribution
For $K$ semantic clusters $\{C_1, C_2, \dots, C_K\}$ over $N$ sampled responses:
$$P(C_k) = \frac{|C_k|}{N}$$

### 3. Shannon Semantic Entropy
$$SE = -\sum_{k=1}^{K} P(C_k) \ln P(C_k)$$

### 4. Normalized Entropy & Confidence
$$SE_{norm} = \frac{SE}{\ln(N)} \in [0, 1]$$
$$\text{Confidence Score} = (1 - SE_{norm}) \times 100\%$$

---

## 📂 Project Structure

```
khushi project/
├── app.py                   # Streamlit Interactive Web Application
├── main.py                  # CLI pipeline runner
├── requirements.txt         # Dependencies
├── README.md                # Documentation & Equations
├── core/
│   ├── __init__.py          # Module exports
│   ├── generator.py         # Multi-sample LLM generator (Offline/HF/OpenAI)
│   ├── embedder.py          # Sentence-Transformers / TF-IDF embedder
│   ├── clustering.py        # Semantic similarity matrix & Agglomerative clustering
│   ├── entropy.py           # Shannon Semantic Entropy & Confidence score computation
│   ├── verifier.py          # RAG (Wikipedia) & Formal Rule Verification Layer
│   └── pipeline.py          # 7-step unified master pipeline
└── tests/
    └── test_pipeline.py     # Comprehensive unit tests
```
